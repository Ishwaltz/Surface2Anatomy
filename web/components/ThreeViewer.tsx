"use client";

import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import {
  createHumanMannequin,
  createDepthSensorRig,
  createPhotoBillboardRig,
} from "@/utils/mannequinBuilder";
import { Eye, EyeOff, Layers, RotateCcw, Sparkles, X, Focus } from "lucide-react";

export interface TargetPrediction {
  target: string;
  target_index: number;
  centroid_canonical_mm: [number, number, number];
  centroid_input_world_mm: [number, number, number];
  uncertainty_mm: number;
  uncertainty_level: "low" | "moderate" | "high";
  seed_predictions_mm: Record<string, [number, number, number]>;
}

interface ThreeViewerProps {
  surfacePoints: number[][] | null; // (N, 3) in mm
  predictions: Record<string, TargetPrediction> | null;
  selectedTarget: string | null;
  onSelectTarget?: (target: string | null) => void;
  coordinateFrame?: string;
  modality?: "mesh" | "depth" | "rgb";
  patientSex?: "female" | "male" | "auto";
  imageUrl?: string | null;
}

/**
 * Maps medical CT coordinates (X: Lat, Y: AP, Z: Height)
 * to standard Three.js upright coordinates:
 * - Three.js X = med[0] (Right + / Left -)
 * - Three.js Y = med[2] (Height: Head + / Feet -)
 * - Three.js Z = med[1] (Anterior + / Posterior -)
 */
function toThreeCoord(med: [number, number, number]): [number, number, number] {
  // In standard anterior anatomical view: anatomical Right (+med[0]) appears on viewer's LEFT (-Three.js X)
  return [-med[0], med[2], med[1]];
}

export function ThreeViewer({
  surfacePoints,
  predictions,
  selectedTarget,
  onSelectTarget,
  modality = "mesh",
  patientSex = "female",
  imageUrl = null,
}: ThreeViewerProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const pointsMeshRef = useRef<THREE.Points | null>(null);
  const targetGroupRef = useRef<THREE.Group | null>(null);
  const mannequinGroupRef = useRef<THREE.Group | null>(null);
  const depthSensorGroupRef = useRef<THREE.Group | null>(null);
  const photoBillboardGroupRef = useRef<THREE.Group | null>(null);
  const imagePlaneMeshRef = useRef<THREE.Mesh | null>(null);
  const textureLoadTokenRef = useRef<number>(0);
  const reqIdRef = useRef<number | null>(null);

  // Interaction & Display State
  const [showMannequin, setShowMannequin] = useState(true);
  const [showSensorRig, setShowSensorRig] = useState(true);
  const [showImagePlane, setShowImagePlane] = useState(true);
  const [showPoints, setShowPoints] = useState(true);
  const [isolateSelected, setIsolateSelected] = useState(true);

  // Sync refs for event listeners and animation loop
  const selectedTargetRef = useRef(selectedTarget);
  selectedTargetRef.current = selectedTarget;
  const onSelectTargetRef = useRef(onSelectTarget);
  onSelectTargetRef.current = onSelectTarget;
  const isolateSelectedRef = useRef(isolateSelected);
  isolateSelectedRef.current = isolateSelected;

  // Manual orbit controls variables (Spherical coordinates)
  const isDraggingRef = useRef(false);
  const mouseDownPosRef = useRef({ x: 0, y: 0 });
  const previousMousePositionRef = useRef({ x: 0, y: 0 });
  const cameraSphericalRef = useRef({ radius: 950, theta: Math.PI / 4, phi: Math.PI / 2.3 });
  const targetCenterRef = useRef(new THREE.Vector3(0, 80, 0)); // Center camera on mid-torso

  const updateCamera = () => {
    if (!cameraRef.current) return;
    const { radius, theta, phi } = cameraSphericalRef.current;
    const x = targetCenterRef.current.x + radius * Math.sin(phi) * Math.sin(theta);
    const y = targetCenterRef.current.y + radius * Math.cos(phi);
    const z = targetCenterRef.current.z + radius * Math.sin(phi) * Math.cos(theta);
    cameraRef.current.position.set(x, y, z);
    cameraRef.current.lookAt(targetCenterRef.current);
  };

  // Smoothly center camera when a target organ is selected
  useEffect(() => {
    if (selectedTarget && predictions && predictions[selectedTarget]) {
      const pred = predictions[selectedTarget];
      const coords = pred.centroid_canonical_mm ?? pred.centroid_input_world_mm;
      const [tx, ty, tz] = toThreeCoord(coords);
      targetCenterRef.current.set(tx * 0.35, ty, tz * 0.35);
      updateCamera();
    } else {
      targetCenterRef.current.set(0, 80, 0);
      updateCamera();
    }
  }, [selectedTarget, predictions]);

  // Initialize Three.js Scene
  useEffect(() => {
    if (!mountRef.current) return;
    const container = mountRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf5f5f0); // Website background #F5F5F0
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, width / height, 1, 5000);
    camera.up.set(0, 1, 0); // Y is UP
    cameraRef.current = camera;
    updateCamera();

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    rendererRef.current = renderer;

    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    // Studio Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.7);
    dirLight1.position.set(300, 600, 400);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xb8c2ad, 0.45);
    dirLight2.position.set(-300, -200, -300);
    scene.add(dirLight2);

    // Grid Floor below feet
    const gridHelper = new THREE.GridHelper(1000, 24, 0xb8c2ad, 0xdce2d5);
    gridHelper.position.y = -490;
    scene.add(gridHelper);


    // 1. Mannequin Group (For 3D Mesh Scans)
    const mannequin = createHumanMannequin();
    scene.add(mannequin);
    mannequinGroupRef.current = mannequin;

    // 2. Depth Sensor Rig Group (For RealSense / Azure Kinect Depth Frames)
    const depthSensor = createDepthSensorRig();
    scene.add(depthSensor);
    depthSensorGroupRef.current = depthSensor;

    // 3. Photo Billboard Rig Group (For Clinical Photographs)
    const photoBillboard = createPhotoBillboardRig();
    scene.add(photoBillboard);
    photoBillboardGroupRef.current = photoBillboard;

    // 4. Anatomical Target Pins Group
    const targetGroup = new THREE.Group();
    scene.add(targetGroup);
    targetGroupRef.current = targetGroup;

    // Animation Loop with Smooth Periodic Pulsing for Selected Pin
    const animate = () => {
      reqIdRef.current = requestAnimationFrame(animate);
      const time = performance.now() * 0.001;

      if (targetGroupRef.current) {
        targetGroupRef.current.children.forEach((child: any) => {
          // Periodically pulsate selected organ sphere
          if (child.userData?.isPulsing) {
            const pulse = (Math.sin(time * 5.5) + 1.0) * 0.5; // 0 to 1
            if (child.material && child.material.emissiveIntensity !== undefined) {
              child.material.emissiveIntensity = 0.8 + 1.4 * pulse;
            }
            const s = 1.0 + 0.22 * pulse;
            child.scale.set(s, s, s);
          }
          // Expanding radiating beacon halo shockwave
          if (child.userData?.isBeacon) {
            const beaconPhase = (time * 1.5) % 1.0;
            const bScale = 1.0 + beaconPhase * 2.5;
            child.scale.set(bScale, bScale, bScale);
            if (child.material) {
              child.material.opacity = Math.max(0, (1.0 - beaconPhase) * 0.75);
            }
          }
        });
      }

      renderer.render(scene, camera);
    };
    animate();

    // Raycaster for Direct 3D Pin Clicking
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onMouseDown = (e: MouseEvent) => {
      isDraggingRef.current = true;
      mouseDownPosRef.current = { x: e.clientX, y: e.clientY };
      previousMousePositionRef.current = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isDraggingRef.current) return;
      const deltaX = e.clientX - previousMousePositionRef.current.x;
      const deltaY = e.clientY - previousMousePositionRef.current.y;

      cameraSphericalRef.current.theta -= deltaX * 0.007;
      cameraSphericalRef.current.phi = Math.max(
        0.05,
        Math.min(Math.PI - 0.05, cameraSphericalRef.current.phi - deltaY * 0.007)
      );

      previousMousePositionRef.current = { x: e.clientX, y: e.clientY };
      updateCamera();
    };

    const onMouseUp = (e: MouseEvent) => {
      isDraggingRef.current = false;
      const dist = Math.hypot(
        e.clientX - mouseDownPosRef.current.x,
        e.clientY - mouseDownPosRef.current.y
      );

      // If user clicked without dragging, raycast to select pin in 3D
      if (dist < 6 && targetGroupRef.current && cameraRef.current && container) {
        const rect = container.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(mouse, cameraRef.current);

        const intersects = raycaster.intersectObjects(targetGroupRef.current.children, true);
        if (intersects.length > 0) {
          const hit = intersects.find((i) => i.object.userData?.targetName);
          if (hit && hit.object.userData?.targetName) {
            const organName = hit.object.userData.targetName;
            if (onSelectTargetRef.current) {
              const current = selectedTargetRef.current;
              onSelectTargetRef.current(current === organName ? null : organName);
            }
          }
        }
      }
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      cameraSphericalRef.current.radius = Math.max(
        200,
        Math.min(2200, cameraSphericalRef.current.radius + e.deltaY * 0.75)
      );
      updateCamera();
    };

    const handleResize = () => {
      if (!container || !renderer || !camera) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (w <= 0 || h <= 0) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };

    const resizeObserver = new ResizeObserver(() => {
      handleResize();
    });
    resizeObserver.observe(container);

    // Touch Controls for Mobile Devices
    let initialPinchDist = 0;
    const onTouchStart = (e: TouchEvent) => {
      if (e.touches.length === 1) {
        isDraggingRef.current = true;
        mouseDownPosRef.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        previousMousePositionRef.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      } else if (e.touches.length === 2) {
        isDraggingRef.current = false;
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        initialPinchDist = Math.hypot(dx, dy);
      }
    };

    const onTouchMove = (e: TouchEvent) => {
      if (e.touches.length === 1 && isDraggingRef.current) {
        const deltaX = e.touches[0].clientX - previousMousePositionRef.current.x;
        const deltaY = e.touches[0].clientY - previousMousePositionRef.current.y;

        cameraSphericalRef.current.theta -= deltaX * 0.008;
        cameraSphericalRef.current.phi = Math.max(
          0.05,
          Math.min(Math.PI - 0.05, cameraSphericalRef.current.phi - deltaY * 0.008)
        );

        previousMousePositionRef.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        updateCamera();
      } else if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const currentDist = Math.hypot(dx, dy);
        const pinchDelta = initialPinchDist - currentDist;

        cameraSphericalRef.current.radius = Math.max(
          250,
          Math.min(2200, cameraSphericalRef.current.radius + pinchDelta * 1.2)
        );
        initialPinchDist = currentDist;
        updateCamera();
      }
    };

    const onTouchEnd = () => {
      isDraggingRef.current = false;
    };

    container.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    container.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("resize", handleResize);

    container.addEventListener("touchstart", onTouchStart, { passive: true });
    window.addEventListener("touchmove", onTouchMove, { passive: true });
    window.addEventListener("touchend", onTouchEnd, { passive: true });

    return () => {
      if (reqIdRef.current) cancelAnimationFrame(reqIdRef.current);
      resizeObserver.disconnect();
      container.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      container.removeEventListener("wheel", onWheel);
      window.removeEventListener("resize", handleResize);

      container.removeEventListener("touchstart", onTouchStart);
      window.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("touchend", onTouchEnd);
      renderer.dispose();
    };
  }, []);

  // Load & Render Real Patient Image Plane (Superimposed under 3D Points & Organs)
  useEffect(() => {
    if (!sceneRef.current) return;
    const scene = sceneRef.current;
    const currentToken = ++textureLoadTokenRef.current;

    // Immediately dispose and remove any previous image plane mesh
    if (imagePlaneMeshRef.current) {
      scene.remove(imagePlaneMeshRef.current);
      imagePlaneMeshRef.current.geometry.dispose();
      const mat = imagePlaneMeshRef.current.material as THREE.MeshBasicMaterial;
      if (mat && mat.map) mat.map.dispose();
      if (mat) mat.dispose();
      imagePlaneMeshRef.current = null;
    }

    // Pure 3D mesh scans never render a 2D image billboard
    if (modality === "mesh") return;

    let textureUrl = imageUrl;
    if (!textureUrl) {
      if (modality === "rgb") {
        textureUrl =
          patientSex === "male"
            ? "/demo/sample_patient_male_rgb.jpg"
            : "/demo/sample_patient_female_rgb.jpg";
      } else if (modality === "depth") {
        textureUrl = "/demo/sample_depth_camera.png";
      }
    }

    if (!textureUrl) return;

    const loader = new THREE.TextureLoader();
    loader.load(
      textureUrl,
      (texture) => {
        // Race condition guard: discard if token expired or scene was torn down
        if (currentToken !== textureLoadTokenRef.current || !sceneRef.current) {
          texture.dispose();
          return;
        }

        texture.colorSpace = THREE.SRGBColorSpace;

        // Anatomically calibrated dimensioning and vertical registration:
        // - Female bust photo: hair top (+410mm) to lower epigastrium (-120mm) -> height 530mm, center Y=+145mm
        // - Male torso photo: chin (+345mm) to mid-thighs (-255mm) -> height 600mm, center Y=+45mm
        // - Depth camera: full body from head to floor (height 840mm, center Y=-50mm)
        let planeW = 420;
        let planeH = 600;
        let planeY = 45;

        const isFemalePhoto =
          patientSex === "female" && (modality === "rgb" || textureUrl.includes("female") || textureUrl.includes("clinical") || textureUrl.includes("woman"));
        const isMalePhoto =
          patientSex === "male" && (modality === "rgb" || textureUrl.includes("male") || textureUrl.includes("man"));
        const isDepth =
          modality === "depth" || textureUrl.includes("depth");

        if (isFemalePhoto) {
          planeW = 400;
          planeH = 530;
          planeY = 145;
        } else if (isMalePhoto) {
          planeW = 420;
          planeH = 600;
          planeY = 45;
        } else if (isDepth) {
          planeW = 1008;
          planeH = 1344;
          planeY = -14;
        }

        const planeGeo = new THREE.PlaneGeometry(planeW, planeH);
        const planeMat = new THREE.MeshBasicMaterial({
          map: texture,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.92,
          depthWrite: false, // Ensures 3D point cloud particles and pins render cleanly on top
        });

        const planeMesh = new THREE.Mesh(planeGeo, planeMat);
        planeMesh.position.set(0, planeY, -35);
        planeMesh.visible = showImagePlane;
        sceneRef.current.add(planeMesh);
        imagePlaneMeshRef.current = planeMesh;
      },
      undefined,
      (err) => {
        console.warn("Failed to load patient image texture:", err);
      }
    );

    return () => {
      if (imagePlaneMeshRef.current && sceneRef.current) {
        sceneRef.current.remove(imagePlaneMeshRef.current);
        imagePlaneMeshRef.current.geometry.dispose();
        const mat = imagePlaneMeshRef.current.material as THREE.MeshBasicMaterial;
        if (mat && mat.map) mat.map.dispose();
        if (mat) mat.dispose();
        imagePlaneMeshRef.current = null;
      }
    };
  }, [modality, patientSex, imageUrl]);

  // Sync image plane visibility
  useEffect(() => {
    if (imagePlaneMeshRef.current) {
      imagePlaneMeshRef.current.visible =
        (modality === "rgb" || modality === "depth") && showImagePlane;
    }
  }, [showImagePlane, modality]);

  // Update Geometry Visibility based on Active Input Modality
  useEffect(() => {
    // 1. Mannequin only on 3D Mesh scans
    if (mannequinGroupRef.current) {
      mannequinGroupRef.current.visible = modality === "mesh" && showMannequin;
    }
    // 2. RealSense Frustum only on 3D Depth Camera
    if (depthSensorGroupRef.current) {
      depthSensorGroupRef.current.visible = modality === "depth" && showSensorRig;
    }
    // 3. Clinical Measurement Backdrop only on RGB Pictures
    if (photoBillboardGroupRef.current) {
      photoBillboardGroupRef.current.visible = modality === "rgb" && showSensorRig;
    }
  }, [modality, showMannequin, showSensorRig]);

  // Update Surface Point Cloud (Dense 4,096 points)
  useEffect(() => {
    if (!sceneRef.current) return;
    const scene = sceneRef.current;

    if (pointsMeshRef.current) {
      scene.remove(pointsMeshRef.current);
      pointsMeshRef.current.geometry.dispose();
      (pointsMeshRef.current.material as THREE.Material).dispose();
      pointsMeshRef.current = null;
    }

    if (!surfacePoints || surfacePoints.length === 0 || !showPoints) return;

    const count = surfacePoints.length;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    // Anatomical alignment
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    for (let i = 0; i < count; i++) {
      const p = surfacePoints[i];
      if (p[0] < minX) minX = p[0];
      if (p[0] > maxX) maxX = p[0];
      if (p[1] < minY) minY = p[1];
      if (p[1] > maxY) maxY = p[1];
      if (p[2] < minZ) minZ = p[2];
      if (p[2] > maxZ) maxZ = p[2];
    }

    const midX = (minX + maxX) / 2;
    const midY = (minY + maxY) / 2;

    let shiftX = 0;
    let shiftY = 0;
    let shiftZ = 0;

    if (Math.abs(midX) > 30) shiftX = midX;
    if (Math.abs(midY - 48) > 30) shiftY = midY - 48;
    if (maxZ > 385) {
      shiftZ = maxZ - 375;
    }

    for (let i = 0; i < count; i++) {
      const p = surfacePoints[i];
      const cx = p[0] - shiftX;
      const cy = p[1] - shiftY;
      const cz = p[2] - shiftZ;

      const [tx, ty, tz] = toThreeCoord([cx, cy, cz]);
      positions[i * 3 + 0] = tx;
      positions[i * 3 + 1] = ty;
      positions[i * 3 + 2] = tz;

      // Olive-green harmonious scientific gradient:
      // Cranial (emerald) -> Torso (primary olive) -> Pelvis (slate-olive)
      if (ty > 260) {
        // Head / Brain / Cranium region
        colors[i * 3 + 0] = 0.35; // R
        colors[i * 3 + 1] = 0.65; // G
        colors[i * 3 + 2] = 0.55; // B
      } else if (ty > 60) {
        // Thorax & Upper Abdomen
        colors[i * 3 + 0] = 0.40;
        colors[i * 3 + 1] = 0.48;
        colors[i * 3 + 2] = 0.32;
      } else {
        // Pelvis & Lower Extremities
        colors[i * 3 + 0] = 0.45;
        colors[i * 3 + 1] = 0.52;
        colors[i * 3 + 2] = 0.42;
      }
    }

    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 3.2,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
    });

    const pointsMesh = new THREE.Points(geometry, material);
    scene.add(pointsMesh);
    pointsMeshRef.current = pointsMesh;
  }, [surfacePoints, showPoints]);

  // Update Target Centroid Pins & Uncertainty Halos
  // When isolateSelected is true, only the selected organ is shown!
  useEffect(() => {
    if (!targetGroupRef.current) return;
    const group = targetGroupRef.current;

    while (group.children.length > 0) {
      const child = group.children[0] as any;
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (Array.isArray(child.material)) {
          child.material.forEach((m: any) => m.dispose());
        } else {
          child.material.dispose();
        }
      }
      group.remove(child);
    }

    if (!predictions) return;

    Object.entries(predictions).forEach(([name, pred]) => {
      const rawCoords = pred.centroid_canonical_mm ?? pred.centroid_input_world_mm;
      const [tx, ty, tz] = toThreeCoord(rawCoords);
      const isSelected = selectedTarget === name;

      // When isolateSelected is active and an organ is selected, only show the selected organ!
      const shouldDisplay = !isolateSelected || !selectedTarget || isSelected;
      if (!shouldDisplay) return;

      // Color scheme:
      // When selected: RADIANT AMBER / GOLD (0xFFB800) with glowing pulsating emissive!
      // Unselected: Low unc (0x4E774A), Moderate (0xC88A2D), High (0xB34A3E)
      let pinColor = 0x4e774a;
      if (pred.uncertainty_level === "moderate") pinColor = 0xc88a2d;
      if (pred.uncertainty_level === "high") pinColor = 0xb34a3e;
      if (isSelected) pinColor = 0xffb800; // Radiant Gold/Amber

      // 1. Organ Centroid Sphere (Interactive clickable mesh)
      const sphereGeo = new THREE.SphereGeometry(isSelected ? 13 : 8, 24, 24);
      const sphereMat = new THREE.MeshStandardMaterial({
        color: pinColor,
        roughness: 0.15,
        metalness: 0.5,
        emissive: isSelected ? 0xffaa00 : pinColor,
        emissiveIntensity: isSelected ? 0.95 : 0.18,
      });
      const sphereMesh = new THREE.Mesh(sphereGeo, sphereMat);
      sphereMesh.position.set(tx, ty, tz);
      sphereMesh.userData = {
        targetName: name,
        isTargetPin: true,
        isPulsing: isSelected,
      };
      group.add(sphereMesh);

      // 2. If Selected: Radiant Expanding Beacon Halo (Periodical Shockwave)
      if (isSelected) {
        const beaconGeo = new THREE.SphereGeometry(15, 18, 18);
        const beaconMat = new THREE.MeshBasicMaterial({
          color: 0xffb800,
          wireframe: true,
          transparent: true,
          opacity: 0.75,
        });
        const beaconMesh = new THREE.Mesh(beaconGeo, beaconMat);
        beaconMesh.position.set(tx, ty, tz);
        beaconMesh.userData = {
          targetName: name,
          isBeacon: true,
        };
        group.add(beaconMesh);
      }

      // 3. Wireframe Uncertainty Radius Sphere
      const radius = Math.max(9, pred.uncertainty_mm);
      const haloGeo = new THREE.SphereGeometry(radius, 16, 16);
      const haloMat = new THREE.MeshBasicMaterial({
        color: pinColor,
        wireframe: true,
        transparent: true,
        opacity: isSelected ? 0.6 : 0.22,
      });
      const haloMesh = new THREE.Mesh(haloGeo, haloMat);
      haloMesh.position.set(tx, ty, tz);
      haloMesh.userData = { targetName: name };
      group.add(haloMesh);

      // 4. Drop Indicator Stalk to Base Plane
      const lineMat = new THREE.LineBasicMaterial({
        color: pinColor,
        transparent: true,
        opacity: isSelected ? 0.75 : 0.3,
      });
      const lineGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(tx, ty, tz),
        new THREE.Vector3(tx, -490, tz),
      ]);

      const line = new THREE.Line(lineGeo, lineMat);
      line.userData = { targetName: name };
      group.add(line);
    });
  }, [predictions, selectedTarget, isolateSelected]);

  const setViewPreset = (preset: "iso" | "coronal_front" | "coronal_back" | "sagittal" | "axial") => {
    if (preset === "coronal_front") {
      cameraSphericalRef.current = { radius: 950, theta: 0, phi: Math.PI / 2 };
    } else if (preset === "coronal_back") {
      cameraSphericalRef.current = { radius: 950, theta: Math.PI, phi: Math.PI / 2 };
    } else if (preset === "sagittal") {
      cameraSphericalRef.current = { radius: 950, theta: Math.PI / 2, phi: Math.PI / 2 };
    } else if (preset === "axial") {
      cameraSphericalRef.current = { radius: 1050, theta: 0, phi: 0.08 };
    } else {
      cameraSphericalRef.current = { radius: 950, theta: Math.PI / 4, phi: Math.PI / 2.4 };
    }
    updateCamera();
  };

  return (
    <div className="relative w-full h-full min-h-[400px] sm:min-h-[520px] rounded-2xl overflow-hidden border border-border bg-[#F5F5F0] shadow-inner flex flex-col">
      {/* 3D Canvas Mounting Element */}
      <div ref={mountRef} className="w-full h-full cursor-grab active:cursor-grabbing touch-none flex-1" />

      {/* Unified Non-Colliding Top Controls Bar */}
      <div className="absolute top-2.5 left-2.5 right-2.5 flex items-center justify-between gap-2 pointer-events-none z-10">
        {/* Left: Modality & Layer Toggles */}
        <div className="flex items-center gap-1 sm:gap-1.5 bg-white/95 backdrop-blur px-2 py-1 sm:px-2.5 sm:py-1.5 rounded-xl border border-border shadow-xs text-xs font-medium text-text-muted pointer-events-auto shrink-0">
          {modality === "mesh" && (
            <button
              onClick={() => setShowMannequin(!showMannequin)}
              className={`px-2 py-1 rounded-lg text-[11px] sm:text-xs transition-colors ${
                showMannequin ? "bg-primary text-white font-semibold shadow-xs" : "hover:bg-slate-100 text-text-muted"
              }`}
              title="Toggle anatomical mannequin silhouette"
            >
              <span>🧍 Mannequin</span>
            </button>
          )}
          {modality === "depth" && (
            <>
              <button
                onClick={() => setShowImagePlane(!showImagePlane)}
                className={`px-2 py-1 rounded-lg text-[11px] sm:text-xs transition-colors ${
                  showImagePlane ? "bg-primary text-white font-semibold shadow-xs" : "hover:bg-slate-100 text-text-muted"
                }`}
                title="Toggle real 3D depth camera image frame"
              >
                <span>📹 Depth Frame</span>
              </button>
              <button
                onClick={() => setShowSensorRig(!showSensorRig)}
                className={`px-1.5 py-1 rounded-lg text-[11px] sm:text-xs transition-colors hidden sm:flex ${
                  showSensorRig ? "bg-slate-200 text-slate-800" : "hover:bg-slate-100 text-text-muted"
                }`}
                title="Toggle Intel RealSense sensor frustum"
              >
                <span>📐 Sensor Rig</span>
              </button>
            </>
          )}
          {modality === "rgb" && (
            <>
              <button
                onClick={() => setShowImagePlane(!showImagePlane)}
                className={`px-2 py-1 rounded-lg text-[11px] sm:text-xs transition-colors ${
                  showImagePlane ? "bg-primary text-white font-semibold shadow-xs" : "hover:bg-slate-100 text-text-muted"
                }`}
                title="Toggle real clinical patient photograph"
              >
                <span>📸 Patient Photo</span>
              </button>
              <button
                onClick={() => setShowSensorRig(!showSensorRig)}
                className={`px-1.5 py-1 rounded-lg text-[11px] sm:text-xs transition-colors hidden sm:flex ${
                  showSensorRig ? "bg-slate-200 text-slate-800" : "hover:bg-slate-100 text-text-muted"
                }`}
                title="Toggle clinical studio frame guides"
              >
                <span>📐 Studio Frame</span>
              </button>
            </>
          )}

          <button
            onClick={() => setShowPoints(!showPoints)}
            className={`px-2 py-1 rounded-lg text-[11px] sm:text-xs transition-colors ${
              showPoints ? "bg-primary text-white font-semibold shadow-xs" : "hover:bg-slate-100 text-text-muted"
            }`}
            title="Toggle 4,096 surface point cloud"
          >
            <span>✨ Cloud</span>
          </button>
        </div>

        {/* Right: Camera Orientation Presets */}
        <div className="flex items-center gap-0.5 sm:gap-1 bg-white/95 backdrop-blur px-1.5 py-1 sm:px-2 sm:py-1.5 rounded-xl border border-border shadow-xs text-xs font-medium text-text-muted pointer-events-auto shrink-0">
          <span className="text-[10px] uppercase font-mono px-1 text-slate-400 hidden sm:inline">Cam:</span>
          <button
            onClick={() => setViewPreset("iso")}
            className="px-1.5 sm:px-2 py-0.5 sm:py-1 text-[11px] rounded hover:bg-background hover:text-text-main transition-colors"
            title="3D Perspective View"
          >
            3D
          </button>
          <button
            onClick={() => setViewPreset("coronal_front")}
            className="px-1.5 sm:px-2 py-0.5 sm:py-1 text-[11px] rounded hover:bg-background hover:text-text-main transition-colors"
            title="Coronal Front View"
          >
            Front
          </button>
          <button
            onClick={() => setViewPreset("sagittal")}
            className="px-1.5 sm:px-2 py-0.5 sm:py-1 text-[11px] rounded hover:bg-background hover:text-text-main transition-colors"
            title="Sagittal Side View"
          >
            Side
          </button>
          <button
            onClick={() => setViewPreset("axial")}
            className="px-1.5 sm:px-2 py-0.5 sm:py-1 text-[11px] rounded hover:bg-background hover:text-text-main transition-colors"
            title="Axial Top View"
          >
            Top
          </button>
        </div>
      </div>

      {/* Selected Target Organ Focus HUD Pill — sits BELOW the top controls row, no overlap */}
      {selectedTarget && (
        <div className="absolute top-12 left-2.5 flex items-center gap-1.5 sm:gap-2 bg-white/95 backdrop-blur px-2.5 py-1 sm:px-3 sm:py-1.5 rounded-xl border border-amber-400 shadow-md text-xs z-10 animate-fade-in max-w-[calc(100%-20px)]">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-500 animate-ping" />
            <span className="font-bold text-slate-900 capitalize">
              {selectedTarget.replace(/_/g, " ")}
            </span>
          </div>

          <div className="h-3 w-px bg-slate-200 mx-0.5" />

          {/* Toggle: Isolate Single Point vs Show All Points */}
          <button
            onClick={() => setIsolateSelected(!isolateSelected)}
            className={`px-2 py-0.5 rounded text-[11px] font-semibold transition-all flex items-center gap-1 ${
              isolateSelected
                ? "bg-amber-500 text-white shadow-xs"
                : "bg-slate-100 text-slate-700 hover:bg-slate-200"
            }`}
            title={isolateSelected ? "Click to show all landmarks" : "Click to isolate only this organ"}
          >
            {isolateSelected ? "👁️ Isolated" : "🌐 All"}
          </button>

          {/* Clear Selection Button */}
          <button
            onClick={() => onSelectTarget?.(null)}
            className="px-1.5 py-0.5 rounded text-[11px] font-medium text-slate-500 hover:text-rose-600 hover:bg-rose-50 transition-colors flex items-center gap-0.5"
            title="Deselect organ"
          >
            <X className="w-3 h-3" />
            <span>Clear</span>
          </button>
        </div>
      )}


      {/* Legend Bottom-Left */}
      <div className="absolute bottom-3 left-3 max-w-[calc(100%-24px)] bg-white/95 backdrop-blur px-2.5 py-1.5 sm:px-3 sm:py-2 rounded-xl border border-border shadow-xs text-[10px] sm:text-xs text-text-muted flex flex-col gap-1 z-10">
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#FFB800] ring-2 ring-amber-400/50 animate-pulse" />
            <span className="font-semibold text-slate-900">Selected (Glow)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-accent-green" />
            <span>Low (&le; 5mm)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-accent-amber" />
            <span>Mod (5-15mm)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-accent-red" />
            <span>High (&gt; 15mm)</span>
          </div>
        </div>
        <span className="text-[9px] sm:text-[10px] text-text-muted/80 hidden sm:inline">
          Click pin to focus &bull; Drag to rotate 360° &bull; Scroll to zoom
        </span>
      </div>
    </div>
  );
}
