"use client";

import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { Play, Pause, Activity, ShieldCheck, Cpu, Sparkles, ChevronRight } from "lucide-react";

export function RoboticArmScanner() {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const reqIdRef = useRef<number | null>(null);

  // Kinematic nodes
  const robotBaseRef = useRef<THREE.Group | null>(null);
  const armShoulderRef = useRef<THREE.Group | null>(null);
  const armElbowRef = useRef<THREE.Group | null>(null);
  const armWristRef = useRef<THREE.Group | null>(null);
  const scannerHeadRef = useRef<THREE.Group | null>(null);
  const scanBeamRef = useRef<THREE.Mesh | null>(null);
  const scanFanRef = useRef<THREE.Mesh | null>(null);
  const scanLineRef = useRef<THREE.Mesh | null>(null);

  // Organ centroid meshes for illumination during sweep
  const organMeshesRef = useRef<Record<string, THREE.Mesh>>({});

  const [isAutoScanning, setIsAutoScanning] = useState(true);
  const [scanProgress, setScanProgress] = useState(0.2); // 0 = head, 1 = feet
  const [activeZone, setActiveZone] = useState<string>("Thorax / Heart");
  const [activeTargetLock, setActiveTargetLock] = useState<string>("Heart & Aorta");

  // Camera Orbit
  const isDraggingRef = useRef(false);
  const previousMousePositionRef = useRef({ x: 0, y: 0 });
  const cameraSphericalRef = useRef({ radius: 620, theta: 0.65, phi: Math.PI / 2.7 });
  const targetCenterRef = useRef(new THREE.Vector3(0, 40, 0));

  const updateCamera = () => {
    if (!cameraRef.current) return;
    const { radius, theta, phi } = cameraSphericalRef.current;
    const x = targetCenterRef.current.x + radius * Math.sin(phi) * Math.sin(theta);
    const y = targetCenterRef.current.y + radius * Math.cos(phi);
    const z = targetCenterRef.current.z + radius * Math.sin(phi) * Math.cos(theta);
    cameraRef.current.position.set(x, y, z);
    cameraRef.current.lookAt(targetCenterRef.current);
  };

  useEffect(() => {
    if (!mountRef.current) return;
    const container = mountRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8fafc);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, width / height, 1, 3000);
    camera.up.set(0, 1, 0);
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
    const ambLight = new THREE.AmbientLight(0xffffff, 0.95);
    scene.add(ambLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight1.position.set(200, 400, 300);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xc8d8e8, 0.45);
    dirLight2.position.set(-200, 150, -200);
    scene.add(dirLight2);

    // High-tech Floor Grid
    const gridHelper = new THREE.GridHelper(800, 32, 0x0284c7, 0xe2e8f0);
    gridHelper.position.y = -65;
    scene.add(gridHelper);

    // ==========================================
    // 1. SCANNER BED / EXAMINATION COUCH
    // ==========================================
    const bedGroup = new THREE.Group();

    // Bed Cantilever Base & Pedestal
    const baseMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.3, metalness: 0.6 });
    const bedBaseGeo = new THREE.CylinderGeometry(45, 55, 60, 24);
    const bedBase = new THREE.Mesh(bedBaseGeo, baseMat);
    bedBase.position.set(0, -35, 0);
    bedGroup.add(bedBase);

    // Longitudinal Rails
    const railGeo = new THREE.BoxGeometry(360, 8, 80);
    const railMesh = new THREE.Mesh(railGeo, baseMat);
    railMesh.position.set(0, -5, 0);
    bedGroup.add(railMesh);

    // Patient Couch Mattress (Clean Medical White)
    const matGeo = new THREE.BoxGeometry(340, 12, 75);
    const matMaterial = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.4,
      metalness: 0.05,
    });
    const mattress = new THREE.Mesh(matGeo, matMaterial);
    mattress.position.set(0, 5, 0);
    bedGroup.add(mattress);

    // Headrest Cushion at X = -140
    const pillowGeo = new THREE.BoxGeometry(45, 10, 55);
    const pillowMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.7 });
    const pillow = new THREE.Mesh(pillowGeo, pillowMat);
    pillow.position.set(-135, 12, 0);
    bedGroup.add(pillow);

    scene.add(bedGroup);

    // ==========================================
    // 2. PATIENT HOLOGRAPHIC BODY (SUPINE / LYING DOWN)
    // Patient orientation: Head at X = -135, Feet at X = +140
    // ==========================================
    const patientGroup = new THREE.Group();
    const patientMat = new THREE.MeshStandardMaterial({
      color: 0x7dd3fc,
      transparent: true,
      opacity: 0.22,
      roughness: 0.15,
      metalness: 0.2,
      depthWrite: false,
    });

    // Cranium / Head
    const pHeadGeo = new THREE.SphereGeometry(22, 24, 20);
    pHeadGeo.scale(1.15, 1.0, 1.0);
    const pHead = new THREE.Mesh(pHeadGeo, patientMat);
    pHead.position.set(-135, 24, 0);
    patientGroup.add(pHead);

    // Neck
    const pNeckGeo = new THREE.CylinderGeometry(10, 11, 14, 16);
    pNeckGeo.rotateZ(Math.PI / 2);
    const pNeck = new THREE.Mesh(pNeckGeo, patientMat);
    pNeck.position.set(-114, 21, 0);
    patientGroup.add(pNeck);

    // Thorax / Chest
    const pChestGeo = new THREE.CylinderGeometry(23, 25, 48, 20);
    pChestGeo.rotateZ(Math.PI / 2);
    pChestGeo.scale(1.0, 0.7, 1.25);
    const pChest = new THREE.Mesh(pChestGeo, patientMat);
    pChest.position.set(-84, 23, 0);
    patientGroup.add(pChest);

    // Abdomen
    const pAbdomenGeo = new THREE.CylinderGeometry(24, 22, 45, 20);
    pAbdomenGeo.rotateZ(Math.PI / 2);
    pAbdomenGeo.scale(1.0, 0.68, 1.2);
    const pAbdomen = new THREE.Mesh(pAbdomenGeo, patientMat);
    pAbdomen.position.set(-40, 22, 0);
    patientGroup.add(pAbdomen);

    // Pelvis Basin
    const pPelvisGeo = new THREE.CylinderGeometry(22, 21, 40, 20);
    pPelvisGeo.rotateZ(Math.PI / 2);
    pPelvisGeo.scale(1.0, 0.72, 1.3);
    const pPelvis = new THREE.Mesh(pPelvisGeo, patientMat);
    pPelvis.position.set(0, 21, 0);
    patientGroup.add(pPelvis);

    // Thighs
    const pThighGeo = new THREE.CylinderGeometry(11, 9, 65, 16);
    pThighGeo.rotateZ(Math.PI / 2);

    const pThighL = new THREE.Mesh(pThighGeo, patientMat);
    pThighL.position.set(50, 19, 14);
    patientGroup.add(pThighL);

    const pThighR = new THREE.Mesh(pThighGeo, patientMat);
    pThighR.position.set(50, 19, -14);
    patientGroup.add(pThighR);

    // Calves
    const pCalfGeo = new THREE.CylinderGeometry(8.5, 6.5, 60, 16);
    pCalfGeo.rotateZ(Math.PI / 2);

    const pCalfL = new THREE.Mesh(pCalfGeo, patientMat);
    pCalfL.position.set(110, 17, 14);
    patientGroup.add(pCalfL);

    const pCalfR = new THREE.Mesh(pCalfGeo, patientMat);
    pCalfR.position.set(110, 17, -14);
    patientGroup.add(pCalfR);

    // Internal Holographic Organ Centroids (Inside Patient Body)
    // Includes Brain, Heart, Liver, Kidneys, Bladder, Uterus & Ovaries
    const organs = [
      { id: "brain", pos: [-135, 24, 0], color: 0x06b6d4, name: "Brain" },
      { id: "heart", pos: [-80, 24, -5], color: 0xf43f5e, name: "Heart" },
      { id: "liver", pos: [-45, 23, 10], color: 0x10b981, name: "Liver" },
      { id: "kidney_left", pos: [-35, 17, -14], color: 0x10b981, name: "Left Kidney" },
      { id: "kidney_right", pos: [-35, 17, 14], color: 0x10b981, name: "Right Kidney" },
      { id: "uterus", pos: [-5, 19, 0], color: 0xd946ef, name: "Uterus" },
      { id: "ovary_left", pos: [-8, 18, -12], color: 0xec4899, name: "Left Ovary" },
      { id: "ovary_right", pos: [-8, 18, 12], color: 0xec4899, name: "Right Ovary" },
      { id: "bladder", pos: [15, 18, 0], color: 0x8b5cf6, name: "Urinary Bladder" },
    ];

    const organMeshes: Record<string, THREE.Mesh> = {};
    organs.forEach((org) => {
      const geo = new THREE.SphereGeometry(4.8, 16, 16);
      const mat = new THREE.MeshStandardMaterial({
        color: org.color,
        emissive: org.color,
        emissiveIntensity: 0.6,
        roughness: 0.2,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(org.pos[0], org.pos[1], org.pos[2]);
      patientGroup.add(mesh);
      organMeshes[org.id] = mesh;
    });
    organMeshesRef.current = organMeshes;

    scene.add(patientGroup);

    // ==========================================
    // 3. ARTICULATED 6-DOF ROBOTIC SCANNER ARM (CAIR IIT MANDI)
    // Mounted at Z = -70 beside the bed
    // ==========================================
    const robotGroup = new THREE.Group();
    robotGroup.position.set(-50, -65, -70); // Base position

    // Pedestal Base
    const rBaseGeo = new THREE.CylinderGeometry(28, 35, 25, 24);
    const rBaseMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.25, metalness: 0.7 });
    const rBase = new THREE.Mesh(rBaseGeo, rBaseMat);
    rBase.position.y = 12.5;
    robotGroup.add(rBase);

    // J1: Waist / Shoulder Turntable Group (Rotates around Y)
    const shoulderGroup = new THREE.Group();
    shoulderGroup.position.set(0, 25, 0);

    const shoulderJointGeo = new THREE.CylinderGeometry(22, 22, 28, 24);
    const robotSilverMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.2, metalness: 0.8 });
    const robotAccentMat = new THREE.MeshStandardMaterial({ color: 0x0284c7, roughness: 0.3, metalness: 0.5 }); // CAIR Sky Blue

    const shoulderHousing = new THREE.Mesh(shoulderJointGeo, robotSilverMat);
    shoulderHousing.position.y = 14;
    shoulderGroup.add(shoulderHousing);

    // Link 1: Upper Arm (Shoulder to Elbow)
    const upperArmGroup = new THREE.Group();
    upperArmGroup.position.set(0, 28, 0);

    const uArmGeo = new THREE.CylinderGeometry(11, 13, 100, 20);
    const uArm = new THREE.Mesh(uArmGeo, robotSilverMat);
    uArm.position.y = 50;
    upperArmGroup.add(uArm);

    // Accent Ring on Upper Arm
    const uRingGeo = new THREE.CylinderGeometry(13.2, 13.2, 12, 20);
    const uRing = new THREE.Mesh(uRingGeo, robotAccentMat);
    uRing.position.y = 50;
    upperArmGroup.add(uRing);

    // J2: Elbow Group
    const elbowGroup = new THREE.Group();
    elbowGroup.position.set(0, 100, 0);

    const elbowJointGeo = new THREE.SphereGeometry(18, 20, 16);
    const elbowHousing = new THREE.Mesh(elbowJointGeo, robotSilverMat);
    elbowGroup.add(elbowHousing);

    // Link 2: Forearm (Elbow to Wrist)
    const forearmGroup = new THREE.Group();
    const fArmGeo = new THREE.CylinderGeometry(9, 11, 95, 20);
    const fArm = new THREE.Mesh(fArmGeo, robotSilverMat);
    fArm.position.y = 47.5;
    forearmGroup.add(fArm);

    // J3: Wrist Group
    const wristGroup = new THREE.Group();
    wristGroup.position.set(0, 95, 0);

    const wristJointGeo = new THREE.CylinderGeometry(11, 11, 18, 16);
    wristJointGeo.rotateX(Math.PI / 2);
    const wristHousing = new THREE.Mesh(wristJointGeo, robotAccentMat);
    wristGroup.add(wristHousing);

    // End-Effector: 3D Optical Scanner Head
    const scannerHead = new THREE.Group();
    scannerHead.position.set(0, 12, 0);

    const headGeo = new THREE.BoxGeometry(26, 16, 44);
    const headMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.2, metalness: 0.9 });
    const headMesh = new THREE.Mesh(headGeo, headMat);
    scannerHead.add(headMesh);

    // Optical Lens Aperture
    const lensGeo = new THREE.CylinderGeometry(8, 8, 4, 20);
    const lensMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      emissive: 0x38bdf8,
      emissiveIntensity: 0.9,
      roughness: 0.1,
    });
    const lens = new THREE.Mesh(lensGeo, lensMat);
    lens.position.set(0, -8, 0);
    scannerHead.add(lens);

    // Sweeping Optical Laser Conical Fan
    const fanGeo = new THREE.ConeGeometry(50, 120, 32, 1, true);
    fanGeo.translate(0, -60, 0);
    const fanMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.28,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const scanFan = new THREE.Mesh(fanGeo, fanMat);
    scanFan.scale.set(0.8, 1.0, 1.6);
    scannerHead.add(scanFan);
    scanFanRef.current = scanFan;

    // Projected Laser Scan Line On Patient Body
    const lineGeo = new THREE.PlaneGeometry(6, 68);
    lineGeo.rotateX(-Math.PI / 2);
    const lineMat = new THREE.MeshBasicMaterial({
      color: 0x00e5ff,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });
    const scanLine = new THREE.Mesh(lineGeo, lineMat);
    scene.add(scanLine);
    scanLineRef.current = scanLine;

    // Assemble Hierarchical Kinematic Tree
    wristGroup.add(scannerHead);
    forearmGroup.add(wristGroup);
    elbowGroup.add(forearmGroup);
    upperArmGroup.add(elbowGroup);
    shoulderGroup.add(upperArmGroup);
    robotGroup.add(shoulderGroup);

    scene.add(robotGroup);

    // Save refs for animation loop
    robotBaseRef.current = robotGroup;
    armShoulderRef.current = shoulderGroup;
    armElbowRef.current = upperArmGroup;
    armWristRef.current = forearmGroup;
    scannerHeadRef.current = scannerHead;

    // ==========================================
    // 4. ANIMATION & KINEMATIC SIMULATION LOOP
    // ==========================================
    let clock = new THREE.Clock();

    const animate = () => {
      reqIdRef.current = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();

      // Determine active scan parameter t (0 = head, 1 = feet)
      let scanT = scanProgress;
      if (isAutoScanning) {
        // Smooth continuous sweeping oscillation
        scanT = (Math.sin(elapsed * 0.7) + 1) / 2;
        setScanProgress(scanT);
      }

      // Map scanT to physical patient X coordinate (-135 mm to +40 mm)
      const minX = -135;
      const maxX = 35;
      const beamX = minX + scanT * (maxX - minX);

      // 6-DOF Inverse Kinematic Reach
      if (robotBaseRef.current) {
        robotBaseRef.current.position.x = beamX * 0.7 - 20;
      }

      if (armShoulderRef.current) {
        armShoulderRef.current.rotation.y = Math.sin(beamX * 0.015) * 0.25;
      }

      if (armElbowRef.current) {
        armElbowRef.current.rotation.z = -0.45 + Math.sin(elapsed * 1.5) * 0.02;
        armElbowRef.current.rotation.x = 0.35;
      }

      if (armWristRef.current) {
        armWristRef.current.rotation.z = 0.85;
        armWristRef.current.rotation.x = -0.3;
      }

      if (scannerHeadRef.current) {
        scannerHeadRef.current.position.y = 12 + Math.sin(elapsed * 3) * 1.2;
      }

      // Position optical scan line on patient body surface
      if (scanLineRef.current) {
        scanLineRef.current.position.set(beamX, 26, 0);
        scanLineRef.current.rotation.y = Math.sin(elapsed * 4) * 0.04;
      }

      // Dynamic Organ Centroid Illumination Beacons
      Object.entries(organMeshesRef.current).forEach(([orgId, mesh]) => {
        const orgX = mesh.position.x;
        const dist = Math.abs(beamX - orgX);
        const mat = mesh.material as THREE.MeshStandardMaterial;

        if (dist < 26) {
          mat.emissiveIntensity = 1.8 + Math.sin(elapsed * 8) * 0.4;
          mesh.scale.set(1.4, 1.4, 1.4);
        } else {
          mat.emissiveIntensity = 0.4;
          mesh.scale.set(1.0, 1.0, 1.0);
        }
      });

      // Update zone & target lock labels
      if (scanT < 0.22) {
        setActiveZone("Cranial Vault (Head)");
        setActiveTargetLock("Brain & Skull (±5.5mm)");
      } else if (scanT < 0.45) {
        setActiveZone("Thoracic Cavity (Chest)");
        setActiveTargetLock("Heart, Aorta & Lungs");
      } else if (scanT < 0.68) {
        setActiveZone("Abdominal Viscera");
        setActiveTargetLock("Liver, Spleen & Kidneys");
      } else {
        setActiveZone("Pelvic Basin");
        setActiveTargetLock("Uterus / Ovaries & Bladder");
      }

      renderer.render(scene, camera);
    };
    animate();

    // Mouse Navigation Controls
    const onMouseDown = (e: MouseEvent) => {
      isDraggingRef.current = true;
      previousMousePositionRef.current = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isDraggingRef.current) return;
      const deltaX = e.clientX - previousMousePositionRef.current.x;
      const deltaY = e.clientY - previousMousePositionRef.current.y;

      cameraSphericalRef.current.theta -= deltaX * 0.007;
      cameraSphericalRef.current.phi = Math.max(
        0.1,
        Math.min(Math.PI / 2.05, cameraSphericalRef.current.phi - deltaY * 0.007)
      );

      previousMousePositionRef.current = { x: e.clientX, y: e.clientY };
      updateCamera();
    };

    const onMouseUp = () => {
      isDraggingRef.current = false;
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      cameraSphericalRef.current.radius = Math.max(
        300,
        Math.min(1200, cameraSphericalRef.current.radius + e.deltaY * 0.5)
      );
      updateCamera();
    };

    // Touch Navigation Controls for Mobile Phones & Tablets
    let initialPinchDist = 0;
    const onTouchStart = (e: TouchEvent) => {
      if (e.touches.length === 1) {
        isDraggingRef.current = true;
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
          0.1,
          Math.min(Math.PI / 2.05, cameraSphericalRef.current.phi - deltaY * 0.008)
        );

        previousMousePositionRef.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        updateCamera();
      } else if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const currentDist = Math.hypot(dx, dy);
        const pinchDelta = initialPinchDist - currentDist;

        cameraSphericalRef.current.radius = Math.max(
          300,
          Math.min(1200, cameraSphericalRef.current.radius + pinchDelta * 1.2)
        );
        initialPinchDist = currentDist;
        updateCamera();
      }
    };

    const onTouchEnd = () => {
      isDraggingRef.current = false;
    };

    // Attach event listeners
    container.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    container.addEventListener("wheel", onWheel, { passive: false });

    container.addEventListener("touchstart", onTouchStart, { passive: true });
    window.addEventListener("touchmove", onTouchMove, { passive: true });
    window.addEventListener("touchend", onTouchEnd, { passive: true });

    // Handle container resize
    const handleResize = () => {
      if (!container || !camera || !renderer) return;
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

    window.addEventListener("resize", handleResize);

    return () => {
      if (reqIdRef.current) cancelAnimationFrame(reqIdRef.current);
      resizeObserver.disconnect();
      container.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      container.removeEventListener("wheel", onWheel);

      container.removeEventListener("touchstart", onTouchStart);
      window.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("touchend", onTouchEnd);
      window.removeEventListener("resize", handleResize);

      renderer.dispose();
    };
  }, [isAutoScanning]);

  const setPresetZone = (zone: "head" | "chest" | "abdomen" | "pelvis") => {
    setIsAutoScanning(false);
    if (zone === "head") setScanProgress(0.1);
    else if (zone === "chest") setScanProgress(0.35);
    else if (zone === "abdomen") setScanProgress(0.58);
    else setScanProgress(0.85);
  };

  return (
    <div className="relative w-full h-[460px] sm:h-[500px] lg:h-[540px] rounded-3xl overflow-hidden border border-slate-200 bg-slate-50/50 shadow-lg select-none">
      <div
        ref={mountRef}
        className="w-full h-full cursor-grab active:cursor-grabbing touch-none"
      />

      {/* Top HUD Overlay: Live Telemetry & CAIR Medical Robotics Status */}
      <div className="absolute top-2.5 sm:top-3 left-2.5 sm:left-3 right-2.5 sm:right-3 flex flex-col sm:flex-row sm:items-start justify-between gap-1.5 sm:gap-2 pointer-events-none z-10">
        {/* Left Telemetry Box */}
        <div className="flex sm:flex-col items-center sm:items-start justify-between sm:justify-start gap-2 sm:gap-1 bg-white/95 backdrop-blur-md px-3 sm:px-4 py-1.5 sm:py-2.5 rounded-xl sm:rounded-2xl border border-slate-200 shadow-xs sm:shadow-sm pointer-events-auto sm:max-w-xs">
          <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
            <span className="w-2 sm:w-2.5 h-2 sm:h-2.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[10px] sm:text-xs font-bold text-slate-900 uppercase tracking-wide">
              CAIR Scanner
            </span>
          </div>
          <div className="hidden sm:flex items-center gap-1.5 text-[10px] sm:text-[11px] text-slate-500 font-mono">
            <Cpu className="w-3 h-3 text-sky-600 shrink-0" />
            <span>6-DOF Articulated Kinematics</span>
          </div>
          <div className="text-[10px] sm:text-xs font-semibold text-slate-800 flex items-center gap-1 shrink-0">
            <Activity className="w-3 h-3 text-primary shrink-0" />
            <span className="hidden sm:inline">Target:</span>
            <span className="text-primary font-bold">{activeTargetLock}</span>
          </div>
        </div>

        {/* Right Steer Buttons */}
        <div className="flex items-center gap-1 bg-white/95 backdrop-blur-md p-1 sm:px-2.5 sm:py-1.5 rounded-xl sm:rounded-2xl border border-slate-200 shadow-xs sm:shadow-sm pointer-events-auto text-xs font-medium text-slate-700 overflow-x-auto no-scrollbar max-w-full">
          <button
            onClick={() => setIsAutoScanning(!isAutoScanning)}
            className={`flex items-center gap-1 px-2 py-0.5 sm:px-2.5 sm:py-1 rounded-lg sm:rounded-xl font-semibold text-[11px] sm:text-xs shrink-0 transition-all ${
              isAutoScanning
                ? "bg-primary text-white shadow-xs"
                : "bg-slate-100 hover:bg-slate-200 text-slate-800"
            }`}
          >
            {isAutoScanning ? (
              <>
                <Pause className="w-3 h-3" />
                <span>Scanning</span>
              </>
            ) : (
              <>
                <Play className="w-3 h-3" />
                <span>Auto</span>
              </>
            )}
          </button>

          <div className="h-3 w-px bg-slate-200 mx-0.5 shrink-0" />

          {/* Quick Zone Presets */}
          <button
            onClick={() => setPresetZone("head")}
            className="px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-md sm:rounded-lg hover:bg-slate-100 text-[11px] transition-colors shrink-0"
          >
            Head
          </button>
          <button
            onClick={() => setPresetZone("chest")}
            className="px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-md sm:rounded-lg hover:bg-slate-100 text-[11px] transition-colors shrink-0"
          >
            Chest
          </button>
          <button
            onClick={() => setPresetZone("abdomen")}
            className="px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-md sm:rounded-lg hover:bg-slate-100 text-[11px] transition-colors shrink-0"
          >
            Abdomen
          </button>
          <button
            onClick={() => setPresetZone("pelvis")}
            className="px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-md sm:rounded-lg hover:bg-slate-100 text-[11px] transition-colors shrink-0"
          >
            Pelvis
          </button>
        </div>
      </div>

      {/* Bottom Overlay: Non-Invasive Optical Scanning Guarantee */}
      <div className="absolute bottom-2.5 sm:bottom-3 left-2.5 sm:left-3 right-2.5 sm:right-3 flex items-center justify-between gap-1.5 sm:gap-3 bg-white/95 backdrop-blur-md px-2.5 sm:px-4 py-1.5 sm:py-2 rounded-xl sm:rounded-2xl border border-slate-200 shadow-xs sm:shadow-sm z-10 text-[10px] sm:text-xs text-slate-600 pointer-events-none">
        <div className="flex items-center gap-1.5 sm:gap-2 truncate">
          <ShieldCheck className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-emerald-600 shrink-0" />
          <span className="truncate">
            <strong className="text-slate-900">Zero Radiation:</strong> Optical 3D surface scanning replaces ionizing CT.
          </span>
        </div>
        <div className="flex items-center gap-1 text-[10px] font-mono text-slate-500 whitespace-nowrap shrink-0">
          <span>IIT Mandi</span>
        </div>
      </div>
    </div>
  );
}
