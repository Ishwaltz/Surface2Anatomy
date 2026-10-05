import * as THREE from "three";

/**
 * Creates an upright, anatomically proportioned human body mannequin
 * with smooth holographic glass styling, aligned with canonical medical coordinates:
 * - Three.js Y: Superior (+) to Inferior (-) [Cranium: +380mm, Feet: -470mm]
 * - Three.js X: Anatomical Right (+) to Left (-)
 * - Three.js Z: Anterior (+) to Posterior (-) [Chest: +85mm, Spine: +15mm]
 */
export function createHumanMannequin(): THREE.Group {
  const group = new THREE.Group();

  // Premium holographic frosted glass material
  const bodyMaterial = new THREE.MeshStandardMaterial({
    color: 0x94b4c7, // Elegant ice-blue translucent tint
    transparent: true,
    opacity: 0.13,
    roughness: 0.2,
    metalness: 0.12,
    depthWrite: false,
    side: THREE.FrontSide, // Clear transparency without dark back-face disc artifacts
  });

  // Subtle accent contour line material
  const edgeMaterial = new THREE.LineBasicMaterial({
    color: 0x608aa8,
    transparent: true,
    opacity: 0.22,
    depthWrite: false,
  });

  function addSmoothPart(
    geo: THREE.BufferGeometry,
    pos: [number, number, number],
    scale: [number, number, number] = [1, 1, 1],
    rot: [number, number, number] = [0, 0, 0]
  ) {
    geo.computeVertexNormals();
    const mesh = new THREE.Mesh(geo, bodyMaterial);
    mesh.position.set(...pos);
    mesh.scale.set(...scale);
    mesh.rotation.set(...rot);
    group.add(mesh);
  }

  // 1. Head & Cranial Vault (Brain is at Y = 350mm)
  const craniumGeo = new THREE.SphereGeometry(62, 32, 24);
  addSmoothPart(craniumGeo, [0, 350, 48], [1.0, 1.15, 1.12]);

  // Jaw / Chin contour (Y: 285 to 325mm)
  const jawGeo = new THREE.CylinderGeometry(36, 24, 45, 24);
  addSmoothPart(jawGeo, [0, 305, 52], [1.0, 1.0, 1.08]);

  // 2. Cervical Neck (Y: 235 - 285mm; C1-C7 vertebrae are at 231 - 280mm)
  const neckGeo = new THREE.CylinderGeometry(32, 38, 50, 24);
  addSmoothPart(neckGeo, [0, 260, 48], [1.0, 1.0, 0.95]);

  // 3. Clavicles / Shoulder Girdle Bridge (Y: ~215mm; clavicles are at 210mm)
  const shoulderGeo = new THREE.CapsuleGeometry(24, 240, 16, 24);
  addSmoothPart(shoulderGeo, [0, 215, 45], [1.0, 0.75, 0.85], [0, 0, Math.PI / 2]);

  // 4. Thorax & Ribcage (Y: 20 to 200mm, height 180mm, center at 110mm)
  // Heart is at Y = 75.8mm, Ribs 1-10 are from 199mm down to 16mm
  const thoraxGeo = new THREE.CylinderGeometry(132, 114, 180, 32);
  addSmoothPart(thoraxGeo, [0, 110, 48], [1.08, 1.0, 0.88]);

  // 5. Abdomen & Waist (Y: -60 to 20mm, height 80mm, center at -20mm)
  // Liver is at Y = 5.2mm, Spleen at Y = 0.4mm, Kidneys at Y = -49.5mm
  const abdomenGeo = new THREE.CylinderGeometry(112, 118, 80, 32);
  addSmoothPart(abdomenGeo, [0, -20, 44], [1.04, 1.0, 0.88]);

  // 6. Pelvis & Lower Abdomen (Y: -140 to -60mm, height 80mm, center at -100mm)
  // Uterus is at -78mm, Bladder at -95mm, Prostate at -105mm
  const pelvisGeo = new THREE.CylinderGeometry(120, 102, 80, 32);
  addSmoothPart(pelvisGeo, [0, -100, 42], [1.12, 1.0, 0.95]);

  // 7. Upper Arms (Left & Right)
  const armGeo = new THREE.CapsuleGeometry(18, 160, 12, 16);
  addSmoothPart(armGeo, [-158, 120, 42], [1.0, 1.0, 0.9], [0, 0, 0.08]);
  addSmoothPart(armGeo, [158, 120, 42], [1.0, 1.0, 0.9], [0, 0, -0.08]);

  // 8. Forearms
  const forearmGeo = new THREE.CapsuleGeometry(15, 140, 12, 16);
  addSmoothPart(forearmGeo, [-174, -20, 42], [1.0, 1.0, 0.9], [0, 0, 0.04]);
  addSmoothPart(forearmGeo, [174, -20, 42], [1.0, 1.0, 0.9], [0, 0, -0.04]);

  // 9. Thighs (Left & Right)
  const thighGeo = new THREE.CapsuleGeometry(38, 170, 16, 20);
  addSmoothPart(thighGeo, [-64, -235, 42], [1.0, 1.0, 0.95], [0, 0, -0.03]);
  addSmoothPart(thighGeo, [64, -235, 42], [1.0, 1.0, 0.95], [0, 0, 0.03]);

  // 10. Lower Legs (Calves)
  const calfGeo = new THREE.CapsuleGeometry(28, 150, 16, 20);
  addSmoothPart(calfGeo, [-66, -400, 40], [0.95, 1.0, 0.9]);
  addSmoothPart(calfGeo, [66, -400, 40], [0.95, 1.0, 0.9]);

  // 11. Sleek Turntable Pedestal Base
  const pedestalGeo = new THREE.CylinderGeometry(190, 200, 8, 36);
  const pedestalMat = new THREE.MeshStandardMaterial({
    color: 0xd9e2ec,
    roughness: 0.4,
    metalness: 0.3,
  });
  const pedestal = new THREE.Mesh(pedestalGeo, pedestalMat);
  pedestal.position.set(0, -490, 40);
  group.add(pedestal);

  // Pedestal glowing ring
  const ringGeo = new THREE.RingGeometry(182, 186, 48);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0x38bdf8,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.6,
  });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.rotation.x = Math.PI / 2;
  ring.position.set(0, -485, 40);
  group.add(ring);

  return group;
}

/**
 * Creates an Intel RealSense / Azure Kinect 3D Depth Sensor representation:
 * Camera enclosure at [0, 80, 750], optical cone ray frustum pointing to patient,
 * and calibrated depth range bounds.
 */
export function createDepthSensorRig(): THREE.Group {
  const group = new THREE.Group();

  // 1. Sleek, non-obstructive Depth Camera Sensor mounted elevated at [0, 480, 650]
  const bodyGeo = new THREE.BoxGeometry(120, 24, 20);
  const bodyMat = new THREE.MeshStandardMaterial({
    color: 0x465133,
    roughness: 0.4,
    metalness: 0.6,
    transparent: true,
    opacity: 0.75,
  });
  const body = new THREE.Mesh(bodyGeo, bodyMat);
  body.position.set(0, 480, 650);
  group.add(body);

  // 2. Optical Lenses (IR projector + stereo depth sensors)
  const lensGeo = new THREE.CylinderGeometry(6, 6, 4, 16);
  const lensMat = new THREE.MeshStandardMaterial({
    color: 0x8b9a6d,
    roughness: 0.2,
    metalness: 0.7,
    emissive: 0x8b9a6d,
    emissiveIntensity: 0.3,
  });

  const lensLeft = new THREE.Mesh(lensGeo, lensMat);
  lensLeft.rotation.x = Math.PI / 2;
  lensLeft.position.set(-36, 480, 640);
  group.add(lensLeft);

  const lensRight = new THREE.Mesh(lensGeo, lensMat);
  lensRight.rotation.x = Math.PI / 2;
  lensRight.position.set(36, 480, 640);
  group.add(lensRight);

  // 3. Subtle Optical Ray Frustum Pyramid (Soft olive lines, zero occlusion)
  const frustumPoints = [
    new THREE.Vector3(0, 480, 640),
    new THREE.Vector3(-220, 360, 20),
    new THREE.Vector3(0, 480, 640),
    new THREE.Vector3(220, 360, 20),
    new THREE.Vector3(0, 480, 640),
    new THREE.Vector3(220, -480, 20),
    new THREE.Vector3(0, 480, 640),
    new THREE.Vector3(-220, -480, 20),
  ];

  const frustumGeo = new THREE.BufferGeometry().setFromPoints(frustumPoints);
  const frustumMat = new THREE.LineBasicMaterial({
    color: 0x8b9a6d,
    transparent: true,
    opacity: 0.22,
  });
  const frustum = new THREE.LineSegments(frustumGeo, frustumMat);
  group.add(frustum);

  return group;
}

/**
 * Creates a Clinical Photographic Backdrop Frame for RGB photos:
 * Vertical measurement grid billboard showing clinical studio calibration.
 */
export function createPhotoBillboardRig(): THREE.Group {
  const group = new THREE.Group();

  // Subtle Clinical Measurement Caliper Guide Frame (Zero occlusion)
  const standMat = new THREE.MeshBasicMaterial({
    color: 0x8b9a6d,
    transparent: true,
    opacity: 0.35,
  });
  const guidePoints = [
    new THREE.Vector3(-195, 430, -36),
    new THREE.Vector3(195, 430, -36),
    new THREE.Vector3(195, -330, -36),
    new THREE.Vector3(-195, -330, -36),
    new THREE.Vector3(-195, 430, -36),
  ];
  const guideGeo = new THREE.BufferGeometry().setFromPoints(guidePoints);
  const guideLine = new THREE.Line(guideGeo, standMat);
  group.add(guideLine);

  return group;
}

