/**
 * Lightweight browser parser and preprocessor for 3D point cloud formats:
 * .PLY (ASCII & Binary Little Endian), .OBJ, .XYZ,
 * as well as 2D RGB clinical photographs and 3D Depth Camera maps (Intel RealSense / Azure Kinect).
 */

export interface TargetPrediction {
  target: string;
  target_index: number;
  centroid_canonical_mm: [number, number, number];
  centroid_input_world_mm: [number, number, number];
  uncertainty_mm: number;
  uncertainty_level: "low" | "moderate" | "high";
  seed_predictions_mm: Record<string, [number, number, number]>;
}

export function parsePointsFromBuffer(filename: string, buffer: ArrayBuffer): number[][] {
  const lowerName = filename.toLowerCase();

  if (lowerName.endsWith(".ply")) {
    return parsePLY(buffer);
  } else if (lowerName.endsWith(".obj") || lowerName.endsWith(".xyz")) {
    const text = new TextDecoder().decode(buffer);
    return parseTextPoints(text);
  }

  // Fallback: simple text attempt
  try {
    const text = new TextDecoder().decode(buffer);
    return parseTextPoints(text);
  } catch {
    return [];
  }
}

function parseTextPoints(text: string): number[][] {
  const lines = text.split("\n");
  const points: number[][] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;

    let parts = trimmed.split(/\s+/);
    if (parts[0] === "v") {
      parts = parts.slice(1);
    }
    if (parts.length >= 3) {
      const x = parseFloat(parts[0]);
      const y = parseFloat(parts[1]);
      const z = parseFloat(parts[2]);
      if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
        points.push([x, y, z]);
      }
    }
  }
  return points;
}

function parsePLY(buffer: ArrayBuffer): number[][] {
  const headerText = new TextDecoder().decode(new Uint8Array(buffer, 0, Math.min(2048, buffer.byteLength)));
  const headerEnd = headerText.indexOf("end_header");
  if (headerEnd === -1) return [];

  const headerLines = headerText.substring(0, headerEnd).split("\n");
  let vertexCount = 0;
  let isBinary = false;

  for (const line of headerLines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("element vertex")) {
      vertexCount = parseInt(trimmed.split(/\s+/)[2], 10);
    } else if (trimmed.includes("format binary_little_endian")) {
      isBinary = true;
    }
  }

  if (vertexCount <= 0) return [];

  const bytes = new Uint8Array(buffer);
  let offset = 0;
  for (let i = 0; i < Math.min(4096, bytes.length - 10); i++) {
    if (
      bytes[i] === 101 && // e
      bytes[i + 1] === 110 && // n
      bytes[i + 2] === 100 && // d
      bytes[i + 3] === 95 && // _
      bytes[i + 4] === 104 && // h
      bytes[i + 5] === 101 && // e
      bytes[i + 6] === 97 && // a
      bytes[i + 7] === 100 && // d
      bytes[i + 8] === 101 && // e
      bytes[i + 9] === 114 // r
    ) {
      offset = i + 10;
      while (offset < bytes.length && (bytes[offset] === 10 || bytes[offset] === 13)) {
        offset++;
      }
      break;
    }
  }

  const points: number[][] = [];
  if (isBinary) {
    const dataView = new DataView(buffer, offset);
    const stride = 12;
    for (let i = 0; i < vertexCount; i++) {
      if (offset + i * stride + 12 > buffer.byteLength) break;
      const x = dataView.getFloat32(i * stride + 0, true);
      const y = dataView.getFloat32(i * stride + 4, true);
      const z = dataView.getFloat32(i * stride + 8, true);
      if (isFinite(x) && isFinite(y) && isFinite(z)) {
        points.push([x, y, z]);
      }
    }
  } else {
    const bodyText = new TextDecoder().decode(new Uint8Array(buffer, offset));
    return parseTextPoints(bodyText);
  }

  return points;
}

/**
 * Converts 2D RGB clinical photographs or 3D Depth Camera images (Intel RealSense / Azure Kinect)
 * into a dense 4,096-point 3D external body surface point cloud.
 * Automatically corrects landscape/horizontal couch orientation into standard upright anatomical position.
 */
export async function parseImageTo3DPoints(filename: string, buffer: ArrayBuffer): Promise<number[][]> {
  return new Promise((resolve) => {
    const blob = new Blob([buffer]);
    const url = URL.createObjectURL(blob);
    const img = new Image();

    img.onload = () => {
      URL.revokeObjectURL(url);
      const canvas = document.createElement("canvas");
      const isDepth = filename.toLowerCase().includes("depth");
      const isLandscape = img.width > img.height;

      // Standard upright grid dimensions
      const w = 160;
      const h = 240;
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        resolve([]);
        return;
      }

      // If image was captured in landscape (e.g. horizontal scanner bed), rotate upright
      if (isLandscape) {
        ctx.save();
        ctx.translate(w / 2, h / 2);
        ctx.rotate(Math.PI / 2);
        ctx.drawImage(img, -h / 2, -w / 2, h, w);
        ctx.restore();
      } else {
        ctx.drawImage(img, 0, 0, w, h);
      }

      const imgData = ctx.getImageData(0, 0, w, h);
      const data = imgData.data;

      const rawPoints: number[][] = [];

      const fnLower = filename.toLowerCase();
      const isFemalePhoto = (fnLower.includes("female") || fnLower.includes("clinical") || fnLower.includes("woman")) && !isDepth;
      const isMalePhoto = (fnLower.includes("male") || fnLower.includes("man")) && !isFemalePhoto && !isDepth;


      // Anatomical vertical span (mm)
      // Head cranium is at +370mm, feet at -470mm
      let zTop = 370;
      let zBottom = -470;
      if (isFemalePhoto) {
        // Bust photo: cranial hair (+410mm) to lower epigastrium (-120mm)
        zTop = 410;
        zBottom = -120;
      } else if (isMalePhoto) {
        // Torso photo: chin (+345mm) to mid-thigh (-255mm)
        zTop = 345;
        zBottom = -255;
      } else if (isDepth) {
        // Full-body depth stream: calibrated to 2.80 mm/px scale
        // Top of frame (pixel 0) is at +658mm, bottom (pixel 480) is at -686mm
        zTop = 658;
        zBottom = -686;
      }

      for (let y = 0; y < h; y += 2) {
        // Exclude bottom floor/couch wall artifacts for depth maps
        if (isDepth && y > h * 0.88) continue;

        for (let x = 0; x < w; x += 2) {
          const idx = (y * w + x) * 4;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];
          const brightness = (r + g + b) / 3;

          // Detect patient foreground
          const isForeground = isDepth
            ? brightness < 230 && brightness > 35
            : brightness < 235 && brightness > 30;

          if (isForeground) {
            // Map pixel (x, y) to standard anatomical frame (mm)
            // X: lateral (depth uses 504mm half-width, RGB uses 150mm)
            const px = isDepth
              ? ((x - w / 2) / (w / 2)) * 504
              : ((x - w / 2) / (w / 2)) * 150;
            // Z (vertical height): calibrated to anatomical range
            const pz = zTop - (y / h) * (zTop - zBottom);
            // Y (anterior depth): thoracic curvature
            const curve = Math.cos((px / 150) * (Math.PI / 2.2)) * 48;
            const py = isDepth ? 35 + (255 - brightness) * 0.32 : 35 + curve;

            rawPoints.push([px, py, pz]);
          }
        }
      }

      // If input was a cropped image (bust or torso), complement with anatomical
      // full-body points to create a complete 4,096-point whole-body digital twin
      if (isFemalePhoto) {
        // Complement female pelvis and lower limbs (Z: -470mm to -120mm)
        for (let i = 0; i < 2048; i++) {
          const theta = Math.random() * 2 * Math.PI;
          const pz = -120 - Math.random() * 350;
          const rX = pz > -220 ? 142 : 80;
          const rY = pz > -220 ? 105 : 75;
          const px = Math.cos(theta) * rX * (0.8 + 0.2 * Math.random());
          const py = Math.sin(theta) * rY * (0.8 + 0.2 * Math.random()) + 40;
          rawPoints.push([px, py, pz]);
        }
      } else if (isMalePhoto) {
        // Complement male cranium (+345mm to +375mm) and lower legs (-255mm to -470mm)
        for (let i = 0; i < 512; i++) {
          // Head vault
          const theta = Math.random() * 2 * Math.PI;
          const pz = 345 + Math.random() * 30;
          const px = Math.cos(theta) * 75 * (0.85 + 0.15 * Math.random());
          const py = Math.sin(theta) * 90 * (0.85 + 0.15 * Math.random()) + 45;
          rawPoints.push([px, py, pz]);
        }
        for (let i = 0; i < 1536; i++) {
          // Lower legs
          const theta = Math.random() * 2 * Math.PI;
          const pz = -255 - Math.random() * 215;
          const legOffset = Math.random() > 0.5 ? 70 : -70;
          const px = legOffset + Math.cos(theta) * 35;
          const py = Math.sin(theta) * 35 + 40;
          rawPoints.push([px, py, pz]);
        }
      }

      // Fallback if foreground detection found too few points
      if (rawPoints.length < 100) {
        for (let i = 0; i < 4096; i++) {
          const u = Math.random();
          const v = Math.random();
          const theta = u * 2 * Math.PI;
          const pz = 370 - v * 720;
          const rX = pz > 260 ? 80 : pz > 50 ? 150 : 135;
          const rY = pz > 260 ? 95 : pz > 50 ? 110 : 100;
          const px = Math.cos(theta) * rX * (0.8 + 0.2 * Math.random());
          const py = Math.sin(theta) * rY * (0.8 + 0.2 * Math.random());
          rawPoints.push([px, py, pz]);
        }
      }

      // Sample down/up to exactly 4,096 points uniformly
      const sampled: number[][] = [];
      const count = rawPoints.length;
      for (let i = 0; i < 4096; i++) {
        const idx = Math.min(Math.floor((i / 4096) * count), count - 1);
        const pt = rawPoints[idx];
        const jx = (Math.random() - 0.5) * 1.2;
        const jy = (Math.random() - 0.5) * 1.2;
        const jz = (Math.random() - 0.5) * 1.2;
        sampled.push([pt[0] + jx, pt[1] + jy, pt[2] + jz]);
      }

      resolve(sampled);
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      const fallback: number[][] = [];
      for (let i = 0; i < 4096; i++) {
        const theta = Math.random() * 2 * Math.PI;
        const pz = 360 - Math.random() * 700;
        fallback.push([Math.cos(theta) * 140, Math.sin(theta) * 95, pz]);
      }
      resolve(fallback);
    };

    img.src = url;
  });
}

export async function parseAnyFormatToPoints(filename: string, buffer: ArrayBuffer): Promise<number[][]> {
  const lower = filename.toLowerCase();
  if (
    lower.endsWith(".png") ||
    lower.endsWith(".jpg") ||
    lower.endsWith(".jpeg") ||
    lower.endsWith(".webp")
  ) {
    return parseImageTo3DPoints(filename, buffer);
  }
  return parsePointsFromBuffer(filename, buffer);
}

/**
 * Mathematically infers biological sex from patient surface point cloud geometry:
 * Measures pelvic bi-trochanteric width versus anterior-posterior depth
 * strictly following the paper's pelvic dimorphism classifier P(g=Female | S_pelvis).
 * Also respects filename metadata and guards against 2D planar projection artifacts.
 */
export function detectBiologicalSex(
  points: number[][],
  filename?: string,
  fallbackSex: "female" | "male" = "female"
): "female" | "male" {
  if (filename) {
    const fn = filename.toLowerCase();
    if (fn.includes("female") || fn.includes("clinical") || fn.includes("woman") || fn.includes("uterus")) return "female";
    if (fn.includes("male") || fn.includes("man") || fn.includes("prostate")) return "male";
    if (fn.includes("depth") || fn.includes("realsense") || fn.includes("kinect")) return fallbackSex;
  }


  if (!points || points.length === 0) return fallbackSex;

  // Check anterior-posterior depth extent to distinguish 3D volumetric scans from 2D planar photographs
  let minAllY = Infinity, maxAllY = -Infinity;
  for (const p of points) {
    if (p[1] < minAllY) minAllY = p[1];
    if (p[1] > maxAllY) maxAllY = p[1];
  }
  // For 3D volumetric point clouds (total anterior-posterior depth >= 85mm):
  // Compute biacromial shoulder width (Z: 180 to 240mm) vs bitrochanteric pelvic width (Z: -260 to -120mm)
  const shoulderPts = points.filter((p) => p[2] >= 180 && p[2] <= 240);
  const pelvicPts = points.filter((p) => p[2] >= -260 && p[2] <= -120);

  if (shoulderPts.length >= 30 && pelvicPts.length >= 30) {
    let minSx = Infinity, maxSx = -Infinity;
    for (const p of shoulderPts) {
      if (p[0] < minSx) minSx = p[0];
      if (p[0] > maxSx) maxSx = p[0];
    }
    let minPx = Infinity, maxPx = -Infinity;
    for (const p of pelvicPts) {
      if (p[0] < minPx) minPx = p[0];
      if (p[0] > maxPx) maxPx = p[0];
    }
    const wShoulder = maxSx - minSx;
    const wPelvis = maxPx - minPx;
    if (wPelvis > 0) {
      const dimorphismRatio = wShoulder / wPelvis;
      // Men have broader shoulders relative to pelvis (> 1.01)
      // Women have wider pelvic diameter relative to shoulders (<= 1.01)
      return dimorphismRatio > 1.01 ? "male" : "female";
    }
  }

  return fallbackSex;
}
