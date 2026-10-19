import * as THREE from 'three';

// All boxes inside the viewer use one internal format, in the LiDAR frame
// (x forward, y left, z up), with (x, y, z) at the box *center*:
//   { x, y, z, dx, dy, dz, yaw, label, score }

const BASE = import.meta.env.BASE_URL;

export async function fetchArrayBuffer(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return res.arrayBuffer();
}

export async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return res.text();
}

// KITTI velodyne scans are raw little-endian float32 records: x, y, z, intensity.
export function parseBin(buffer) {
  if (buffer.byteLength % 16 !== 0) {
    throw new Error(`Not a KITTI .bin: ${buffer.byteLength} bytes is not a multiple of 16`);
  }
  return new Float32Array(buffer);
}

// Returns the matrix that maps rectified camera coordinates to LiDAR coordinates.
export function parseCalib(text) {
  const rows = {};
  for (const line of text.split(/\r?\n/)) {
    const idx = line.indexOf(':');
    if (idx < 0) continue;
    rows[line.slice(0, idx).trim()] = line.slice(idx + 1).trim().split(/\s+/).map(Number);
  }
  const r0 = rows.R0_rect ?? rows.R_rect;
  const tr = rows.Tr_velo_to_cam ?? rows.Tr_velo_cam;
  if (!r0 || !tr) throw new Error('calib file is missing R0_rect or Tr_velo_to_cam');

  const R0 = new THREE.Matrix4().set(
    r0[0], r0[1], r0[2], 0,
    r0[3], r0[4], r0[5], 0,
    r0[6], r0[7], r0[8], 0,
    0, 0, 0, 1,
  );
  const Tr = new THREE.Matrix4().set(
    tr[0], tr[1], tr[2], tr[3],
    tr[4], tr[5], tr[6], tr[7],
    tr[8], tr[9], tr[10], tr[11],
    0, 0, 0, 1,
  );
  // rect = R0 * Tr * velo  =>  velo = Tr^-1 * R0^-1 * rect
  const veloFromRect = Tr.clone().invert().multiply(R0.clone().invert());
  return { veloFromRect };
}

// KITTI label_2 lines are in the rectified camera frame: location is the box's
// bottom center, dimensions are (h, w, l) and rotation_y is around the camera's y axis (down).
export function parseKittiLabel(text, calib) {
  const boxes = [];
  for (const line of text.split(/\r?\n/)) {
    const f = line.trim().split(/\s+/);
    if (f.length < 15 || f[0] === 'DontCare') continue;
    const [h, w, l, x, y, z, ry] = f.slice(8, 15).map(Number);
    const center = new THREE.Vector3(x, y - h / 2, z).applyMatrix4(calib.veloFromRect);
    boxes.push({
      x: center.x, y: center.y, z: center.z,
      dx: l, dy: w, dz: h,
      yaw: -ry - Math.PI / 2,
      label: f[0],
      score: f.length > 15 ? Number(f[15]) : 1,
    });
  }
  return boxes;
}

// Prediction JSON (see scripts/export_preds.py). mmdet3d LiDAR boxes have z at the
// box bottom by default, so z_origin defaults to "bottom".
export function parsePredictions(json) {
  if (!Array.isArray(json.boxes)) throw new Error('prediction JSON has no "boxes" array');
  const bottom = (json.z_origin ?? 'bottom') === 'bottom';
  return {
    frame: json.frame ?? null,
    model: json.model ?? 'model',
    boxes: json.boxes.map((b) => ({
      x: +b.x, y: +b.y, z: bottom ? +b.z + b.dz / 2 : +b.z,
      dx: +b.dx, dy: +b.dy, dz: +b.dz, yaw: +b.yaw,
      label: b.label ?? 'Unknown',
      score: b.score ?? 1,
    })),
  };
}

export async function loadManifest() {
  try {
    const json = JSON.parse(await fetchText(`${BASE}data/frames.json`));
    return Array.isArray(json.frames) ? json.frames : [];
  } catch {
    return [];
  }
}

// Loads one manifest entry:
// { id, bin, gt?: { label, calib }, predictions?: [{ model?, file }] }
export async function loadManifestFrame(entry) {
  const url = (p) => `${BASE}data/${p}`;
  const frame = {
    id: entry.id,
    points: parseBin(await fetchArrayBuffer(url(entry.bin))),
    gt: null,
    preds: {},
    warnings: [],
  };
  if (entry.gt) {
    try {
      const calib = parseCalib(await fetchText(url(entry.gt.calib)));
      frame.gt = parseKittiLabel(await fetchText(url(entry.gt.label)), calib);
    } catch (e) {
      frame.warnings.push(`GT not loaded: ${e.message}`);
    }
  }
  for (const p of entry.predictions ?? []) {
    try {
      const pred = parsePredictions(JSON.parse(await fetchText(url(p.file))));
      frame.preds[p.model ?? pred.model] = pred.boxes;
    } catch (e) {
      frame.warnings.push(`Predictions ${p.file} not loaded: ${e.message}`);
    }
  }
  return frame;
}

// Merges dropped files into `frame` (a new .bin starts a fresh frame).
// Calib is read before labels so a label + calib pair can arrive in one drop.
export async function loadDroppedFiles(files, frame) {
  const items = await Promise.all([...files].map(async (file) => {
    const name = file.name.toLowerCase();
    if (name.endsWith('.bin')) return { kind: 'bin', file, data: await file.arrayBuffer() };
    const text = await file.text();
    if (name.endsWith('.json')) return { kind: 'json', file, data: JSON.parse(text) };
    if (/Tr_velo_to_cam|Tr_velo_cam/.test(text)) return { kind: 'calib', file, data: text };
    return { kind: 'label', file, data: text };
  }));

  const bin = items.find((i) => i.kind === 'bin');
  if (bin) {
    frame = {
      id: bin.file.name.replace(/\.bin$/i, ''),
      points: parseBin(bin.data),
      gt: null,
      preds: {},
      warnings: [],
      calib: null,
    };
  }
  if (!frame) throw new Error('Drop a .bin point cloud first');
  frame.warnings = [];

  const calibItem = items.find((i) => i.kind === 'calib');
  if (calibItem) frame.calib = parseCalib(calibItem.data);

  for (const item of items) {
    if (item.kind === 'label') {
      if (!frame.calib) {
        frame.warnings.push(`${item.file.name}: drop the matching calib .txt too (labels are in camera coordinates)`);
        continue;
      }
      frame.gt = parseKittiLabel(item.data, frame.calib);
    } else if (item.kind === 'json') {
      const pred = parsePredictions(item.data);
      frame.preds[pred.model] = pred.boxes;
    }
  }
  return frame;
}
