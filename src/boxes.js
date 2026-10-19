import * as THREE from 'three';

export const CLASS_COLORS = {
  Car: '#ff8c2a',
  Van: '#ffb066',
  Truck: '#ff5d5d',
  Pedestrian: '#3ddc84',
  Person_sitting: '#7cf0a8',
  Cyclist: '#36c5ff',
  Tram: '#c38cff',
  Misc: '#b0b4c0',
};
export const GT_COLOR = '#ffffff';

export function classColor(label) {
  return CLASS_COLORS[label] ?? CLASS_COLORS.Misc;
}

// Bottom face 0-3, top face 4-7, counter-clockwise from front-left seen from above.
const CORNERS = [
  [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
  [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
];
const EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
  [0, 5], [1, 4], // cross on the front face marks the heading
];

export function boxCorners(b) {
  const c = Math.cos(b.yaw);
  const s = Math.sin(b.yaw);
  return CORNERS.map(([sx, sy, sz]) => {
    const lx = (sx * b.dx) / 2;
    const ly = (sy * b.dy) / 2;
    return [b.x + c * lx - s * ly, b.y + s * lx + c * ly, b.z + (sz * b.dz) / 2];
  });
}

// All boxes of a set are merged into one LineSegments geometry (one draw call),
// with per-vertex colors so classes can share it.
export function buildBoxes(boxes, { colorFor, dashed = false }) {
  const positions = new Float32Array(boxes.length * EDGES.length * 6);
  const colors = new Float32Array(positions.length);
  const color = new THREE.Color();
  let k = 0;
  for (const b of boxes) {
    const corners = boxCorners(b);
    color.set(colorFor(b));
    for (const [a, z] of EDGES) {
      for (const p of [corners[a], corners[z]]) {
        positions.set(p, k);
        colors[k] = color.r; colors[k + 1] = color.g; colors[k + 2] = color.b;
        k += 3;
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const material = dashed
    ? new THREE.LineDashedMaterial({ vertexColors: true, dashSize: 0.3, gapSize: 0.2 })
    : new THREE.LineBasicMaterial({ vertexColors: true });
  const lines = new THREE.LineSegments(geometry, material);
  if (dashed) lines.computeLineDistances();
  return lines;
}
