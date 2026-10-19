import * as THREE from 'three';

// Polynomial approximation of Google's Turbo colormap (blue -> green -> yellow -> red).
function turbo(t, out, i) {
  t = Math.min(1, Math.max(0, t));
  out[i] = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  out[i + 1] = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  out[i + 2] = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
}

// One BufferGeometry + one Points object = one draw call for the whole scan.
export function buildPointCloud(data, size) {
  const n = data.length / 4;
  const positions = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    positions[3 * i] = data[4 * i];
    positions[3 * i + 1] = data[4 * i + 1];
    positions[3 * i + 2] = data[4 * i + 2];
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  geometry.computeBoundingSphere();
  const material = new THREE.PointsMaterial({ size, vertexColors: true, sizeAttenuation: true });
  return new THREE.Points(geometry, material);
}

// mode "height" maps z in [zMin, zMax]; mode "intensity" maps reflectance in [0, 1].
export function colorPoints(points, data, { mode, zMin, zMax }) {
  const colors = points.geometry.attributes.color;
  const arr = colors.array;
  const n = data.length / 4;
  const span = Math.max(1e-6, zMax - zMin);
  for (let i = 0; i < n; i++) {
    const t = mode === 'intensity' ? Math.sqrt(data[4 * i + 3]) : (data[4 * i + 2] - zMin) / span;
    turbo(t, arr, 3 * i);
  }
  colors.needsUpdate = true;
}
