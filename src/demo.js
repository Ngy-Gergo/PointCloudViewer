// Synthetic fallback scene, used only when public/data/frames.json lists no frames.
// It ray-casts a 64-beam spinning LiDAR against a ground plane, walls and boxes,
// so the viewer can be tried without any KITTI data. Clearly labeled as synthetic in the UI.

const SENSOR_HEIGHT = 1.73;
const GROUND = -SENSOR_HEIGHT;

function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SIZES = {
  Car: [3.9, 1.6, 1.5],
  Pedestrian: [0.8, 0.6, 1.75],
  Cyclist: [1.76, 0.6, 1.73],
};

// [label, x, y, yaw, speed (m/frame along heading)]
const ACTORS = [
  ['Car', 8, -3.5, 0.03, 1.2],
  ['Car', 16, 3.3, Math.PI + 0.02, 1.5],
  ['Car', 24, -3.6, 0.0, 1.0],
  ['Car', 34, 3.4, Math.PI, 1.4],
  ['Car', -9, -3.4, 0.01, 1.3],
  ['Car', 11, -8.6, 0.0, 0],
  ['Car', 16, -8.7, 0.02, 0],
  ['Car', 21, 8.5, Math.PI, 0],
  ['Pedestrian', 9, 9.6, -Math.PI / 2, 0.15],
  ['Pedestrian', 13, 9.8, Math.PI / 2, 0.12],
  ['Cyclist', 19, -6.4, 0.06, 0.5],
];
const POLES = [[5, -10.4], [15, -10.4], [25, -10.4], [10, 10.6], [30, 10.6]];

function intersectBox(ox, oy, oz, dx, dy, dz, b) {
  // Ray in the box's local frame, then slab test.
  const c = Math.cos(-b.yaw);
  const s = Math.sin(-b.yaw);
  const px = ox - b.x, py = oy - b.y, pz = oz - b.z;
  const lo = [c * px - s * py, s * px + c * py, pz];
  const ld = [c * dx - s * dy, s * dx + c * dy, dz];
  const half = [b.dx / 2, b.dy / 2, b.dz / 2];
  let tmin = 0, tmax = Infinity;
  for (let a = 0; a < 3; a++) {
    if (Math.abs(ld[a]) < 1e-9) {
      if (Math.abs(lo[a]) > half[a]) return Infinity;
      continue;
    }
    let t1 = (-half[a] - lo[a]) / ld[a];
    let t2 = (half[a] - lo[a]) / ld[a];
    if (t1 > t2) [t1, t2] = [t2, t1];
    tmin = Math.max(tmin, t1);
    tmax = Math.min(tmax, t2);
    if (tmin > tmax) return Infinity;
  }
  return tmin > 0 ? tmin : Infinity;
}

function scan(objects, rand) {
  const out = [];
  const beams = 64;
  const steps = 1800;
  for (let r = 0; r < beams; r++) {
    const el = ((-24.9 + (26.9 * r) / (beams - 1)) * Math.PI) / 180;
    const ce = Math.cos(el), se = Math.sin(el);
    for (let a = 0; a < steps; a++) {
      const az = (a / steps) * 2 * Math.PI;
      const dx = ce * Math.cos(az), dy = ce * Math.sin(az), dz = se;
      let t = 80, intensity = 0;
      if (dz < 0) {
        const tg = GROUND / dz;
        if (tg < t) { t = tg; intensity = 0.12 + 0.1 * rand(); }
      }
      for (const wallY of [11.2, -11.2]) {
        if (dy * wallY > 0) {
          const tw = wallY / dy;
          const z = tw * dz;
          if (tw < t && z > GROUND && z < 4) { t = tw; intensity = 0.25 + 0.15 * rand(); }
        }
      }
      for (const o of objects) {
        const tb = intersectBox(0, 0, 0, dx, dy, dz, o);
        if (tb < t) { t = tb; intensity = o.refl + 0.1 * rand(); }
      }
      if (t >= 80) continue;
      t += (rand() - 0.5) * 0.04;
      out.push(t * dx, t * dy, t * dz, Math.min(1, intensity));
    }
  }
  return new Float32Array(out);
}

function perturb(b, rand, pos, yawNoise, score) {
  return {
    ...b,
    x: b.x + (rand() - 0.5) * pos,
    y: b.y + (rand() - 0.5) * pos,
    z: b.z + (rand() - 0.5) * pos * 0.5,
    dx: b.dx * (1 + (rand() - 0.5) * 0.1),
    dy: b.dy * (1 + (rand() - 0.5) * 0.1),
    yaw: b.yaw + (rand() - 0.5) * yawNoise,
    score,
  };
}

function fakePredictions(gt, rand, { pos, yawNoise, missRate, falsePositive }) {
  const preds = [];
  for (const b of gt) {
    if (rand() < missRate) continue;
    const base = b.label === 'Car' ? 0.75 : 0.45;
    preds.push(perturb(b, rand, pos, yawNoise, Math.min(0.99, base + rand() * 0.24)));
  }
  if (falsePositive) {
    preds.push({ x: 27, y: -9.2, z: GROUND + 0.75, dx: 3.6, dy: 1.6, dz: 1.5, yaw: 0.4, label: 'Car', score: 0.21 });
  }
  return preds;
}

export function makeDemoFrames(count = 3) {
  const frames = [];
  for (let f = 0; f < count; f++) {
    const rand = mulberry32(1000 + f);
    const gt = ACTORS.map(([label, x, y, yaw, speed]) => {
      const [dx, dy, dz] = SIZES[label];
      // Ego drives forward 1.2 m/frame; others move along their heading.
      return {
        x: x + f * (Math.cos(yaw) * speed - 1.2),
        y: y + f * Math.sin(yaw) * speed,
        z: GROUND + dz / 2,
        dx, dy, dz, yaw, label, score: 1,
      };
    }).filter((b) => Math.hypot(b.x, b.y) > 3);
    const poles = POLES.map(([x, y]) => ({ x: x - f * 1.2, y, z: GROUND + 2, dx: 0.25, dy: 0.25, dz: 4, yaw: 0, refl: 0.6 }));
    const objects = [...gt.map((b) => ({ ...b, refl: b.label === 'Car' ? 0.45 : 0.3 })), ...poles];

    frames.push({
      id: `synthetic-${String(f).padStart(3, '0')}`,
      synthetic: true,
      points: scan(objects, rand),
      gt,
      preds: {
        'demo-voxel': fakePredictions(gt, rand, { pos: 0.25, yawNoise: 0.08, missRate: 0.05, falsePositive: false }),
        'demo-pillar': fakePredictions(gt, rand, { pos: 0.45, yawNoise: 0.15, missRate: 0.15, falsePositive: true }),
      },
      warnings: [],
    });
  }
  return frames;
}
