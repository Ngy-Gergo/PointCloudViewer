import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import GUI from 'lil-gui';
import { loadManifest, loadManifestFrame, loadDroppedFiles } from './loaders.js';
import { buildPointCloud, colorPoints } from './points.js';
import { buildBoxes, classColor, GT_COLOR } from './boxes.js';
import { makeDemoFrames } from './demo.js';

// ---------- scene ----------
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x0b0d12);
document.getElementById('app').appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 500);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// KITTI LiDAR is z-up (x forward, y left); Three.js is y-up. Instead of converting
// every point and box, everything lives in a group rotated -90° around x,
// which maps LiDAR (x, y, z) to Three (x, z, -y). Boxes keep their native yaw.
const world = new THREE.Group();
world.rotation.x = -Math.PI / 2;
scene.add(world);

const grid = new THREE.GridHelper(160, 32, 0x2a3040, 0x1a1e28);
grid.position.y = -1.73; // ground under a KITTI sensor mounted at 1.73 m
scene.add(grid);
world.add(new THREE.AxesHelper(2)); // red = forward (x), green = left (y), blue = up (z)

// point_cloud_range of the CenterPoint models (LiDAR_Obj_Detection kitti_car_common.py):
// x 0..67.2 m, y -38.4..38.4 m. Objects outside it can never be predicted.
const DETECTION_RANGE = [0, -38.4, 67.2, 38.4];
const rangeOutline = (() => {
  const [x0, y0, x1, y1] = DETECTION_RANGE;
  const z = -1.7; // just above the grid so the outline is not hidden by it
  const geometry = new THREE.BufferGeometry().setFromPoints(
    [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(([x, y]) => new THREE.Vector3(x, y, z)),
  );
  return new THREE.LineLoop(geometry, new THREE.LineBasicMaterial({ color: 0x4a8cff }));
})();
world.add(rangeOutline);

function resetView() {
  camera.position.set(-14, 11, 0); // LiDAR (-14, 0, 11): behind and above the ego vehicle
  controls.target.set(14, 0, 0);
  controls.update();
}
resetView();

// ---------- state ----------
const settings = {
  frame: '',
  colorMode: 'height',
  pointSize: 0.06,
  zMin: -2.0,
  zMax: 1.0,
  showGT: true,
  showPred: true,
  model: '',
  scoreThreshold: 0.3,
  showRange: true,
  prev: () => step(-1),
  next: () => step(1),
  openFiles: () => fileInput.click(),
  resetView,
};

let frames = []; // { id, entry?, data? }
let current = null; // loaded frame
let pointsObj = null;
let gtObj = null;
let predObj = null;

// ---------- UI ----------
const gui = new GUI({ title: 'Controls' });
let frameCtrl = gui.add(settings, 'frame', []).name('Frame');
const nav = gui.addFolder('Navigate');
nav.add(settings, 'prev').name('◀ Previous frame');
nav.add(settings, 'next').name('Next frame ▶');
nav.add(settings, 'resetView').name('Reset camera');
nav.add(settings, 'openFiles').name('Open files…');

const pts = gui.addFolder('Points');
pts.add(settings, 'colorMode', ['height', 'intensity']).name('Color by').onChange(recolor);
pts.add(settings, 'pointSize', 0.01, 0.2, 0.005).name('Point size').onChange((v) => {
  if (pointsObj) pointsObj.material.size = v;
});
pts.add(settings, 'zMin', -4, 0, 0.1).name('Height min (m)').onChange(recolor);
pts.add(settings, 'zMax', -1, 4, 0.1).name('Height max (m)').onChange(recolor);

const bx = gui.addFolder('Boxes');
bx.add(settings, 'showGT').name('Ground truth').onChange(rebuildBoxes);
bx.add(settings, 'showPred').name('Predictions').onChange(rebuildBoxes);
let modelCtrl = bx.add(settings, 'model', []).name('Model');
bx.add(settings, 'scoreThreshold', 0, 1, 0.01).name('Score ≥').onChange(rebuildBoxes);
bx.add(settings, 'showRange').name('Detection range').onChange((v) => { rangeOutline.visible = v; });

function setFrameOptions() {
  frameCtrl.destroy();
  frameCtrl = gui.add(settings, 'frame', frames.map((f) => f.id)).name('Frame')
    .onChange((id) => selectFrame(frames.findIndex((f) => f.id === id)));
  // keep the frame selector at the top
  gui.$children.prepend(frameCtrl.domElement);
}

function setModelOptions(models) {
  if (!models.includes(settings.model)) settings.model = models[0] ?? '';
  modelCtrl.destroy();
  modelCtrl = bx.add(settings, 'model', models).name('Model').onChange(rebuildBoxes);
  bx.$children.insertBefore(modelCtrl.domElement, bx.$children.children[2] ?? null);
}

const fileInput = Object.assign(document.createElement('input'), { type: 'file', multiple: true });
fileInput.addEventListener('change', () => handleFiles(fileInput.files));

// ---------- rendering of a frame ----------
function disposeObj(obj) {
  if (!obj) return;
  obj.parent?.remove(obj);
  obj.geometry.dispose();
  obj.material.dispose();
}

function recolor() {
  if (pointsObj) colorPoints(pointsObj, current.points, settings);
}

function visiblePreds() {
  const boxes = current?.preds[settings.model] ?? [];
  return boxes.filter((b) => b.score >= settings.scoreThreshold);
}

function rebuildBoxes() {
  disposeObj(gtObj);
  disposeObj(predObj);
  gtObj = predObj = null;
  if (!current) return;
  if (settings.showGT && current.gt?.length) {
    gtObj = buildBoxes(current.gt, { colorFor: () => GT_COLOR, dashed: true });
    world.add(gtObj);
  }
  const preds = visiblePreds();
  if (settings.showPred && preds.length) {
    predObj = buildBoxes(preds, { colorFor: (b) => classColor(b.label) });
    world.add(predObj);
  }
  updateInfo();
}

function showFrame(frame) {
  current = frame;
  disposeObj(pointsObj);
  pointsObj = buildPointCloud(frame.points, settings.pointSize);
  world.add(pointsObj);
  recolor();
  setModelOptions(Object.keys(frame.preds));
  rebuildBoxes();
}

async function selectFrame(i) {
  if (i < 0 || i >= frames.length) return;
  const f = frames[i];
  settings.frame = f.id;
  frameCtrl.updateDisplay();
  try {
    f.data ??= await loadManifestFrame(f.entry);
    showFrame(f.data);
  } catch (e) {
    showError(`Could not load ${f.id}: ${e.message}`);
  }
}

function step(delta) {
  const i = frames.findIndex((f) => f.id === settings.frame);
  selectFrame((i + delta + frames.length) % frames.length);
}

// ---------- overlay ----------
const infoEl = document.getElementById('info');
const legendEl = document.getElementById('legend');

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
}

function updateInfo() {
  if (!current) return;
  const nPreds = current.preds[settings.model]?.length ?? 0;
  const shown = visiblePreds();
  const lines = [
    `<div>Frame: <b>${escapeHtml(current.id)}</b></div>`,
    `<div>Points: ${(current.points.length / 4).toLocaleString('en-US')}</div>`,
    `<div>Ground truth boxes: ${current.gt ? current.gt.length : '—'}</div>`,
    `<div>Predictions (${escapeHtml(settings.model || '—')}): ${shown.length} / ${nPreds}</div>`,
  ];
  if (current.synthetic) {
    lines.push('<div class="warn">Synthetic demo scene — no KITTI data loaded. Drop real files or add them to public/data/frames.json.</div>');
  }
  for (const w of current.warnings) lines.push(`<div class="warn">${escapeHtml(w)}</div>`);
  infoEl.innerHTML = lines.join('');

  const labels = [...new Set(shown.map((b) => b.label))].sort();
  legendEl.innerHTML = [
    current.gt?.length ? `<span><i class="dashed"></i>GT</span>` : '',
    ...labels.map((l) => `<span><i style="background:${classColor(l)}"></i>${escapeHtml(l)}</span>`),
  ].join('');
}

function showError(msg) {
  infoEl.innerHTML = `<div class="warn">${escapeHtml(msg)}</div>` + infoEl.innerHTML;
}

// ---------- drag & drop ----------
const dropZone = document.getElementById('drop-zone');
let dragDepth = 0;
window.addEventListener('dragenter', (e) => { e.preventDefault(); dragDepth++; dropZone.classList.add('active'); });
window.addEventListener('dragleave', () => { if (--dragDepth <= 0) dropZone.classList.remove('active'); });
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => {
  e.preventDefault();
  dragDepth = 0;
  dropZone.classList.remove('active');
  handleFiles(e.dataTransfer.files);
});

async function handleFiles(files) {
  if (!files?.length) return;
  try {
    const frame = await loadDroppedFiles(files, current?.synthetic ? null : current);
    if (frame !== current) {
      frame.id = frames.some((f) => f.id === frame.id) ? `${frame.id} (dropped)` : frame.id;
      frames.push({ id: frame.id, data: frame });
      setFrameOptions();
      settings.frame = frame.id;
      frameCtrl.updateDisplay();
      showFrame(frame);
    } else {
      showFrame(frame);
    }
  } catch (e) {
    showError(e.message);
  }
}

window.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement) return;
  if (e.key === 'ArrowLeft') step(-1);
  if (e.key === 'ArrowRight') step(1);
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

// ---------- start ----------
loadManifest().then((manifest) => {
  frames = manifest.length
    ? manifest.map((entry) => ({ id: entry.id, entry }))
    : makeDemoFrames().map((data) => ({ id: data.id, data }));
  setFrameOptions();
  selectFrame(0);
});
