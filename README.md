# PointCloudViewer

Browser-based 3D viewer for KITTI LiDAR point clouds, built with Three.js.
It colors points by height and draws 3D bounding boxes from KITTI ground-truth
labels and from detector predictions exported as JSON.

**Live demo:** https://ngy-gergo.github.io/PointCloudViewer/

<!-- Add a screenshot or GIF here: docs/screenshot.png -->

## Features

- Loads KITTI velodyne `.bin` scans (float32 × 4 per point: x, y, z, intensity)
- Colors points by height (Turbo colormap) or intensity, with adjustable range and point size
- Draws ground-truth boxes from KITTI `label_2` + `calib` files (dashed white)
- Draws predicted boxes from JSON, colored by class, with a score threshold slider
- Switches between several models' predictions for the same frame (for example voxel vs. pillar)
- Steps through frames with buttons or ←/→, and orbits, zooms and pans with OrbitControls
- Accepts files dropped onto the page, so you can inspect local data without rebuilding
- Falls back to a ray-cast synthetic scene, labeled as synthetic, when no data is configured

## Run locally

```bash
npm install
npm run dev
```

## Adding frames

Put the files in `public/data/` and list them in `public/data/frames.json`:

```json
{
  "frames": [
    {
      "id": "000008",
      "bin": "000008.bin",
      "gt": { "label": "000008_label.txt", "calib": "000008_calib.txt" },
      "predictions": [
        { "model": "pointpillars", "file": "000008_pointpillars.json" }
      ]
    }
  ]
}
```

`gt` and `predictions` are optional.

### Prediction JSON

```json
{
  "frame": "000008",
  "model": "pointpillars",
  "z_origin": "bottom",
  "boxes": [
    { "x": 12.3, "y": -1.2, "z": -1.7, "dx": 3.9, "dy": 1.6, "dz": 1.5, "yaw": 0.12, "label": "Car", "score": 0.87 }
  ]
}
```

The boxes are in the LiDAR frame. `z_origin` is `"bottom"` by default, which matches mmdet3d's
`LiDARInstance3DBoxes`; use `"center"` if `z` is the box center.

### Exporting predictions from my CenterPoint models

The predictions come from the Car-only CenterPoint models trained in
[LiDAR_Obj_Detection](https://github.com/Ngy-Gergo/LiDAR_Obj_Detection): the voxel model (0.075 m voxels,
20 epochs) and the low-latency pillar model (0.2 m pillars, 30 epochs).
[`scripts/export_preds.py`](scripts/export_preds.py) runs from that repository's root in its mmdet3d
environment. It copies the KITTI scan, label and calib files, runs each model on the same
`velodyne_reduced` input the evaluation used, and writes the JSON files and `frames.json`:

```bash
python /path/to/PointCloudViewer/scripts/export_preds.py \
  --kitti-root data/KITTI_Obj_Detect \
  --frames 000008 000015 000019 \
  --model voxel0075 research/configs/centerpoint/voxel0075.py /path/to/voxel0075.pth \
  --model pillar02 research/configs/centerpoint/tuning/pillar02_duration30.py /path/to/pillar02.pth \
  --out-dir /path/to/PointCloudViewer/public/data
```

Only validation-split frames are accepted. The blue rectangle in the viewer is the models'
`point_cloud_range` (0–67.2 m ahead, ±38.4 m sideways). The models cannot predict objects outside it.

## How it works

- **Reading `.bin`:** `fetch` → `ArrayBuffer` → `Float32Array`. Every 4 values are one point.
- **Coordinates:** KITTI LiDAR is z-up (x forward, y left), and Three.js is y-up. All points and boxes
  live in one group rotated −90° around x, which maps (x, y, z) → (x, z, −y). Box yaw therefore stays a
  rotation about the LiDAR z axis.
- **GT labels:** KITTI labels are in rectified camera coordinates, with the location at the box bottom
  center. They are transformed with `Tr_velo_to_cam⁻¹ · R0_rect⁻¹`, and yaw becomes `−ry − π/2`.
- **Performance:** each scan is one `BufferGeometry` drawn as one `THREE.Points` (one draw call for
  ~120k points). Each box set is merged into one `LineSegments` geometry.

## Deploy

Pushing to `main` triggers `.github/workflows/deploy.yml`, which builds and publishes `dist/` to GitHub Pages.
In the repo settings, set **Pages → Source** to **GitHub Actions**.

## Data and license

Point clouds and labels come from the [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/)
(Geiger et al., CVPR 2012), licensed under
[CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). KITTI data is not covered by
this repository's license.
