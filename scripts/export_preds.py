"""Export KITTI frames + CenterPoint predictions for the web viewer.

Run on the machine that has the LiDAR_Obj_Detection environment (mmdet3d,
`pip install -e research`), the KITTI data and the trained checkpoints, from the
LiDAR_Obj_Detection repository root:

    python /path/to/PointCloudViewer/scripts/export_preds.py \
        --kitti-root data/KITTI_Obj_Detect \
        --frames 000008 000015 000019 \
        --model voxel0075 research/configs/centerpoint/voxel0075.py /path/to/voxel0075.pth \
        --model pillar02 research/configs/centerpoint/tuning/pillar02_duration30.py /path/to/pillar02.pth \
        --out-dir /path/to/PointCloudViewer/public/data

For each frame it copies the full velodyne scan, label_2 and calib files, runs
every model on the same `velodyne_reduced` input the evaluation used, writes
`<frame>_<model>.json`, and rewrites `frames.json`. Use validation-split frames;
predictions on training frames would overstate what the models do.
"""
import argparse
import json
import shutil
from pathlib import Path

from mmdet3d.apis import inference_detector, init_model


def predict(model, classes, bin_path, min_score):
    result = inference_detector(model, str(bin_path))
    if isinstance(result, tuple):  # mmdet3d 1.x returns (result, data)
        result = result[0]
    pred = result.pred_instances_3d
    boxes = pred.bboxes_3d.tensor.cpu().numpy()  # x, y, z_bottom, dx, dy, dz, yaw
    scores = pred.scores_3d.cpu().numpy()
    labels = pred.labels_3d.cpu().numpy()
    return [
        {
            "x": round(float(b[0]), 3), "y": round(float(b[1]), 3), "z": round(float(b[2]), 3),
            "dx": round(float(b[3]), 3), "dy": round(float(b[4]), 3), "dz": round(float(b[5]), 3),
            "yaw": round(float(b[6]), 4),
            "label": classes[int(l)],
            "score": round(float(s), 4),
        }
        for b, s, l in zip(boxes, scores, labels)
        if s >= min_score
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kitti-root", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", required=True, help="KITTI sample ids, e.g. 000008")
    parser.add_argument("--model", nargs=3, action="append", required=True,
                        metavar=("NAME", "CONFIG", "CHECKPOINT"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.10,
                        help="0.10 matches the presentation threshold; the viewer slider filters further")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    training = args.kitti_root / "training"
    val_list = args.kitti_root / "ImageSets" / "val.txt"
    if val_list.exists():
        val_ids = set(val_list.read_text().split())
        not_val = [f for f in args.frames if f not in val_ids]
        if not_val:
            parser.error(f"not in the validation split: {', '.join(not_val)}")
    else:
        print(f"warning: {val_list} not found, cannot check that frames are validation frames")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for frame in args.frames:
        shutil.copyfile(training / "velodyne" / f"{frame}.bin", args.out_dir / f"{frame}.bin")
        shutil.copyfile(training / "label_2" / f"{frame}.txt", args.out_dir / f"{frame}_label.txt")
        shutil.copyfile(training / "calib" / f"{frame}.txt", args.out_dir / f"{frame}_calib.txt")
        manifest.append({
            "id": frame,
            "bin": f"{frame}.bin",
            "gt": {"label": f"{frame}_label.txt", "calib": f"{frame}_calib.txt"},
            "predictions": [],
        })

    for name, config, checkpoint in args.model:
        model = init_model(config, checkpoint, device=args.device)
        classes = model.dataset_meta["classes"]
        for entry in manifest:
            frame = entry["id"]
            reduced = training / "velodyne_reduced" / f"{frame}.bin"
            source = reduced if reduced.exists() else training / "velodyne" / f"{frame}.bin"
            boxes = predict(model, classes, source, args.min_score)
            out_name = f"{frame}_{name}.json"
            (args.out_dir / out_name).write_text(json.dumps({
                "frame": frame,
                "model": name,
                "checkpoint": Path(checkpoint).name,
                "z_origin": "bottom",
                "boxes": boxes,
            }, indent=1))
            entry["predictions"].append({"model": name, "file": out_name})
            print(f"{name} {frame}: {len(boxes)} boxes (input {source.parent.name})")
        del model

    (args.out_dir / "frames.json").write_text(json.dumps({"frames": manifest}, indent=2))
    print(f"wrote {args.out_dir / 'frames.json'}")


if __name__ == "__main__":
    main()
