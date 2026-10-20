"""Export matched road/background ablations from an existing checkpoint.

Run from the repository root with PYTHONPATH=. See docs/road_render_diagnostics.md.
Heavy project/CUDA dependencies are imported only after argument validation.
"""

import argparse
import csv
import hashlib
import html
import json
import logging
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from utils.road_render_diagnostics import (
    depth_edges, expected_depth_normals, normalize_rgb, render_layers, roi_mask, summarize,
)


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO.parent / "output/qcraft_20260613_102138_QCJPTDJ07384_3298_3308/20261019_lidar+traincam2_3_5_9_12_0_1_6_7_10_11sdf-heightv2"
THRESHOLDS = dict(support_alpha=0.05, coverage_alpha=0.9, background_alpha=0.1,
                  other_alpha=0.1, dark_luma=0.15, relative_depth_jump=0.05)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--checkpoint", type=Path, help="Overrides run-dir/group0/checkpoint_final.pth")
    p.add_argument("--output-dir", type=Path, default=REPO / "diagnostics/road_height_v2")
    p.add_argument("--frames", type=int, nargs="+", default=[0, 10, 40, 50, 79, 98],
                   help="Zero-based indices into the complete 99-frame timeline")
    p.add_argument("--cam-ids", type=int, nargs="+", default=[2, 3, 5, 9, 12])
    p.add_argument("--traj-types", nargs="+", default=["original_traj", "left_shift_3m"])
    p.add_argument("--downscale", type=float, default=1)
    p.add_argument("--roi-file", type=Path,
                   help='JSON list: [{"name":"artifact","camera":9,"frame":79,"traj":"left_shift_3m","box":[0,.5,1,1]}]')
    p.add_argument("--reference-dir", type=Path,
                   help="Existing novel_traj directory; default run-dir/novel_traj")
    p.add_argument("--depth-max", type=float, default=100., help="Shared depth PNG range in metres")
    p.add_argument("--skip-arrays", action="store_true", help="Skip float32 NPZ exports to save disk space")
    p.add_argument("--overwrite", action="store_true", help="Allow replacing an existing manifest")
    p.add_argument("--opts", nargs="*", default=[], help="OmegaConf overrides, e.g. data.data_root=/new/path")
    return p


def numpy(value):
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def save_rgb(path, value):
    Image.fromarray(np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)).save(path)


def scalar_rgb(value, maximum=1., valid=None):
    """Shared fixed scale: near/small black, far/large white; invalid magenta."""
    image = np.repeat(np.clip(value / maximum, 0, 1)[..., None], 3, -1)
    if valid is not None:
        image[~valid] = [1, 0, 1]
    return image


def save_panel(path, cells):
    width, height = 480, 270
    panel = Image.new("RGB", (width * 3, (height + 30) * ((len(cells) + 2) // 3)), "white")
    draw = ImageDraw.Draw(panel)
    for index, (label, value) in enumerate(cells):
        x, y = index % 3 * width, index // 3 * (height + 30)
        draw.text((x + 8, y + 8), label, fill="black")
        im = Image.fromarray(np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8))
        im.thumbnail((width, height))
        panel.paste(im, (x, y + 30))
    panel.save(path)


def compose(trainer, output, infos):
    sky = trainer.models["Sky"](infos)
    return trainer.affine_transformation(
        output["rgb_gaussians"] + sky * (1 - output["opacity"]), infos).clamp(0, 1)


def matching_rois(rois, traj, camera, frame, shape):
    yield "lower_half_context", roi_mask(shape, [0, .5, 1, 1])
    for roi in rois:
        if (roi.get("traj", traj) == traj and roi.get("camera", camera) == camera
                and roi.get("frame", frame) == frame):
            yield roi["name"], roi_mask(shape, roi["box"])


def export_frame(out, layers, contribution, composed, without_bg, ego, intrinsics, depth_max, skip_arrays):
    arrays = {"contribution": contribution, "composed_rgb": composed,
              "no_background_composed_rgb": without_bg, "ego_mask": ego}
    cells = [("Full (sky + affine; before ego blend)", composed),
             ("No Background (sky + affine)", without_bg),
             ("Visible alpha: R=road G=background B=other", contribution)]
    save_rgb(out / "full_composed.png", composed)
    save_rgb(out / "no_background_composed.png", without_bg)
    save_rgb(out / "visible_contributions.png", contribution)
    for channel, name in enumerate(("road", "background", "other")):
        save_rgb(out / f"visible_{name}_alpha.png", scalar_rgb(contribution[..., channel]))
    delta = np.abs(composed - without_bg).mean(-1)
    save_rgb(out / "background_removal_absdiff_x5.png", scalar_rgb(delta, .2))
    for name, data in layers.items():
        rgb, depth, alpha = data["rgb_gaussians"], data["depth"][..., 0], data["opacity"][..., 0]
        normalized, supported = normalize_rgb(rgb, alpha, THRESHOLDS["support_alpha"])
        valid = supported & np.isfinite(depth) & (depth > 0)
        edge = depth_edges(depth, valid)
        normalized_display = normalized.copy()
        normalized_display[~supported] = [1, 0, 1]
        depth_image = scalar_rgb(depth, depth_max, valid)
        save_rgb(out / f"{name}_rgb_black.png", rgb)
        save_rgb(out / f"{name}_rgb_normalized.png", normalized_display)
        save_rgb(out / f"{name}_alpha.png", scalar_rgb(alpha))
        save_rgb(out / f"{name}_depth.png", depth_image)
        save_rgb(out / f"{name}_depth_edges.png", scalar_rgb(edge, .1, valid))
        arrays.update({f"{name}_rgb_premultiplied": rgb, f"{name}_depth": depth,
                       f"{name}_alpha": alpha, f"{name}_depth_valid": valid})
        if name in ("full", "road"):
            normals, normal_valid = expected_depth_normals(depth, valid, intrinsics)
            normal_valid &= edge <= THRESHOLDS["relative_depth_jump"]
            normal_image = normals * .5 + .5
            normal_image[~normal_valid] = [1, 0, 1]
            save_rgb(out / f"{name}_expected_depth_normals.png", normal_image)
            arrays[f"{name}_expected_depth_normals"] = normals
            arrays[f"{name}_normal_valid"] = normal_valid
        if name in ("road", "background"):
            cells.extend([(f"{name}: RGB / alpha (invalid magenta)", normalized_display),
                          (f"{name}: isolated accumulated alpha", scalar_rgb(alpha)),
                          (f"{name}: expected camera Z, 0-{depth_max:g}m", depth_image)])
    cells.extend([("Road relative depth jump, 0-10%", scalar_rgb(
        depth_edges(layers["road"]["depth"][..., 0], arrays["road_depth_valid"]), .1,
        arrays["road_depth_valid"])),
        ("Full Gaussian RGB on black", layers["full"]["rgb_gaussians"]),
        ("Background removal abs difference x5", scalar_rgb(delta, .2))])
    save_panel(out / "panel.jpg", cells)
    if not skip_arrays:
        np.savez_compressed(out / "arrays.npz", **arrays)


def main(args):
    if args.downscale <= 0 or args.depth_max <= 0 or min(args.frames) < 0:
        raise ValueError("downscale/depth-max must be positive and frames nonnegative")
    checkpoint = (args.checkpoint or args.run_dir / "group0/checkpoint_final.pth").resolve()
    config_path = checkpoint.parent / "config.yaml"
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint/config missing: {checkpoint}, {config_path}")
    out_root = args.output_dir.resolve()
    manifest_path = out_root / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Use another --output-dir or --overwrite: {manifest_path}")
    rois = json.loads(args.roi_file.read_text()) if args.roi_file else []
    for roi in rois:
        roi_mask((10, 10), roi["box"])
        if not roi.get("name"):
            raise ValueError("Every ROI needs a nonempty name")

    import torch
    from omegaconf import OmegaConf
    from datasets.driving_dataset_novel_view import DrivingDatasetNovelView
    from utils.misc import import_str

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gsplat rendering; no rendering was performed")
    cfg = OmegaConf.merge(OmegaConf.load(config_path), OmegaConf.from_dotlist(args.opts))
    # Do not subset training camera metadata: embeddings and image IDs must match checkpoint.
    dataset = DrivingDatasetNovelView(data_cfg=cfg.data)
    if max(args.frames) >= dataset.frame_num:
        raise ValueError(f"Frame index exceeds timeline length {dataset.frame_num}")
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer, num_timesteps=dataset.num_img_timesteps, model_config=cfg.model,
        num_full_images=len(dataset.train_indices + dataset.test_indices),
        test_set_indices=dataset.test_timesteps, scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=torch.device("cuda"))
    trainer.resume_from_checkpoint(ckpt_path=str(checkpoint), load_only_model=True)
    trainer.set_eval()
    if not {"RoadNodes", "Background"} <= trainer.gaussian_classes.keys():
        raise ValueError("This diagnostic requires RoadNodes and Background")
    cameras = dataset.load_specified_cameras(args.cam_ids, [args.downscale] * len(args.cam_ids))
    out_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_root / "resolved_config.yaml")
    stat = checkpoint.stat()
    manifest = dict(
        status="running", checkpoint=str(checkpoint), checkpoint_size=stat.st_size,
        checkpoint_mtime_ns=stat.st_mtime_ns, checkpoint_step=int(trainer.step),
        frames=args.frames, cameras=args.cam_ids, trajectories=args.traj_types,
        downscale=args.downscale, thresholds=THRESHOLDS, rois=rois,
        torch_version=torch.__version__, device=torch.cuda.get_device_name(),
        command=sys.argv, code_sha256={}, samples=[],
        notes=["Pinhole raster coordinates; fisheye remapping is deliberately not applied.",
               "ED is alpha-normalized expected camera-Z depth, not a measured road surface.",
               "Isolated layers remove occluders; only one-hot contributions preserve full-scene visibility.",
               "lower_half_context is a context ROI, not a road segmentation mask.",
               "Statistics exclude ego and other-visible pixels; thresholds are diagnostic choices.",
               "No independent novel-view GT or automatic causal artifact classification."])
    for name in ("models/trainers/scene_graph.py", "models/trainers/base.py", "models/nodes/road.py",
                 "models/gaussians/vanilla.py", "datasets/base/pixel_source.py",
                 "utils/camera.py", "utils/road_render_diagnostics.py", "tools/diagnose_road_render.py"):
        manifest["code_sha256"][name] = hashlib.sha256((REPO / name).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    rows = []
    with torch.no_grad():
        for traj_name in args.traj_types:
            # Generate the complete timeline BEFORE selecting frames, preserving time/pose alignment.
            trajectory = dataset.get_novel_render_traj(traj_name, target_frames=dataset.frame_num)
            if trajectory is None:
                raise ValueError(f"Cannot generate trajectory {traj_name}")
            for camera_id, camera_data in cameras.items():
                timeline = dataset.prepare_novel_view_render_data(
                    traj=trajectory, target_cam_data=camera_data,
                    is_original_traj=traj_name == "original_traj")
                for frame in args.frames:
                    data = {group: {key: value.to("cuda") if torch.is_tensor(value) else value
                                    for key, value in values.items()}
                            for group, values in timeline[frame].items()}
                    infos = data["image_infos"]
                    full, gs = trainer(image_infos=infos, camera_infos=data["cam_infos"],
                                       novel_view=True, render_only=True)
                    cam = trainer.process_camera(camera_infos=data["cam_infos"],
                                                 image_ids=infos["img_idx"].flatten()[0], novel_view=True)
                    layers_t, contribution_t, counts = render_layers(trainer, gs, cam, full)
                    composed = numpy(full["rgb"].clamp(0, 1))
                    no_bg = numpy(compose(trainer, layers_t["no_background"], infos))
                    layers = {name: {key: numpy(value) for key, value in layer.items()}
                              for name, layer in layers_t.items()}
                    contribution = numpy(contribution_t)
                    shape = composed.shape[:2]
                    ego = numpy(infos["egocar_masks"]).astype(bool) if "egocar_masks" in infos else np.zeros(shape, bool)
                    ego = ego.reshape(shape)
                    relative = Path(traj_name) / f"frame{frame:06d}_cam{camera_id}"
                    out = out_root / relative
                    out.mkdir(parents=True, exist_ok=True)
                    export_frame(out, layers, contribution, composed, no_bg, ego,
                                 numpy(cam.Ks), args.depth_max, args.skip_arrays)
                    pose = dict(camera_to_world=numpy(cam.camtoworlds).tolist(),
                                intrinsics=numpy(cam.Ks).tolist(), height=int(cam.H), width=int(cam.W),
                                normed_time=float(infos["normed_time"].flatten()[0]),
                                image_id=int(infos["img_idx"].flatten()[0]))
                    (out / "camera.json").write_text(json.dumps(pose, indent=2))
                    sample = dict(trajectory=traj_name, frame=frame, camera=int(camera_id),
                                  path=str(relative), gaussian_counts=counts, camera_metadata=pose)
                    # Compare the normal display path to existing PNGs only when coordinate systems match.
                    reference_dir = args.reference_dir or args.run_dir / "novel_traj"
                    reference = reference_dir / f"{traj_name}_step{trainer.step}" / "images" / f"{frame:06d}" / f"{camera_id}_rgbs.png"
                    if getattr(camera_data, "is_fisheye", False):
                        sample["reference_check"] = "skipped: existing export uses fisheye remapping"
                    elif reference.exists():
                        display = composed.copy()
                        template = getattr(camera_data, "ego_template", None)
                        if template is not None:
                            display[ego] = numpy(template)[ego]
                        with Image.open(reference) as image:
                            ref = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
                        if ref.shape == display.shape:
                            difference = np.abs(ref - display)
                            sample["reference_check"] = dict(path=str(reference),
                                mae=float(difference.mean()), max_abs=float(difference.max()),
                                fraction_gt_2_levels=float((difference.max(-1) > 2 / 255).mean()))
                        else:
                            sample["reference_check"] = "skipped: reference resolution differs"
                    else:
                        sample["reference_check"] = "skipped: no existing reference PNG"
                    for roi_name, region in matching_rois(rois, traj_name, camera_id, frame, shape):
                        stats = summarize(layers, contribution, composed, no_bg, region & ~ego, THRESHOLDS)
                        rows.append(dict(trajectory=traj_name, frame=frame, camera=int(camera_id), roi=roi_name, **stats))
                        if stats["contribution_alpha_max_abs_error"] > 1e-3:
                            raise RuntimeError("One-hot contribution sum differs from full alpha; check renderer consistency")
                    (out / "sample.json").write_text(json.dumps(sample, indent=2))
                    manifest["samples"].append(sample)
                    manifest_path.write_text(json.dumps(manifest, indent=2))
                    logging.info("Exported %s", relative)
                    del full, gs, layers_t, contribution_t, data
                del timeline
    with (out_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    links = "\n".join(
        f'<h3>{html.escape(s["path"])}</h3><a href="{s["path"]}/panel.jpg">'
        f'<img loading="lazy" width="1200" src="{s["path"]}/panel.jpg"></a>'
        for s in manifest["samples"])
    (out_root / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Road layer diagnostics</title>'
        '<h1>Road layer diagnostics</h1><p>R=road, G=background, B=other visible alpha. '
        'Isolated layers remove occlusion. Magenta marks invalid depth/normalized RGB. '
        'All depth panels share the same range. Statistics are associations, not causal labels.</p>' + links)
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Diagnostics saved to {out_root}; open index.html and summary.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main(parser().parse_args())
