#!/usr/bin/env python3
"""Create a confidence-masked Polycam point cloud for Gaussian Splatting init."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a point cloud from Polycam keyframe depth maps, mask invalid depth with the "
            "confidence maps, transform everything into the transforms.json world frame, save a "
            "PLY in the dataset root, and add ply_file_path to transforms.json."
        )
    )
    parser.add_argument("dataset_root", type=Path, help="Dataset directory containing transforms.json and keyframes/")
    parser.add_argument(
        "--point-count",
        type=int,
        default=1_000_000,
        help="Total number of points to sample across all frames.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="point_cloud_1m_confidence_masked.ply",
        help="Name of the PLY file to write inside dataset_root.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1e-3,
        help="Multiply raw depth values by this scale to convert them to meters.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=int,
        default=0,
        help="Only sample pixels whose confidence value is strictly greater than this threshold.",
    )
    return parser.parse_args()


def sorted_files(directory: Path, suffix: str) -> List[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == suffix)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_inputs(dataset_root: Path) -> Tuple[List[Path], List[Path], List[Path], List[Path], List[dict]]:
    transforms_path = dataset_root / "transforms.json"
    keyframes_root = dataset_root / "keyframes"
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing transforms.json at {transforms_path}")
    for subdir in ("images", "depth", "confidence", "cameras"):
        if not (keyframes_root / subdir).is_dir():
            raise FileNotFoundError(f"Missing keyframes/{subdir} in {dataset_root}")

    images = sorted_files(keyframes_root / "images", ".jpg")
    depths = sorted_files(keyframes_root / "depth", ".png")
    confidences = sorted_files(keyframes_root / "confidence", ".png")
    cameras = sorted_files(keyframes_root / "cameras", ".json")
    transforms = load_json(transforms_path)
    frames = transforms.get("frames", [])

    counts = {
        "images": len(images),
        "depths": len(depths),
        "confidences": len(confidences),
        "cameras": len(cameras),
        "frames": len(frames),
    }
    if len(set(counts.values())) != 1:
        raise ValueError(f"Mismatched dataset counts: {counts}")

    for idx, (image_path, depth_path, conf_path, camera_path) in enumerate(zip(images, depths, confidences, cameras)):
        stems = {image_path.stem, depth_path.stem, conf_path.stem, camera_path.stem}
        if len(stems) != 1:
            raise ValueError(
                f"Keyframe stem mismatch at index {idx}: "
                f"{image_path.name}, {depth_path.name}, {conf_path.name}, {camera_path.name}"
            )

    return images, depths, confidences, cameras, frames


def compute_targets(valid_counts: np.ndarray, total_points: int) -> np.ndarray:
    if valid_counts.sum() < total_points:
        raise ValueError(
            f"Requested {total_points} points, but only {int(valid_counts.sum())} valid confidence-masked pixels exist."
        )

    num_frames = len(valid_counts)
    targets = np.full(num_frames, total_points // num_frames, dtype=np.int64)
    targets[: total_points % num_frames] += 1

    if np.all(valid_counts >= targets):
        return targets

    targets = np.minimum(targets, valid_counts)
    deficit = int(total_points - targets.sum())
    while deficit > 0:
        eligible = np.flatnonzero(valid_counts > targets)
        if eligible.size == 0:
            break
        # Refill one point at a time per eligible frame to keep the distribution as even as possible.
        eligible = eligible[np.argsort(targets[eligible], kind="stable")]
        take = min(deficit, eligible.size)
        targets[eligible[:take]] += 1
        deficit -= take

    if deficit != 0:
        raise RuntimeError(f"Failed to allocate the requested number of points. Remaining deficit: {deficit}")
    return targets


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(points.shape[0], dtype=vertex_dtype)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_path = dataset_root / args.output_name
    transforms_path = dataset_root / "transforms.json"

    images, depths, confidences, cameras, frames = validate_inputs(dataset_root)
    num_frames = len(frames)
    print(f"Validated {num_frames} aligned frames in {dataset_root}")

    valid_counts = np.empty(num_frames, dtype=np.int64)
    for idx, (depth_path, conf_path) in enumerate(zip(depths, confidences)):
        depth = np.array(Image.open(depth_path), dtype=np.uint16)
        confidence = np.array(Image.open(conf_path), dtype=np.uint8)
        if depth.shape != confidence.shape:
            raise ValueError(f"Depth/confidence shape mismatch for {depth_path.name}: {depth.shape} vs {confidence.shape}")
        valid_counts[idx] = int(((depth > 0) & (confidence > args.confidence_threshold)).sum())
        if (idx + 1) % 500 == 0 or idx + 1 == num_frames:
            print(f"Counted valid pixels for {idx + 1}/{num_frames} frames")

    targets = compute_targets(valid_counts, args.point_count)
    print(
        "Sampling plan: "
        f"zero_valid_frames={int((valid_counts == 0).sum())}, "
        f"frames_with_points={int((targets > 0).sum())}, "
        f"min_valid={int(valid_counts.min())}, max_valid={int(valid_counts.max())}, "
        f"min_target={int(targets.min())}, max_target={int(targets.max())}"
    )

    points = np.empty((args.point_count, 3), dtype=np.float32)
    colors = np.empty((args.point_count, 3), dtype=np.uint8)
    rng = np.random.default_rng(args.seed)
    cursor = 0

    for idx, (frame, image_path, depth_path, conf_path, camera_path, target_count) in enumerate(
        zip(frames, images, depths, confidences, cameras, targets)
    ):
        if target_count == 0:
            continue

        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        depth = np.array(Image.open(depth_path), dtype=np.uint16)
        confidence = np.array(Image.open(conf_path), dtype=np.uint8)
        camera = load_json(camera_path)

        valid_mask = (depth > 0) & (confidence > args.confidence_threshold)
        valid_indices = np.flatnonzero(valid_mask.reshape(-1))
        if valid_indices.size < target_count:
            raise ValueError(
                f"Frame {image_path.name} only has {valid_indices.size} valid pixels, "
                f"but {target_count} were requested."
            )

        sampled = rng.choice(valid_indices, size=int(target_count), replace=False)
        ys, xs = np.divmod(sampled, depth.shape[1])

        rgb_h, rgb_w = image.shape[:2]
        depth_h, depth_w = depth.shape
        scale_x = rgb_w / depth_w
        scale_y = rgb_h / depth_h

        u_full = (xs.astype(np.float32) + 0.5) * scale_x - 0.5
        v_full = (ys.astype(np.float32) + 0.5) * scale_y - 0.5

        z = depth[ys, xs].astype(np.float32) * args.depth_scale
        x = (u_full - float(camera["cx"])) / float(camera["fx"]) * z
        y = -((v_full - float(camera["cy"])) / float(camera["fy"]) * z)
        cam_points = np.stack((x, y, -z), axis=1)

        transform = np.asarray(frame["transform_matrix"], dtype=np.float32)
        world_points = cam_points @ transform[:3, :3].T + transform[:3, 3]

        color_x = np.clip(np.rint(u_full).astype(np.int32), 0, rgb_w - 1)
        color_y = np.clip(np.rint(v_full).astype(np.int32), 0, rgb_h - 1)
        sampled_colors = image[color_y, color_x]

        next_cursor = cursor + int(target_count)
        points[cursor:next_cursor] = world_points
        colors[cursor:next_cursor] = sampled_colors
        cursor = next_cursor

        if (idx + 1) % 250 == 0 or idx + 1 == num_frames:
            print(f"Sampled and transformed {idx + 1}/{num_frames} frames")

    if cursor != args.point_count:
        raise RuntimeError(f"Expected to write {args.point_count} points, but assembled {cursor}")

    write_binary_ply(output_path, points, colors)
    transforms = load_json(transforms_path)
    transforms["ply_file_path"] = args.output_name
    with transforms_path.open("w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=4)
        f.write("\n")

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    print(f"Wrote {args.point_count} points to {output_path}")
    print(f"Updated {transforms_path} with ply_file_path={args.output_name}")
    print(
        "Bounds: "
        f"x=[{mins[0]:.3f}, {maxs[0]:.3f}], "
        f"y=[{mins[1]:.3f}, {maxs[1]:.3f}], "
        f"z=[{mins[2]:.3f}, {maxs[2]:.3f}]"
    )


if __name__ == "__main__":
    main()
