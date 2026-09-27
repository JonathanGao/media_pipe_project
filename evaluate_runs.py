"""Score MediaPipe landmark runs against manual ground-truth labels.

Compares DB ground_truth rows to landmarks rows for the same video_id.
Does not compare input vs output video files.

Example:
  python3 evaluate_runs.py --video-id 3672TrackingCropped.mp4 \\
      --source-video vbPracticeFilm/3672TrackingCropped.mp4
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd
from mediapipe.tasks.python.vision import PoseLandmark

from sqlManager import ensure_tables, loadGroundTruth, loadLandmarksForRun, loadRunsForVideo

cwd = Path.cwd()

LEFT_SHOULDER = int(PoseLandmark.LEFT_SHOULDER)
RIGHT_SHOULDER = int(PoseLandmark.RIGHT_SHOULDER)
LEFT_HIP = int(PoseLandmark.LEFT_HIP)
RIGHT_HIP = int(PoseLandmark.RIGHT_HIP)


def _open_source_dims(source_video: Path) -> tuple[int, int]:
    if not source_video.exists():
        raise SystemExit(f"Source video not found: {source_video}")
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open source video: {source_video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise SystemExit(f"Invalid source video dimensions: {width}x{height}")
    return width, height


def _pixel_distance(ax: float, ay: float, bx: float, by: float, width: int, height: int) -> float:
    dx = (ax - bx) * width
    dy = (ay - by) * height
    return math.hypot(dx, dy)


def _torso_length_px(gt_by_idx: dict[int, tuple[float, float]], width: int, height: int) -> float | None:
    """Shoulder midpoint → hip midpoint distance in pixels from GT, if available."""
    needed = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    if not all(idx in gt_by_idx for idx in needed):
        return None
    ls = gt_by_idx[LEFT_SHOULDER]
    rs = gt_by_idx[RIGHT_SHOULDER]
    lh = gt_by_idx[LEFT_HIP]
    rh = gt_by_idx[RIGHT_HIP]
    shoulder_mid = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
    hip_mid = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
    length = _pixel_distance(shoulder_mid[0], shoulder_mid[1], hip_mid[0], hip_mid[1], width, height)
    return length if length > 1e-6 else None


def _best_person_prediction(
    preds_by_person: dict[int, dict[int, tuple[float, float]]],
    gt_by_idx: dict[int, tuple[float, float]],
    width: int,
    height: int,
) -> dict[int, tuple[float, float]]:
    """If multiple people exist on a frame, pick the person closest to GT overall."""
    if not preds_by_person:
        return {}
    if len(preds_by_person) == 1:
        return next(iter(preds_by_person.values()))

    best_person = None
    best_score = float("inf")
    for person_id, joints in preds_by_person.items():
        errors = []
        for idx, (gx, gy) in gt_by_idx.items():
            if idx not in joints:
                continue
            px, py = joints[idx]
            errors.append(_pixel_distance(gx, gy, px, py, width, height))
        if not errors:
            continue
        mean_err = sum(errors) / len(errors)
        if mean_err < best_score:
            best_score = mean_err
            best_person = person_id
    if best_person is None:
        return next(iter(preds_by_person.values()))
    return preds_by_person[best_person]


def evaluate_video(video_id: str, source_video: Path, output_csv: Path) -> pd.DataFrame:
    ensure_tables()
    width, height = _open_source_dims(source_video)

    gt_rows = loadGroundTruth(video_id)
    if not gt_rows:
        raise SystemExit(
            f"No ground_truth rows for video_id={video_id!r}. "
            f"Run label_frames.py on {source_video} first."
        )

    runs = loadRunsForVideo(video_id)
    if not runs:
        raise SystemExit(
            f"No runs found for video_id={video_id!r}. "
            "Process the video with main.py so landmarks are stored under the same video_id."
        )

    # frame_number -> {idx: (x, y)}
    gt_by_frame: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    labeled_frames = set()
    for frame_number, _timestamp_ms, idx, x, y in gt_rows:
        gt_by_frame[frame_number][idx] = (x, y)
        labeled_frames.add(frame_number)

    print(f"Evaluating video_id={video_id}")
    print(f"Source: {source_video} ({width}x{height})")
    print(f"GT joints: {len(gt_rows)} across {len(labeled_frames)} frames")
    print(f"Runs: {len(runs)}")

    results = []
    for run_id, model_version in runs:
        landmark_rows = loadLandmarksForRun(run_id, video_id)
        # frame -> person -> idx -> (x, y)
        preds: dict[int, dict[int, dict[int, tuple[float, float]]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        frames_with_any_pose = set()
        for frame_number, person_id, idx, x, y in landmark_rows:
            preds[frame_number][person_id][idx] = (x, y)
            if frame_number in labeled_frames:
                frames_with_any_pose.add(frame_number)

        errors = []
        pck_hits_01 = 0
        pck_hits_02 = 0
        pck_total = 0
        matched_joints = 0

        for frame_number, gt_by_idx in gt_by_frame.items():
            torso = _torso_length_px(gt_by_idx, width, height)
            # Fallback threshold if torso joints weren't labeled on this frame.
            fallback_threshold_01 = 0.1 * math.hypot(width, height) * 0.25
            fallback_threshold_02 = 0.2 * math.hypot(width, height) * 0.25
            thr_01 = 0.1 * torso if torso is not None else fallback_threshold_01
            thr_02 = 0.2 * torso if torso is not None else fallback_threshold_02

            person_joints = _best_person_prediction(
                preds.get(frame_number, {}), gt_by_idx, width, height
            )

            for idx, (gx, gy) in gt_by_idx.items():
                pck_total += 1
                if idx not in person_joints:
                    # Miss for PCK; excluded from MPJPE
                    continue
                px, py = person_joints[idx]
                err = _pixel_distance(gx, gy, px, py, width, height)
                errors.append(err)
                matched_joints += 1
                if err <= thr_01:
                    pck_hits_01 += 1
                if err <= thr_02:
                    pck_hits_02 += 1

        mpjpe = float(sum(errors) / len(errors)) if errors else float("nan")
        pck_01 = (pck_hits_01 / pck_total) if pck_total else float("nan")
        pck_02 = (pck_hits_02 / pck_total) if pck_total else float("nan")
        detection_rate = (
            len(frames_with_any_pose) / len(labeled_frames) if labeled_frames else float("nan")
        )

        if matched_joints == 0:
            print(
                f"Warning: run_id={run_id} ({model_version}) has no overlapping "
                f"(frame_number, idx) with ground truth — skipping from ranking metrics as empty."
            )

        results.append(
            {
                "run_id": run_id,
                "model_version": model_version,
                "mpjpe_px": mpjpe,
                "pck_0_1": pck_01,
                "pck_0_2": pck_02,
                "detection_rate": detection_rate,
                "matched_joints": matched_joints,
                "labeled_joints": pck_total,
                "labeled_frames": len(labeled_frames),
                "detected_labeled_frames": len(frames_with_any_pose),
            }
        )

    df = pd.DataFrame(results)
    # Rank: higher PCK@0.2 first, then lower MPJPE, then higher detection rate.
    df = df.sort_values(
        by=["pck_0_2", "mpjpe_px", "detection_rate"],
        ascending=[False, True, False],
        na_position="last",
    ).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\n=== Run ranking ===")
    display_cols = [
        "rank",
        "run_id",
        "model_version",
        "mpjpe_px",
        "pck_0_1",
        "pck_0_2",
        "detection_rate",
        "matched_joints",
    ]
    with pd.option_context("display.max_colwidth", 80, "display.width", 160):
        print(df[display_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote {output_csv}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate landmark runs against ground-truth labels for one video."
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default="3672TrackingCropped.mp4",
        help="video_id used in landmarks / ground_truth tables",
    )
    parser.add_argument(
        "--source-video",
        type=Path,
        default=cwd / "vbPracticeFilm" / "3672TrackingCropped.mp4",
        help="Source video file (for width/height when converting normalized coords)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=cwd / "evaluation" / "run_scores_3672.csv",
        help="Where to write the ranking CSV",
    )
    args = parser.parse_args()
    evaluate_video(args.video_id, args.source_video, args.output_csv)


if __name__ == "__main__":
    main()
