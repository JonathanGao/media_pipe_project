"""Interactive OpenCV tool to click ground-truth joints on sampled frames.

Default target: vbPracticeFilm/3672TrackingCropped.mp4

Controls (per joint):
  left click  - place the current joint
  s           - skip this joint (leave unlabeled)
  u           - undo last placed joint on this frame
  n           - finish this frame and go to the next sampled frame
  r           - reset all joints on the current frame
  q           - save progress and quit

Joints are stored as normalized coordinates (0-1) in the ground_truth table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from mediapipe.tasks.python.vision import PoseLandmark

from sqlManager import deleteGroundTruthFrame, ensure_tables, upsertGroundTruthJoint

cwd = Path.cwd()

# Core body joints for labeling (aligned with PoseLandmark indices).
LABEL_JOINTS = [
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_HIP,
    PoseLandmark.RIGHT_HIP,
    PoseLandmark.LEFT_KNEE,
    PoseLandmark.RIGHT_KNEE,
    PoseLandmark.LEFT_ANKLE,
    PoseLandmark.RIGHT_ANKLE,
]


class FrameLabelState:
    def __init__(self, frame_number: int, timestamp_ms: float, frame_bgr):
        self.frame_number = frame_number
        self.timestamp_ms = timestamp_ms
        self.frame_bgr = frame_bgr
        self.labels: dict[int, tuple[float, float]] = {}  # idx -> normalized (x, y)
        self.joint_index = 0


def _sample_frame_numbers(total_frames: int, num_samples: int) -> list[int]:
    if total_frames <= 0:
        return []
    num_samples = max(1, min(num_samples, total_frames))
    if num_samples == 1:
        return [1]
    # Frame numbers in this project are 1-indexed (processVideo increments before use).
    return [
        1 + round(i * (total_frames - 1) / (num_samples - 1))
        for i in range(num_samples)
    ]


def _draw_overlay(state: FrameLabelState, width: int, height: int):
    canvas = state.frame_bgr.copy()
    for idx, (nx, ny) in state.labels.items():
        px, py = int(nx * width), int(ny * height)
        cv2.circle(canvas, (px, py), 6, (0, 255, 0), -1)
        try:
            name = PoseLandmark(idx).name
        except ValueError:
            name = str(idx)
        cv2.putText(canvas, name, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    if state.joint_index < len(LABEL_JOINTS):
        current = LABEL_JOINTS[state.joint_index]
        prompt = f"Frame {state.frame_number} | Click: {current.name} ({state.joint_index + 1}/{len(LABEL_JOINTS)})"
    else:
        prompt = f"Frame {state.frame_number} | All joints set — press n for next, u undo, r reset, q quit"

    cv2.rectangle(canvas, (0, 0), (width, 36), (0, 0, 0), -1)
    cv2.putText(canvas, prompt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(
        canvas,
        "s skip | u undo | r reset | n next frame | q quit/save",
        (10, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
    )
    return canvas


def _save_frame_labels(video_id: str, state: FrameLabelState):
    deleteGroundTruthFrame(video_id, state.frame_number)
    for idx, (x, y) in state.labels.items():
        upsertGroundTruthJoint(video_id, state.frame_number, state.timestamp_ms, idx, x, y)
    print(
        f"Saved {len(state.labels)} joints for frame {state.frame_number} "
        f"({state.timestamp_ms:.0f} ms)"
    )


def label_video(
    video_path: Path,
    video_id: str,
    num_samples: int = 30,
    start_ms: float | None = None,
):
    ensure_tables()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        raise SystemExit(f"Invalid video dimensions: {width}x{height}")

    frame_numbers = _sample_frame_numbers(total_frames, num_samples)
    if start_ms is not None:
        start_frame = max(1, int(start_ms / 1000.0 * fps) + 1)
        frame_numbers = [fn for fn in frame_numbers if fn >= start_frame]
        if not frame_numbers:
            frame_numbers = [min(start_frame, total_frames)]

    print(f"Labeling {video_id}")
    print(f"Video: {video_path} ({width}x{height}, {total_frames} frames @ {fps:.2f} fps)")
    print(f"Sampled frames: {frame_numbers}")

    window_name = "Ground Truth Labeler"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    click_point = {"xy": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_point["xy"] = (x, y)

    cv2.setMouseCallback(window_name, on_mouse)

    for frame_number in frame_numbers:
        # OpenCV POS_FRAMES is 0-indexed; our frame_number is 1-indexed.
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ok, frame = cap.read()
        if not ok:
            print(f"Skipping unreadable frame {frame_number}")
            continue

        timestamp_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        state = FrameLabelState(frame_number, timestamp_ms, frame)
        click_point["xy"] = None

        while True:
            canvas = _draw_overlay(state, width, height)
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF

            if click_point["xy"] is not None and state.joint_index < len(LABEL_JOINTS):
                px, py = click_point["xy"]
                click_point["xy"] = None
                joint = LABEL_JOINTS[state.joint_index]
                nx, ny = px / width, py / height
                state.labels[int(joint)] = (nx, ny)
                state.joint_index += 1

            if key == ord("s"):
                if state.joint_index < len(LABEL_JOINTS):
                    state.joint_index += 1
            elif key == ord("u"):
                if state.labels:
                    # Remove the most recently added joint among LABEL_JOINTS order.
                    for joint in reversed(LABEL_JOINTS[: state.joint_index]):
                        idx = int(joint)
                        if idx in state.labels:
                            del state.labels[idx]
                            state.joint_index = LABEL_JOINTS.index(joint)
                            break
            elif key == ord("r"):
                state.labels.clear()
                state.joint_index = 0
            elif key == ord("n"):
                _save_frame_labels(video_id, state)
                break
            elif key == ord("q"):
                _save_frame_labels(video_id, state)
                cap.release()
                cv2.destroyAllWindows()
                print("Quit — progress saved for current frame.")
                return

    cap.release()
    cv2.destroyAllWindows()
    print("Finished all sampled frames.")


def main():
    parser = argparse.ArgumentParser(description="Click ground-truth pose joints on sampled frames.")
    parser.add_argument(
        "--video",
        type=Path,
        default=cwd / "vbPracticeFilm" / "3672TrackingCropped.mp4",
        help="Path to the source video to label",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="video_id stored in the DB (defaults to the video filename)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="Number of evenly spaced frames to label",
    )
    parser.add_argument(
        "--start-ms",
        type=float,
        default=None,
        help="Optional earliest timestamp (ms) to include in sampling",
    )
    args = parser.parse_args()

    video_path = args.video
    video_id = args.video_id or video_path.name
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    label_video(video_path, video_id, num_samples=args.num_samples, start_ms=args.start_ms)


if __name__ == "__main__":
    main()
