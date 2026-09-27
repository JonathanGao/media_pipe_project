# media_pipe_project

Pose landmark processing with MediaPipe. Processed videos are saved under `outputVids/`. Landmark data and ground-truth labels are stored in `databases/sampleLandmarks.db`.

## Output filename key

Each run appends a settings tag to the base output name:

```text
{base_name}_{ModelClass}_{model_file}_{settings...}.mp4
```

### Example

```text
3672TrackingCropped_output_PoseLandmarker_pose_landmarker_full_np1_det0p60_pres0p80_trk0p60.mp4
```

| Segment | Meaning |
| --- | --- |
| `3672TrackingCropped_output` | Base output name you passed in |
| `PoseLandmarker` | MediaPipe task class (`PoseLandmarker` or `HolisticLandmarker`) |
| `pose_landmarker_full` | `.task` model file stem (weights used) |
| `np1` | `num_poses=1` (max people to detect; Pose Landmarker only) |
| `det0p60` | `min_pose_detection_confidence=0.60` |
| `pres0p80` | `min_pose_presence_confidence=0.80` (Pose Landmarker only) |
| `trk0p60` | `min_tracking_confidence=0.60` (Pose Landmarker only) |

### Confidence formatting

Decimals use `p` instead of `.` so filenames stay filesystem-safe:

- `0.60` → `0p60`
- `0.80` → `0p80`

### Optional / model-specific tags

These appear only when that option exists on the chosen model:

| Tag | Setting |
| --- | --- |
| `lm0p70` | `min_pose_landmarks_confidence` (Holistic) |
| `hand0p70` | `min_hand_landmarks_confidence` (Holistic) |
| `face0p70` | `min_face_detection_confidence` (Holistic) |
| `ts57000-90000` | Analyzed timestamp range in milliseconds |
| `ts1000-1000_5000-5000` | Multiple analyzed ranges / points |

If `analyze_timestamps` is omitted, there is no `ts...` segment and the whole video is processed.

The same settings tag is stored in `runIds.model_version` so evaluation rankings stay readable.

## Accuracy evaluation (3672)

To compare setups objectively, label joints once on a source clip, then score every stored run for that `video_id` against those labels.

Scoring uses the **database** (ground truth ↔ landmarks), not input vs output video files. Output videos may have different names (settings tags) or lengths (timestamp windows); that does not block evaluation.

### 1. Label ground truth

```bash
python3 label_frames.py \
  --video vbPracticeFilm/3672TrackingCropped.mp4 \
  --video-id 3672TrackingCropped.mp4 \
  --num-samples 30
```

Controls:

| Key / action | Meaning |
| --- | --- |
| Left click | Place the current joint |
| `s` | Skip this joint |
| `u` | Undo last joint on this frame |
| `r` | Reset the frame |
| `n` | Save frame and go to next sample |
| `q` | Save current frame and quit |

Default joints: shoulders, elbows, wrists, hips, knees, ankles (12). Coordinates are stored normalized (0–1).

**Important:** use the same source file / `video_id` that `main.py` used when recording landmarks.

### 2. Process setups you want to compare

Run `main.py` with each model/settings combo on that same video so landmarks are saved under the same `video_id`.

### 3. Evaluate

```bash
python3 evaluate_runs.py \
  --video-id 3672TrackingCropped.mp4 \
  --source-video vbPracticeFilm/3672TrackingCropped.mp4 \
  --output-csv evaluation/run_scores_3672.csv
```

Preflight checks (script exits or warns if these fail):

- Ground-truth rows exist for `video_id`
- Runs exist for that `video_id`
- Source video opens with valid width/height
- A run with no overlapping labeled joints is warned and ranks poorly / empty

### Metrics

| Metric | Meaning | Better |
| --- | --- | --- |
| **MPJPE** | Mean pixel distance between your click and the model joint (matched joints only) | Lower |
| **PCK@0.1** | Fraction of labeled joints within 0.1 × torso length of GT | Higher |
| **PCK@0.2** | Same with 0.2 × torso length | Higher |
| **detection_rate** | Fraction of labeled frames where the run has any pose | Higher |

Torso length = shoulder midpoint → hip midpoint from ground truth on that frame. Missing predictions count as PCK misses and are excluded from MPJPE.

Ranking order: higher `pck_0_2`, then lower `mpjpe_px`, then higher `detection_rate`.
