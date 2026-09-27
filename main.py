import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles, PoseLandmark, PoseLandmarksConnections

from pathlib import Path
import cv2

from sqlManager import recordSampleFrameLandmarks, recordAndReturnRunId

# Define paths that will remain constant
cwd = Path.cwd()
holisticLandmarkerModelPath = cwd / "models" / "pose_landmarker_full.task"
PoseLandmarkerModelPath = cwd / "models" / "pose_landmarker_full.task"
sampleVideoPath = cwd / "sampleVids"
filmVideoPath = cwd / "vbPracticeFilm"
outputVideoPath = cwd / "outputVids"

# create shortcuts for base options and vision running mode
baseOptions = mp.tasks.BaseOptions
visionRunningMode = mp.tasks.vision.RunningMode

# Define the different models
holisticLandmarker = mp.tasks.vision.HolisticLandmarker
holisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions

poseLandmarker = mp.tasks.vision.PoseLandmarker
poseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

# Define options for the holistic landmarker.
options1 = holisticLandmarkerOptions(
    base_options=baseOptions(model_asset_path=str(holisticLandmarkerModelPath)),
    running_mode=visionRunningMode.VIDEO,
    min_face_detection_confidence=0.7,
    min_pose_detection_confidence=0.7,
    min_hand_landmarks_confidence=0.7,
    min_pose_landmarks_confidence=0.7,
)

# Define options for the pose landmarker.
options2 = poseLandmarkerOptions(
    base_options=baseOptions(model_asset_path=str(PoseLandmarkerModelPath)),
    running_mode=visionRunningMode.VIDEO,
    num_poses=4,
    min_pose_detection_confidence=0.6,
    min_pose_presence_confidence=0.8,
    min_tracking_confidence=0.6,
)

# Define the different sample videos.
sample1VideoName = "TheSpikeTORQVB.mp4"
sample2VideoName = "ElevateYourself.mp4"
sample3VideoName = "EiroMotiko.mp4"
sample4VideoName = "Azamgarh.mp4"
sample5VideoName = "USAvsCuba.mp4"
sample6VideoName = "HittingLines.mp4"

output1VideoPath = outputVideoPath / sample1VideoName.replace(".mp4", "_output.mp4")
output2VideoPath = outputVideoPath / sample2VideoName.replace(".mp4", "_output.mp4")
output3VideoPath = outputVideoPath / sample3VideoName.replace(".mp4", "_output.mp4")
output4VideoPath = outputVideoPath / sample4VideoName.replace(".mp4", "_output.mp4")
output5VideoPath = outputVideoPath / sample5VideoName.replace(".mp4", "_output.mp4")
output6VideoPath = outputVideoPath / sample6VideoName.replace(".mp4", "_output.mp4")

film3670VideoName = "IMG_3670.MOV";
film3671VideoName = "IMG_3671.MOV";
film3672VideoName = "IMG_3672.MOV";

output3670Path = outputVideoPath / film3670VideoName.replace(".MOV", "_output.mp4")
output3671Path = outputVideoPath / film3671VideoName.replace(".MOV", "_output.mp4")
output3672Path = outputVideoPath / film3672VideoName.replace(".MOV", "_output.mp4")

film3672TrackingVideoName = "3672Tracking.mp4"
film3672TrackingCroppedVideoName = "3672TrackingCropped.mp4"

film3672TrackingOutputPath = outputVideoPath / film3672TrackingVideoName.replace(".mp4", "_output.mp4")
film3672TrackingCroppedOutputPath = outputVideoPath / film3672TrackingCroppedVideoName.replace(".mp4", "_output.mp4")
# Main skeleton only: skip face cluster, fingers, heels, and toes.
MAIN_JOINTS = {
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
    PoseLandmark.LEFT_HEEL,
    PoseLandmark.RIGHT_HEEL,
    PoseLandmark.LEFT_FOOT_INDEX,
    PoseLandmark.RIGHT_FOOT_INDEX,
    PoseLandmark.NOSE,
    PoseLandmark.LEFT_INDEX,
    PoseLandmark.RIGHT_INDEX,
    PoseLandmark.LEFT_THUMB,
    PoseLandmark.RIGHT_THUMB,
    PoseLandmark.LEFT_PINKY,
    PoseLandmark.RIGHT_PINKY,
}
MAIN_CONNECTIONS = [
    connection
    for connection in PoseLandmarksConnections.POSE_LANDMARKS
    if connection.start in MAIN_JOINTS and connection.end in MAIN_JOINTS
]

def _normalize_analyze_timestamps(analyze_timestamps, fps):
    """Turn optional timestamp input into inclusive (start_ms, end_ms) ranges.

    Accepted forms (all values in milliseconds):
      None              -> analyze the whole video
      1500              -> nearest frame around 1500 ms
      (1000, 3000)      -> every frame in that range
      [1000, 2500]      -> nearest frames around each timestamp
      [(1000, 3000)]    -> every frame in that range
      [1000, (2000, 2500)] -> mix of single points and ranges
    """
    if analyze_timestamps is None:
        return None

    if isinstance(analyze_timestamps, (int, float)):
        analyze_timestamps = [analyze_timestamps]
    elif (
        isinstance(analyze_timestamps, tuple)
        and len(analyze_timestamps) == 2
        and all(isinstance(value, (int, float)) for value in analyze_timestamps)
    ):
        # Bare (start, end) means one range, not two single timestamps.
        analyze_timestamps = [analyze_timestamps]

    # Half a frame so a single ms value still matches one decoded frame.
    tolerance_ms = max(1, int(1000 / (2 * max(fps, 1))))
    ranges = []
    for item in analyze_timestamps:
        if (
            isinstance(item, (tuple, list))
            and len(item) == 2
            and all(isinstance(value, (int, float)) for value in item)
        ):
            start_ms, end_ms = int(item[0]), int(item[1])
            if end_ms < start_ms:
                start_ms, end_ms = end_ms, start_ms
            ranges.append((start_ms, end_ms))
        else:
            ts = int(item)
            ranges.append((ts - tolerance_ms, ts + tolerance_ms))
    return ranges

# Checks if the frame timestamp should be analyzed based on the analyze_ranges
def _timestamp_should_be_analyzed(frame_timestamp_ms, analyze_ranges):
    if analyze_ranges is None:
        return True
    return any(start_ms <= frame_timestamp_ms <= end_ms for start_ms, end_ms in analyze_ranges)


def _format_confidence(value):
    return f"{float(value):.2f}".replace(".", "p")


def _build_run_tag(landmarkerModelChosen, landmarkerOptions, analyze_timestamps=None):
    """Build a filesystem-safe tag describing model + key settings for this run."""
    model_name = landmarkerModelChosen.__name__
    model_file = Path(landmarkerOptions.base_options.model_asset_path).stem

    parts = [model_name, model_file]

    if hasattr(landmarkerOptions, "num_poses"):
        parts.append(f"np{landmarkerOptions.num_poses}")
    if hasattr(landmarkerOptions, "min_pose_detection_confidence"):
        parts.append(f"det{_format_confidence(landmarkerOptions.min_pose_detection_confidence)}")
    if hasattr(landmarkerOptions, "min_pose_presence_confidence"):
        parts.append(f"pres{_format_confidence(landmarkerOptions.min_pose_presence_confidence)}")
    if hasattr(landmarkerOptions, "min_tracking_confidence"):
        parts.append(f"trk{_format_confidence(landmarkerOptions.min_tracking_confidence)}")
    if hasattr(landmarkerOptions, "min_pose_landmarks_confidence"):
        parts.append(f"lm{_format_confidence(landmarkerOptions.min_pose_landmarks_confidence)}")
    if hasattr(landmarkerOptions, "min_hand_landmarks_confidence"):
        parts.append(f"hand{_format_confidence(landmarkerOptions.min_hand_landmarks_confidence)}")
    if hasattr(landmarkerOptions, "min_face_detection_confidence"):
        parts.append(f"face{_format_confidence(landmarkerOptions.min_face_detection_confidence)}")

    if analyze_timestamps is not None:
        ranges = _normalize_analyze_timestamps(analyze_timestamps, fps=30)
        if ranges:
            range_bits = [f"{start}-{end}" for start, end in ranges]
            parts.append("ts" + "_".join(range_bits))

    return "_".join(parts)


def _with_run_settings_in_path(outputPath, landmarkerModelChosen, landmarkerOptions, analyze_timestamps=None):
    """Insert model/settings tag into the output filename stem."""
    outputPath = Path(outputPath)
    run_tag = _build_run_tag(landmarkerModelChosen, landmarkerOptions, analyze_timestamps)
    return outputPath.with_name(f"{outputPath.stem}_{run_tag}{outputPath.suffix}")


# Processes the video and records the landmarks
def processVideo(
    videoPath,
    outputPath,
    landmarkerModelChosen,
    landmarkerOptions,
    sampleVideoDirectory,
    analyze_timestamps=None,
):
    # Open video file for reading
    cap = cv2.VideoCapture(str(sampleVideoDirectory / videoPath))

    # Get properties of video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    analyze_ranges = _normalize_analyze_timestamps(analyze_timestamps, fps)

    # Include model + settings in the saved filename for this run.
    outputPath = _with_run_settings_in_path(
        outputPath, landmarkerModelChosen, landmarkerOptions, analyze_timestamps
    )

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(outputPath), fourcc, fps, (width, height))

    frameNumber = 0

    with landmarkerModelChosen.create_from_options(landmarkerOptions) as landmarker:
        runTag = _build_run_tag(landmarkerModelChosen, landmarkerOptions, analyze_timestamps)
        runId = recordAndReturnRunId(videoPath, runTag)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frameNumber += 1

            # Get frame timestamp in milliseconds (required for VIDEO mode)
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            if _timestamp_should_be_analyzed(frame_timestamp_ms, analyze_ranges):
                # Convert OpenCV BGR image to MediaPipe Image object
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Perform detection for video frame
                result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                # Draw pose landmarks + skeleton connections (like MediaPipe demos).
                # draw_landmarks hides points with visibility/presence < 0.5, so clear
                # those scores first to force drawing every joint (including "out of view").
                if result.pose_landmarks:

                    for personId, person in enumerate(result.pose_landmarks):
                        for idx, landmark in enumerate(person):
                            if idx in MAIN_JOINTS:
                                landmark.visibility = 1.0
                                landmark.presence = 1.0
                                recordSampleFrameLandmarks(idx, runId, landmark, videoPath, personId, frameNumber, frame_timestamp_ms)
                            else:
                                landmark.visibility = 0.0
                                landmark.presence = 0.0

                        drawing_utils.draw_landmarks(
                            frame,
                            person,
                            MAIN_CONNECTIONS,
                            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
                        )

                out.write(frame)

    # Release the video capture and writer objects.
    cap.release()
    out.release()

    # Print video properties
    print("---------------Input Video Properties---------------")
    print("Width:", width)
    print("Height:", height)
    print("FPS:", fps)
    print("Writer opened:", out.isOpened())
    print("----------------------------------------------------")
    print("\n")

    # Print output video properties
    print("---------------Output Video Properties---------------")
    print("Output exists:", outputPath.exists())
    print("Output size:", outputPath.stat().st_size, "bytes")
    print(f"Output video saved to {outputPath}")
    print("----------------------------------------------------")

processVideo(film3672VideoName, output3672Path, poseLandmarker, options2, filmVideoPath)