import mediapipe as mp
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles, PoseLandmarksConnections

from pathlib import Path
import cv2

cwd = Path.cwd()
modelPath = cwd / "pose_landmarker_full.task"
sampleVideoPath = cwd / "sampleVids"
outputVideoPath = cwd / "outputVids"

baseOptions = mp.tasks.BaseOptions
poseLandmarker = mp.tasks.vision.PoseLandmarker
poseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
visionRunningMode = mp.tasks.vision.RunningMode

# HolisticLandmarker is single-person only. PoseLandmarker + num_poses
# tracks multiple people (pose joints only — no face mesh / hands).
options = poseLandmarkerOptions(
    base_options=baseOptions(model_asset_path=str(modelPath)),
    running_mode=visionRunningMode.VIDEO,
    num_poses=4,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

sampleVideoName = "TheSpikeTORQVB.mp4"
sample2VideoName = "ElevateYourself.mp4"

TorqvbSample = sampleVideoPath / sampleVideoName
ElevateYourselfSample = sampleVideoPath / sample2VideoName
output1VideoPath = outputVideoPath / sampleVideoName.replace(".mp4", "_output.mp4")
output2VideoPath = outputVideoPath / sample2VideoName.replace(".mp4", "_output.mp4")

# Open video file for reading
cap = cv2.VideoCapture(str(ElevateYourselfSample))

# Get properties of video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output2VideoPath), fourcc, fps, (width, height))

print("Width:", width)
print("Height:", height)
print("FPS:", fps)
print("Writer opened:", out.isOpened())

with poseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get frame timestamp in milliseconds (required for VIDEO mode)
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Convert OpenCV BGR image to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform detection for video frame
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # pose_landmarks is a list of people; each person is a list of landmarks
        for person_landmarks in result.pose_landmarks:
            for landmark in person_landmarks:
                landmark.visibility = 1.0
                landmark.presence = 1.0

            drawing_utils.draw_landmarks(
                frame,
                person_landmarks,
                PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
            )

        out.write(frame)

cap.release()
out.release()

print("Output exists:", output2VideoPath.exists())
print("Output size:", output2VideoPath.stat().st_size, "bytes")

print(f"Output video saved to {output2VideoPath}")
