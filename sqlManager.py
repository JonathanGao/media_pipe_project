import sqlite3
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from pathlib import Path

cwd = Path.cwd()
databasePath = cwd / "databases"

sampleLandmarksDatabase = databasePath / "sampleLandmarks.db"


def _connect():
    databasePath.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(sampleLandmarksDatabase)


def ensure_tables(conn=None):
    """Create runIds, landmarks, and ground_truth tables if they do not exist."""
    owns_connection = conn is None
    if owns_connection:
        conn = _connect()
    cursor = conn.cursor()

    # Create table to track run ids if it doesn't exist.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS runIds (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT NOT NULL,
        model_version TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS landmarks (
        run_id INTEGER NOT NULL,
        video_id TEXT NOT NULL,
        person_id INTEGER NOT NULL,
        frame_number INTEGER NOT NULL,
        timestamp REAL NOT NULL,
        idx INTEGER NOT NULL,
        x REAL NOT NULL,
        y REAL NOT NULL,
        z REAL NOT NULL,
        visibility REAL NOT NULL,
        presence REAL NOT NULL,
        PRIMARY KEY (run_id, person_id, frame_number, idx)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ground_truth (
        video_id TEXT NOT NULL,
        frame_number INTEGER NOT NULL,
        timestamp_ms REAL NOT NULL,
        idx INTEGER NOT NULL,
        x REAL NOT NULL,
        y REAL NOT NULL,
        PRIMARY KEY (video_id, frame_number, idx)
    )
    """)

    conn.commit()
    if owns_connection:
        conn.close()


def recordAndReturnRunId(videoId: str, modelVersion: str):
    # Open connection to the sample landmarks database.
    conn = _connect()
    ensure_tables(conn)
    cursor = conn.cursor()

    # Get the current run id to insert into the landmarks table.
    # modelVersion should be the full run settings tag when available.
    # modelVersion is the run settings tag from _build_run_tag (model, weights file, and confidence settings).
    cursor.execute("""
    INSERT INTO runIds (video_id, model_version) VALUES (?, ?)
    RETURNING run_id
    """, (videoId, modelVersion))
    runId = cursor.fetchone()[0]

    conn.commit()
    conn.close()

    print(f"Run ID: {runId} inserted for video: {videoId} and model: {modelVersion}")

    return runId


def recordSampleFrameLandmarks(idx: int, runId: int, landmark: NormalizedLandmark, videoId: str, personId: int, frameNumber: int, timestamp: float):
    # Open connection to the sample landmarks database.
    conn = _connect()
    ensure_tables(conn)
    cursor = conn.cursor()

    # Record the landmark.
    cursor.execute("""
    INSERT INTO landmarks (run_id, video_id, person_id, frame_number, timestamp, idx, x, y, z, visibility, presence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (runId, videoId, personId, frameNumber, timestamp, idx, landmark.x, landmark.y, landmark.z, landmark.visibility, landmark.presence))
    conn.commit()
    conn.close()


def upsertGroundTruthJoint(videoId: str, frameNumber: int, timestampMs: float, idx: int, x: float, y: float):
    """Insert or replace one ground-truth joint (normalized x/y in [0, 1])."""
    conn = _connect()
    ensure_tables(conn)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO ground_truth (video_id, frame_number, timestamp_ms, idx, x, y)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(video_id, frame_number, idx) DO UPDATE SET
        timestamp_ms = excluded.timestamp_ms,
        x = excluded.x,
        y = excluded.y
    """, (videoId, frameNumber, timestampMs, idx, x, y))
    conn.commit()
    conn.close()


def deleteGroundTruthFrame(videoId: str, frameNumber: int):
    """Remove all ground-truth joints for one frame (used when re-labeling)."""
    conn = _connect()
    ensure_tables(conn)
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM ground_truth WHERE video_id = ? AND frame_number = ?",
        (videoId, frameNumber),
    )
    conn.commit()
    conn.close()


def loadGroundTruth(videoId: str):
    """Return list of (frame_number, timestamp_ms, idx, x, y) for a video."""
    conn = _connect()
    ensure_tables(conn)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT frame_number, timestamp_ms, idx, x, y
    FROM ground_truth
    WHERE video_id = ?
    ORDER BY frame_number, idx
    """, (videoId,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def loadRunsForVideo(videoId: str):
    """Return list of (run_id, model_version) for a video."""
    conn = _connect()
    ensure_tables(conn)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT run_id, model_version
    FROM runIds
    WHERE video_id = ?
    ORDER BY run_id
    """, (videoId,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def loadLandmarksForRun(runId: int, videoId: str):
    """Return list of (frame_number, person_id, idx, x, y) for a run."""
    conn = _connect()
    ensure_tables(conn)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT frame_number, person_id, idx, x, y
    FROM landmarks
    WHERE run_id = ? AND video_id = ?
    ORDER BY frame_number, person_id, idx
    """, (runId, videoId))
    rows = cursor.fetchall()
    conn.close()
    return rows
