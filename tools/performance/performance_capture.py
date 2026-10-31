"""Markerless performance capture from ordinary video using MediaPipe Tasks.

Records what an actor does on camera — 3D body pose, both hands, and the 52
ARKit-style face blendshapes plus head rotation — then derives readable
acting descriptors (expression, gaze, body language, motion energy) and
splits the take into beats. The JSON it writes is the raw material for
`performance_memory` and for retargeting onto 3D characters.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    ToolResult,
    ToolStability,
    ToolStatus,
    ToolTier,
)
from tools.performance import features

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = REPO_ROOT / ".models" / "mediapipe"
SCHEMA_ID = "openmontage.performance_capture/v1"

_MODEL_BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "body": ("pose_landmarker_heavy.task", f"{_MODEL_BASE}/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"),
    "hands": ("hand_landmarker.task", f"{_MODEL_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"),
    "face": ("face_landmarker.task", f"{_MODEL_BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def models_dir(inputs: dict[str, Any] | None = None) -> Path:
    override = (inputs or {}).get("models_dir") or os.environ.get("OPENMONTAGE_MEDIAPIPE_MODELS")
    return Path(override) if override else DEFAULT_MODELS_DIR


def _round_points(points: Any, with_visibility: bool) -> list[list[float]]:
    rows = []
    for p in points:
        row = [round(p.x, 4), round(p.y, 4), round(p.z, 4)]
        if with_visibility:
            row.append(round(getattr(p, "visibility", 1.0) or 0.0, 3))
        rows.append(row)
    return rows


class PerformanceCapture(BaseTool):
    name = "performance_capture"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "analysis"
    provider = "mediapipe"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = ["python:mediapipe", "python:cv2"]
    install_instructions = (
        "pip install mediapipe opencv-python-headless  (or: make install-performance)\n"
        "Then fetch the models once: run this tool with operation=download_models."
    )
    agent_skills = ["performance-capture"]

    capabilities = [
        "performance_capture",
        "body_pose_3d",
        "hand_tracking",
        "face_blendshapes",
        "head_pose_estimation",
        "expression_analysis",
        "body_language_analysis",
    ]
    best_for = [
        "Recording an actor's body, hand and facial performance from a phone video",
        "Building the motion and emotion library that drives animated characters",
    ]

    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["capture", "download_models"],
                "default": "capture",
            },
            "input_path": {"type": "string", "description": "Video or still image of the performer"},
            "output_path": {"type": "string", "description": "Capture JSON path (default: <input>.performance.json)"},
            "capture": {
                "type": "array",
                "items": {"type": "string", "enum": ["body", "hands", "face"]},
                "default": ["body", "hands", "face"],
            },
            "sample_fps": {
                "type": "number",
                "default": 15,
                "description": "Frames per second to analyse; 0 = every frame",
            },
            "max_seconds": {"type": "number", "description": "Stop after this much of the source"},
            "mirrored": {
                "type": "boolean",
                "default": False,
                "description": "True for selfie-mirrored footage. MediaPipe labels hands as if mirrored, so unmirrored footage gets its labels swapped.",
            },
            "include_face_mesh": {
                "type": "boolean",
                "default": False,
                "description": "Also store the 478-point face mesh per frame (large)",
            },
            "segment_min_seconds": {"type": "number", "default": 1.0},
            "segment_max_seconds": {"type": "number", "default": 6.0},
            "models_dir": {"type": "string"},
        },
    }

    output_schema = {
        "type": "object",
        "properties": {
            "output": {"type": "string"},
            "frames_analysed": {"type": "integer"},
            "segments": {"type": "integer"},
            "coverage": {"type": "object"},
            "summary": {"type": "object"},
        },
    }

    resource_profile = ResourceProfile(cpu_cores=4, ram_mb=2048, vram_mb=0, disk_mb=200)
    idempotency_key_fields = ["input_path", "capture", "sample_fps", "max_seconds", "mirrored", "include_face_mesh"]
    side_effects = ["writes performance capture JSON to output_path", "download_models writes .task files to models_dir"]
    user_visible_verification = [
        "Check segment descriptions against what the performer actually did",
        "Confirm body/face coverage is high; low coverage means the performer left frame or was occluded",
    ]

    def _missing_models(self, parts: list[str], inputs: dict[str, Any] | None = None) -> list[str]:
        root = models_dir(inputs)
        return [part for part in parts if not (root / MODELS[part][0]).exists()]

    def get_status(self) -> ToolStatus:
        status = super().get_status()
        if status != ToolStatus.AVAILABLE:
            return status
        return ToolStatus.DEGRADED if self._missing_models(list(MODELS)) else ToolStatus.AVAILABLE

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if inputs.get("operation", "capture") == "download_models":
            return self._download_models(inputs)
        return self._capture(inputs)

    def _download_models(self, inputs: dict[str, Any]) -> ToolResult:
        root = models_dir(inputs)
        root.mkdir(parents=True, exist_ok=True)
        fetched, present = [], []
        for filename, url in MODELS.values():
            target = root / filename
            if target.exists():
                present.append(str(target))
                continue
            tmp = target.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(target)
            fetched.append(str(target))
        return ToolResult(success=True, data={"models_dir": str(root), "downloaded": fetched, "already_present": present}, artifacts=fetched)

    def _make_landmarkers(self, parts: list[str], inputs: dict[str, Any]) -> dict[str, Any]:
        from mediapipe.tasks.python import BaseOptions, vision

        root = models_dir(inputs)
        mode = vision.RunningMode.VIDEO

        def opts(part: str) -> BaseOptions:
            return BaseOptions(model_asset_path=str(root / MODELS[part][0]))

        made: dict[str, Any] = {}
        if "body" in parts:
            made["body"] = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(base_options=opts("body"), running_mode=mode, num_poses=1)
            )
        if "hands" in parts:
            made["hands"] = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(base_options=opts("hands"), running_mode=mode, num_hands=2)
            )
        if "face" in parts:
            made["face"] = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=opts("face"),
                    running_mode=mode,
                    num_faces=1,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                )
            )
        return made

    def _analyse_frame(self, landmarkers: dict[str, Any], rgb: Any, ts_ms: int, inputs: dict[str, Any]) -> dict[str, Any]:
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame: dict[str, Any] = {}

        if "body" in landmarkers:
            res = landmarkers["body"].detect_for_video(image, ts_ms)
            if res.pose_world_landmarks:
                frame["pose_world"] = _round_points(res.pose_world_landmarks[0], with_visibility=True)
                frame["pose_image"] = _round_points(res.pose_landmarks[0], with_visibility=False)

        if "hands" in landmarkers:
            res = landmarkers["hands"].detect_for_video(image, ts_ms)
            hands = []
            for i, landmarks in enumerate(res.hand_landmarks):
                label = res.handedness[i][0].category_name.lower()
                if not inputs.get("mirrored", False):
                    label = {"left": "right", "right": "left"}.get(label, label)
                hands.append({
                    "side": label,
                    "confidence": round(res.handedness[i][0].score, 3),
                    "landmarks": _round_points(landmarks, with_visibility=False),
                    "world": _round_points(res.hand_world_landmarks[i], with_visibility=False),
                })
            if hands:
                frame["hands"] = hands

        if "face" in landmarkers:
            res = landmarkers["face"].detect_for_video(image, ts_ms)
            if res.face_blendshapes:
                blendshapes = {c.category_name: round(c.score, 4) for c in res.face_blendshapes[0] if c.category_name != "_neutral"}
                face: dict[str, Any] = {"blendshapes": blendshapes}
                if res.facial_transformation_matrixes:
                    matrix = [list(map(float, row)) for row in res.facial_transformation_matrixes[0]]
                    face["head_pose"] = features.head_pose_from_matrix(matrix)
                    frame["head_pose"] = face["head_pose"]
                if inputs.get("include_face_mesh"):
                    face["mesh"] = _round_points(res.face_landmarks[0], with_visibility=False)
                frame["face"] = face
                frame["expression"] = features.expression_from_blendshapes(blendshapes)
                frame["gaze"] = features.gaze_from_blendshapes(blendshapes)

        if "pose_world" in frame:
            frame["body_language"] = features.body_language_from_pose(frame["pose_world"], frame.get("head_pose"))
        return frame

    def _capture(self, inputs: dict[str, Any]) -> ToolResult:
        if not inputs.get("input_path"):
            return ToolResult(success=False, error="input_path is required for operation=capture")
        input_path = Path(inputs["input_path"])
        if not input_path.exists():
            return ToolResult(success=False, error=f"Input not found: {input_path}")

        parts = [p for p in inputs.get("capture", ["body", "hands", "face"]) if p in MODELS]
        if not parts:
            return ToolResult(success=False, error="capture must include at least one of body, hands, face")
        missing = self._missing_models(parts, inputs)
        if missing:
            return ToolResult(
                success=False,
                error=f"MediaPipe models missing for {missing} in {models_dir(inputs)}. Run operation=download_models first.",
            )

        import cv2

        output_path = Path(inputs.get("output_path") or input_path.with_suffix(".performance.json"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()

        is_image = input_path.suffix.lower() in IMAGE_SUFFIXES
        if is_image:
            still = cv2.imread(str(input_path))
            if still is None:
                return ToolResult(success=False, error=f"Could not read image: {input_path}")
            source_frames = iter([still])
            fps, total = 1.0, 1
            height, width = still.shape[:2]
        else:
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                return ToolResult(success=False, error=f"Could not open video: {input_path}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            def _read():
                while True:
                    ok, img = cap.read()
                    if not ok:
                        return
                    yield img

            source_frames = _read()

        sample_fps = inputs.get("sample_fps", 15)
        step = 1 if not sample_fps or sample_fps >= fps else max(1, round(fps / sample_fps))
        max_seconds = inputs.get("max_seconds")

        landmarkers = self._make_landmarkers(parts, inputs)
        frames: list[dict[str, Any]] = []
        try:
            for idx, bgr in enumerate(source_frames):
                t = idx / fps
                if max_seconds is not None and t > max_seconds:
                    break
                if idx % step:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frame = self._analyse_frame(landmarkers, rgb, int(round(t * 1000)), inputs)
                frame = {"t": round(t, 3), "frame_index": idx, **frame}
                frames.append(frame)
        finally:
            for lm in landmarkers.values():
                lm.close()
            if not is_image:
                cap.release()

        for frame, energy in zip(frames, features.motion_energy(frames)):
            frame["energy"] = energy
        segments = features.segment_frames(
            frames,
            min_seconds=inputs.get("segment_min_seconds", 1.0),
            max_seconds=inputs.get("segment_max_seconds", 6.0),
        )
        summary = features.summarize_window(frames) if frames else {}

        coverage = {
            part: round(sum(1 for f in frames if f.get(key)) / len(frames), 3) if frames else 0.0
            for part, key in (("body", "pose_world"), ("hands", "hands"), ("face", "face"))
            if part in parts
        }

        document = {
            "schema": SCHEMA_ID,
            "source": {
                "path": str(input_path),
                "kind": "image" if is_image else "video",
                "width": width,
                "height": height,
                "fps": round(fps, 3),
                "duration_seconds": round(total / fps, 3) if fps else 0.0,
                "analysed_fps": round(fps / step, 3),
                "mirrored": bool(inputs.get("mirrored", False)),
            },
            "captured": parts,
            "conventions": {
                "pose_world": "33 MediaPipe Pose landmarks, [x, y, z, visibility], metres, hip-centred, y down, -z toward camera",
                "pose_image": "33 landmarks, [x, y, z] normalised to image size",
                "hands": "21 landmarks per hand; side is the performer's own side",
                "face.blendshapes": "52 ARKit-compatible blendshape weights 0..1",
                "head_pose": "degrees; yaw +performer's left, pitch +down, roll +tilt",
            },
            "coverage": coverage,
            "summary": summary,
            "segments": segments,
            "frames": frames,
        }
        output_path.write_text(json.dumps(document), encoding="utf-8")

        return ToolResult(
            success=True,
            data={
                "output": str(output_path),
                "frames_analysed": len(frames),
                "segments": len(segments),
                "coverage": coverage,
                "summary": {k: summary.get(k) for k in ("description", "dominant_expression", "body_tags", "energy")},
            },
            artifacts=[str(output_path)],
            duration_seconds=round(time.time() - start, 2),
        )
