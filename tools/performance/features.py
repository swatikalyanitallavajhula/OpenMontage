"""Pure feature extraction for captured human performances.

Turns raw MediaPipe output (52 ARKit-style face blendshapes, 33 pose world
landmarks, facial transformation matrix) into readable acting descriptors:
facial expression, head pose, body language, motion energy, and segments.

No MediaPipe import here, so everything is unit-testable with synthetic data.

Expression scores are FACS-based readings of the *visible* expression, not a
claim about what the performer feels. That is exactly what animation needs:
the memory stores how an emotion is played, so a director can ask for it.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "contempt"]

# Pose landmark indices (MediaPipe Pose, 33-point topology).
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_ANKLE, R_ANKLE = 27, 28

MIN_VISIBILITY = 0.5
# Gestures like crossed arms hide the wrists by nature, so wrists get a lower bar.
MIN_WRIST_VISIBILITY = 0.3

# Valence/arousal anchors per expression (Russell circumplex, roughly).
_AFFECT = {
    "joy": (0.8, 0.5),
    "sadness": (-0.7, -0.4),
    "anger": (-0.6, 0.8),
    "fear": (-0.7, 0.7),
    "surprise": (0.1, 0.9),
    "disgust": (-0.7, 0.3),
    "contempt": (-0.5, 0.1),
}


def _bs(blendshapes: dict[str, float], *names: str) -> float:
    """Mean of the named blendshapes, missing names count as 0."""
    return sum(blendshapes.get(n, 0.0) for n in names) / len(names)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def expression_from_blendshapes(blendshapes: dict[str, float]) -> dict[str, Any]:
    """Score the seven basic expressions from ARKit blendshape weights.

    Each score combines the blendshapes for that expression's FACS action
    units (e.g. joy = AU6 cheek raise + AU12 lip corner pull). Scores are
    0..1, and `neutral` is high when nothing else is.
    """
    b = blendshapes
    smile = _bs(b, "mouthSmileLeft", "mouthSmileRight")
    cheek = _bs(b, "cheekSquintLeft", "cheekSquintRight")
    frown = _bs(b, "mouthFrownLeft", "mouthFrownRight")
    brow_down = _bs(b, "browDownLeft", "browDownRight")
    brow_inner = b.get("browInnerUp", 0.0)
    brow_outer = _bs(b, "browOuterUpLeft", "browOuterUpRight")
    eye_wide = _bs(b, "eyeWideLeft", "eyeWideRight")
    eye_squint = _bs(b, "eyeSquintLeft", "eyeSquintRight")
    jaw_open = b.get("jawOpen", 0.0)
    stretch = _bs(b, "mouthStretchLeft", "mouthStretchRight")
    press = _bs(b, "mouthPressLeft", "mouthPressRight")
    sneer = _bs(b, "noseSneerLeft", "noseSneerRight")
    upper_up = _bs(b, "mouthUpperUpLeft", "mouthUpperUpRight")
    smile_asym = abs(b.get("mouthSmileLeft", 0.0) - b.get("mouthSmileRight", 0.0))
    dimple_asym = abs(b.get("mouthDimpleLeft", 0.0) - b.get("mouthDimpleRight", 0.0))

    scores = {
        # AU6 + AU12
        "joy": _clamp(0.7 * smile + 0.3 * cheek),
        # AU1 + AU4 + AU15, minus smiling
        "sadness": _clamp(0.45 * frown + 0.35 * brow_inner + 0.2 * brow_down - 0.5 * smile),
        # AU4 + AU7 + AU23/24, minus brow raising. A Duchenne smile also lowers
        # the brows and squints the eyes, so smiling suppresses anger.
        "anger": _clamp(0.45 * brow_down + 0.25 * eye_squint + 0.3 * press - 0.3 * brow_outer - 0.9 * smile),
        # AU1 + AU2 + AU4 + AU5 + AU20
        "fear": _clamp(0.3 * eye_wide + 0.25 * brow_inner + 0.3 * stretch + 0.15 * brow_down),
        # AU1 + AU2 + AU5 + AU26, minus mouth stretch (which reads as fear)
        "surprise": _clamp(0.3 * brow_outer + 0.2 * brow_inner + 0.25 * eye_wide + 0.25 * jaw_open - 0.3 * stretch),
        # AU9 + AU10 + AU15; a broad smile raises the upper lip too
        "disgust": _clamp(0.5 * sneer + 0.3 * upper_up + 0.2 * frown - 0.6 * smile),
        # Unilateral AU12/AU14
        "contempt": _clamp(1.5 * smile_asym + 1.0 * dimple_asym - 0.5 * min(b.get("mouthSmileLeft", 0.0), b.get("mouthSmileRight", 0.0))),
    }
    top = max(scores, key=scores.get)
    peak = scores[top]
    scores["neutral"] = _clamp(1.0 - peak * 2.5)

    total = sum(scores[e] for e in EMOTIONS) or 1.0
    valence = sum(_AFFECT[e][0] * scores[e] for e in EMOTIONS) / total if peak > 0.05 else 0.0
    arousal = sum(_AFFECT[e][1] * scores[e] for e in EMOTIONS) / total if peak > 0.05 else -0.2

    dominant = top if peak >= 0.2 else "neutral"
    return {
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "dominant": dominant,
        "intensity": round(peak, 3),
        "valence": round(valence * min(1.0, peak * 2.5), 3),
        "arousal": round(arousal * min(1.0, max(peak, 0.2) * 2.5), 3),
    }


def gaze_from_blendshapes(blendshapes: dict[str, float]) -> dict[str, Any]:
    """Coarse eye direction and blink state from the eye blendshapes."""
    b = blendshapes
    down = _bs(b, "eyeLookDownLeft", "eyeLookDownRight")
    up = _bs(b, "eyeLookUpLeft", "eyeLookUpRight")
    # Performer's left eye looking out and right eye looking in = looking to their left.
    left = (b.get("eyeLookOutLeft", 0.0) + b.get("eyeLookInRight", 0.0)) / 2
    right = (b.get("eyeLookInLeft", 0.0) + b.get("eyeLookOutRight", 0.0)) / 2
    blink = _bs(b, "eyeBlinkLeft", "eyeBlinkRight")
    candidates = {"down": down, "up": up, "left": left, "right": right}
    direction = max(candidates, key=candidates.get)
    if candidates[direction] < 0.35:
        direction = "center"
    return {"direction": direction, "eyes_closed": blink > 0.6}


def head_pose_from_matrix(matrix: Sequence[Sequence[float]]) -> dict[str, float]:
    """Yaw/pitch/roll in degrees from a 4x4 facial transformation matrix."""
    r = matrix
    pitch = math.degrees(math.atan2(r[2][1], r[2][2]))
    yaw = math.degrees(math.atan2(-r[2][0], math.hypot(r[2][1], r[2][2])))
    roll = math.degrees(math.atan2(r[1][0], r[0][0]))
    return {"yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": round(roll, 1)}


def _visible(lm: Sequence[Sequence[float]], *indices: int, threshold: float = MIN_VISIBILITY) -> bool:
    return all(len(lm[i]) < 4 or lm[i][3] >= threshold for i in indices)


def _mid(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.dist(a[:3], b[:3])


def body_language_from_pose(
    world: Sequence[Sequence[float]],
    head_pose: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Read posture and gesture cues from 33 pose world landmarks.

    `world` is a list of [x, y, z, visibility] in meters, hip-centred, with
    y pointing down and negative z toward the camera (MediaPipe convention).
    `head_pose` (from the face model) gives a better head tilt than the ears.
    Measures needing a landmark the model could not see are left out.
    """
    if len(world) < 33 or not _visible(world, L_SHOULDER, R_SHOULDER):
        return {"measures": {}, "tags": []}

    ls, rs = world[L_SHOULDER], world[R_SHOULDER]
    sw = _dist(ls, rs) or 1e-6
    shoulder_mid = _mid(ls, rs)
    measures: dict[str, float] = {}
    tags: list[str] = []

    wrists_seen = _visible(world, L_WRIST, R_WRIST, threshold=MIN_WRIST_VISIBILITY)
    lw, rw = world[L_WRIST], world[R_WRIST]
    # Lifting the arms to shoulder height raises the shoulders and drops the
    # head relative to them, so those two cues are unreliable while it lasts.
    arms_elevated = wrists_seen and min(lw[1], rw[1]) < shoulder_mid[1] + 0.25 * sw

    if _visible(world, L_HIP, R_HIP):
        hip_mid = _mid(world[L_HIP], world[R_HIP])
        torso_len = _dist(shoulder_mid, hip_mid) or 1e-6
        # Positive = shoulders ahead of hips toward the camera.
        lean = (hip_mid[2] - shoulder_mid[2]) / torso_len
        measures["lean_forward"] = round(lean, 3)
        if lean > 0.25:
            tags.append("leaning_forward")
        elif lean < -0.2:
            tags.append("leaning_back")

        if _visible(world, L_ANKLE, R_ANKLE):
            ankle_mid = _mid(world[L_ANKLE], world[R_ANKLE])
            shift = (hip_mid[0] - ankle_mid[0]) / sw
            measures["weight_shift"] = round(shift, 3)
            if abs(shift) > 0.25:
                tags.append("weight_shifted")

        upright = (hip_mid[1] - shoulder_mid[1]) / torso_len
        measures["torso_upright"] = round(upright, 3)

    if _visible(world, NOSE):
        head_height = (shoulder_mid[1] - world[NOSE][1]) / sw
        measures["head_height"] = round(head_height, 3)
        if head_height < 0.25 and not arms_elevated:
            tags.append("head_down")
            # A bowed head over a torso folding forward reads as a slump.
            if measures.get("torso_upright", 1.0) < 0.85:
                tags.append("slumped")

    if head_pose is not None:
        measures["head_tilt_deg"] = head_pose["roll"]
    if _visible(world, L_EAR, R_EAR):
        le, re = world[L_EAR], world[R_EAR]
        ear_span = abs(le[0] - re[0]) / sw
        # In profile the ears overlap and their angle means nothing.
        if head_pose is None and ear_span > 0.4:
            measures["head_tilt_deg"] = round(math.degrees(math.atan2(le[1] - re[1], abs(le[0] - re[0]))), 1)
        ear_mid = _mid(le, re)
        neck = (shoulder_mid[1] - ear_mid[1]) / sw
        measures["neck_length"] = round(neck, 3)
        if neck < 0.3 and not arms_elevated:
            tags.append("shoulders_raised")
    if abs(measures.get("head_tilt_deg", 0.0)) > 12:
        tags.append("head_tilted")

    if wrists_seen:
        spread = abs(lw[0] - rw[0]) / sw
        measures["arm_openness"] = round(spread, 3)

        # Each wrist on the opposite shoulder's side of the midline, below the shoulders.
        crossed = (
            (lw[0] - shoulder_mid[0]) * (rs[0] - shoulder_mid[0]) > 0
            and (rw[0] - shoulder_mid[0]) * (ls[0] - shoulder_mid[0]) > 0
            and lw[1] > shoulder_mid[1]
            and rw[1] > shoulder_mid[1]
        )
        if crossed:
            tags.append("arms_crossed")
        if spread > 1.8:
            tags.append("open_posture")
        elif crossed or spread < 0.5:
            tags.append("closed_posture")

        if _visible(world, NOSE):
            near_face = min(_dist(lw, world[NOSE]), _dist(rw, world[NOSE])) / sw
            measures["hand_to_face"] = round(near_face, 3)
            if near_face < 0.6:
                tags.append("hand_to_face")

        if _visible(world, L_HIP, R_HIP, L_ELBOW, R_ELBOW):
            on_hips = (
                _dist(lw, world[L_HIP]) / sw < 0.45
                and _dist(rw, world[R_HIP]) / sw < 0.45
                and abs(world[L_ELBOW][0] - world[R_ELBOW][0]) / sw > 1.3
            )
            if on_hips:
                tags.append("hands_on_hips")

        # A wrist clearly above the shoulders (y is down).
        if min(lw[1], rw[1]) < shoulder_mid[1] - 0.3 * sw:
            tags.append("hands_raised")

    return {"measures": measures, "tags": tags}


def _smoothed_track(frames: Sequence[dict[str, Any]], k: int, radius: int = 2) -> list[tuple[float, float, float] | None]:
    """Centred moving average of one landmark, skipping low-visibility samples."""
    raw: list[Sequence[float] | None] = []
    for f in frames:
        pose = f.get("pose_world")
        raw.append(pose[k] if pose and _visible(pose, k) else None)
    track: list[tuple[float, float, float] | None] = []
    for i, p in enumerate(raw):
        if p is None:
            track.append(None)
            continue
        window = [q for q in raw[max(0, i - radius): i + radius + 1] if q is not None]
        track.append(tuple(sum(q[d] for q in window) / len(window) for d in range(3)))
    return track


def motion_energy(frames: Sequence[dict[str, Any]]) -> list[float]:
    """Per-frame mean speed of upper-body landmarks in metres/second.

    Positions are smoothed first so tracking jitter on a still performer
    does not read as movement.
    """
    keypoints = [NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST]
    tracks = [_smoothed_track(frames, k) for k in keypoints]
    energies = [0.0] * len(frames)
    for i in range(1, len(frames)):
        dt = frames[i]["t"] - frames[i - 1]["t"]
        if dt <= 0:
            continue
        speeds = [_dist(tr[i - 1], tr[i]) / dt for tr in tracks if tr[i - 1] and tr[i]]
        if speeds:
            energies[i] = round(sum(speeds) / len(speeds), 4)
    return energies


def energy_label(speed: float) -> str:
    if speed < 0.12:
        return "low"
    if speed < 0.45:
        return "medium"
    return "high"


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize_window(frames: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a run of analysed frames into one segment description."""
    faces = [f["expression"] for f in frames if f.get("expression")]
    scores = {e: round(_mean(x["scores"][e] for x in faces), 3) for e in EMOTIONS + ["neutral"]} if faces else {}
    dominant = "neutral"
    if scores:
        top = max(EMOTIONS, key=lambda e: scores[e])
        if scores[top] >= 0.15:
            dominant = top

    tag_counts: dict[str, int] = {}
    body_frames = [f for f in frames if f.get("body_language")]
    for f in body_frames:
        for tag in f["body_language"]["tags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    # A cue must hold for at least 40% of the window to describe it.
    body_tags = sorted(t for t, c in tag_counts.items() if body_frames and c / len(body_frames) >= 0.4)

    gaze_counts: dict[str, int] = {}
    for f in frames:
        if f.get("gaze"):
            d = f["gaze"]["direction"]
            gaze_counts[d] = gaze_counts.get(d, 0) + 1
    gaze = max(gaze_counts, key=gaze_counts.get) if gaze_counts else None

    speed = _mean(f.get("energy", 0.0) for f in frames)
    energy = energy_label(speed)
    head = [f["head_pose"] for f in frames if f.get("head_pose")]
    head_avg = {k: round(_mean(h[k] for h in head), 1) for k in ("yaw", "pitch", "roll")} if head else None

    tags = [dominant] + body_tags + [f"{energy}_energy"]
    if gaze and gaze != "center":
        tags.append(f"gaze_{gaze}")
    if head_avg and abs(head_avg["yaw"]) > 25:
        tags.append("looking_away")

    return {
        "start": frames[0]["t"],
        "end": frames[-1]["t"],
        "expression_scores": scores,
        "dominant_expression": dominant,
        "valence": round(_mean(x["valence"] for x in faces), 3) if faces else None,
        "arousal": round(_mean(x["arousal"] for x in faces), 3) if faces else None,
        "body_tags": body_tags,
        "gaze": gaze,
        "head_pose": head_avg,
        "energy": energy,
        "mean_speed": round(speed, 4),
        "face_coverage": round(len(faces) / len(frames), 3),
        "body_coverage": round(len(body_frames) / len(frames), 3),
        "tags": tags,
        "description": describe(dominant, body_tags, energy, gaze),
    }


def describe(dominant: str, body_tags: Sequence[str], energy: str, gaze: str | None) -> str:
    parts = [f"{dominant} expression" if dominant != "neutral" else "neutral face"]
    if body_tags:
        parts.append(", ".join(t.replace("_", " ") for t in body_tags))
    parts.append(f"{energy} movement energy")
    if gaze and gaze != "center":
        parts.append(f"eyes {gaze}")
    return "; ".join(parts)


def segment_frames(
    frames: Sequence[dict[str, Any]],
    min_seconds: float = 1.0,
    max_seconds: float = 6.0,
) -> list[dict[str, Any]]:
    """Split analysed frames into acting beats.

    A new beat starts when the dominant expression or the body-language tag
    set changes and the current beat has lasted at least `min_seconds`, or
    when a beat reaches `max_seconds`. Signatures are mode-smoothed over five
    frames so a one-frame tracking glitch does not cut a beat.
    """
    if not frames:
        return []

    def signature(f: dict[str, Any]) -> tuple:
        expr = (f.get("expression") or {}).get("dominant", "neutral")
        tags = tuple(sorted((f.get("body_language") or {}).get("tags", [])))
        return expr, tags

    raw = [signature(f) for f in frames]
    smoothed = []
    for i in range(len(raw)):
        window = raw[max(0, i - 2): i + 3]
        smoothed.append(max(set(window), key=window.count))

    segments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = [frames[0]]
    for i in range(1, len(frames)):
        f = frames[i]
        length = f["t"] - current[0]["t"]
        changed = smoothed[i] != smoothed[i - 1]
        if length >= max_seconds or (changed and length >= min_seconds):
            segments.append(summarize_window(current))
            current = [f]
        else:
            current.append(f)
    segments.append(summarize_window(current))

    for i, seg in enumerate(segments):
        seg["index"] = i
    return segments
