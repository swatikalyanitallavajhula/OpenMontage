"""Expression, body-language and segmentation readings from synthetic captures."""

from __future__ import annotations

import copy

from tools.performance import features as F


def standing_pose() -> list[list[float]]:
    """A relaxed, upright, front-facing performer in MediaPipe world coordinates."""
    pose = [[0.0, 0.0, 0.0, 1.0] for _ in range(33)]
    pose[F.NOSE] = [0.0, -0.62, -0.05, 1.0]
    pose[F.L_EAR], pose[F.R_EAR] = [0.07, -0.6, 0.0, 1.0], [-0.07, -0.6, 0.0, 1.0]
    pose[F.L_SHOULDER], pose[F.R_SHOULDER] = [0.18, -0.45, 0.0, 1.0], [-0.18, -0.45, 0.0, 1.0]
    pose[F.L_ELBOW], pose[F.R_ELBOW] = [0.2, -0.2, 0.0, 1.0], [-0.2, -0.2, 0.0, 1.0]
    pose[F.L_WRIST], pose[F.R_WRIST] = [0.2, 0.0, 0.0, 1.0], [-0.2, 0.0, 0.0, 1.0]
    pose[F.L_HIP], pose[F.R_HIP] = [0.1, 0.0, 0.0, 1.0], [-0.1, 0.0, 0.0, 1.0]
    pose[F.L_ANKLE], pose[F.R_ANKLE] = [0.1, 0.85, 0.0, 1.0], [-0.1, 0.85, 0.0, 1.0]
    return pose


SMILE = {
    "mouthSmileLeft": 0.96, "mouthSmileRight": 0.93, "browDownLeft": 0.84, "browDownRight": 0.82,
    "mouthUpperUpLeft": 0.76, "mouthUpperUpRight": 0.76, "eyeSquintLeft": 0.74, "eyeSquintRight": 0.66,
    "cheekSquintLeft": 0.3, "cheekSquintRight": 0.3,
}
SAD = {"mouthFrownLeft": 0.7, "mouthFrownRight": 0.7, "browInnerUp": 0.8, "browDownLeft": 0.2, "browDownRight": 0.2}
SURPRISE = {
    "browOuterUpLeft": 0.8, "browOuterUpRight": 0.8, "browInnerUp": 0.8,
    "eyeWideLeft": 0.7, "eyeWideRight": 0.7, "jawOpen": 0.7,
}


def tags_of(pose, head_pose=None) -> list[str]:
    return F.body_language_from_pose(pose, head_pose)["tags"]


def test_relaxed_standing_has_no_cues() -> None:
    assert tags_of(standing_pose()) == []


def test_crossed_arms_detected_even_with_partly_hidden_wrist() -> None:
    pose = standing_pose()
    pose[F.L_WRIST] = [-0.08, -0.2, -0.1, 0.35]
    pose[F.R_WRIST] = [0.08, -0.2, -0.1, 0.8]
    tags = tags_of(pose)
    assert "arms_crossed" in tags and "closed_posture" in tags


def test_raised_arms_do_not_read_as_tension_or_bowed_head() -> None:
    pose = standing_pose()
    pose[F.L_WRIST], pose[F.R_WRIST] = [0.8, -0.47, 0.0, 1.0], [-0.8, -0.47, 0.0, 1.0]
    # Lifting the arms pulls the shoulders up toward the ears and head.
    pose[F.L_SHOULDER][1] = pose[F.R_SHOULDER][1] = -0.53
    tags = tags_of(pose)
    assert "open_posture" in tags
    assert "shoulders_raised" not in tags and "head_down" not in tags


def test_bowed_head_and_lean() -> None:
    pose = standing_pose()
    pose[F.NOSE] = [0.0, -0.5, -0.1, 1.0]
    assert "head_down" in tags_of(pose)

    leaning = standing_pose()
    for k in (F.L_SHOULDER, F.R_SHOULDER, F.NOSE, F.L_EAR, F.R_EAR):
        leaning[k][2] -= 0.2
    assert "leaning_forward" in tags_of(leaning)


def test_head_tilt_prefers_face_roll_and_ignores_profile_ears() -> None:
    profile = standing_pose()
    profile[F.L_EAR] = [0.01, -0.62, 0.05, 1.0]
    profile[F.R_EAR] = [-0.01, -0.58, -0.05, 1.0]
    assert "head_tilted" not in tags_of(profile)
    assert "head_tilted" in tags_of(standing_pose(), {"yaw": 0.0, "pitch": 0.0, "roll": 20.0})


def test_occluded_shoulders_give_no_reading() -> None:
    pose = standing_pose()
    pose[F.L_SHOULDER][3] = 0.1
    assert F.body_language_from_pose(pose) == {"measures": {}, "tags": []}


def test_expressions() -> None:
    smile = F.expression_from_blendshapes(SMILE)
    assert smile["dominant"] == "joy"
    # A real smile lowers the brows and lifts the upper lip; neither is anger or disgust.
    assert smile["scores"]["anger"] == 0.0 and smile["scores"]["disgust"] == 0.0
    assert smile["valence"] > 0.3

    assert F.expression_from_blendshapes(SAD)["dominant"] == "sadness"
    assert F.expression_from_blendshapes(SAD)["valence"] < 0
    assert F.expression_from_blendshapes(SURPRISE)["dominant"] == "surprise"
    assert F.expression_from_blendshapes({})["dominant"] == "neutral"


def test_contempt_is_one_sided() -> None:
    smirk = {"mouthSmileLeft": 0.5, "mouthDimpleLeft": 0.3}
    assert F.expression_from_blendshapes(smirk)["dominant"] == "contempt"


def test_gaze_and_head_pose() -> None:
    assert F.gaze_from_blendshapes({"eyeLookDownLeft": 0.7, "eyeLookDownRight": 0.7})["direction"] == "down"
    assert F.gaze_from_blendshapes({})["direction"] == "center"
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    assert F.head_pose_from_matrix(identity) == {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}


def _frames(seconds: float, fps: int, blendshapes: dict, start: float = 0.0, wobble: float = 0.0) -> list[dict]:
    frames = []
    for i in range(int(seconds * fps)):
        pose = standing_pose()
        offset = wobble if i % 2 else -wobble
        pose[F.L_WRIST][0] += offset
        pose[F.R_WRIST][0] -= offset
        frames.append({
            "t": round(start + i / fps, 3),
            "pose_world": pose,
            "body_language": F.body_language_from_pose(pose),
            "expression": F.expression_from_blendshapes(blendshapes),
        })
    return frames


def test_tracking_jitter_reads_as_still_and_real_motion_as_high() -> None:
    still = _frames(2, 15, {}, wobble=0.004)
    assert max(F.motion_energy(still)) < 0.12

    moving = copy.deepcopy(still)
    for i, f in enumerate(moving):
        f["pose_world"][F.L_WRIST][1] = -0.6 if (i // 4) % 2 else 0.0
        f["pose_world"][F.R_WRIST][1] = -0.6 if (i // 4) % 2 else 0.0
        f["pose_world"][F.L_ELBOW][1] = -0.4 if (i // 4) % 2 else -0.2
        f["pose_world"][F.R_ELBOW][1] = -0.4 if (i // 4) % 2 else -0.2
    energy = F.motion_energy(moving)
    assert F.energy_label(sum(energy) / len(energy)) == "high"


def test_segments_split_on_expression_change_not_on_glitch() -> None:
    frames = _frames(2, 10, SMILE) + _frames(2, 10, SAD, start=2.0)
    # A single misread frame inside the happy beat must not cut it.
    frames[7]["expression"] = F.expression_from_blendshapes(SURPRISE)
    for f, e in zip(frames, F.motion_energy(frames)):
        f["energy"] = e

    segments = F.segment_frames(frames)
    assert [s["dominant_expression"] for s in segments] == ["joy", "sadness"]
    assert segments[0]["start"] == 0.0 and segments[1]["start"] == 2.0
    assert "joy" in segments[0]["tags"] and segments[0]["description"].startswith("joy expression")


def test_long_beats_are_capped() -> None:
    segments = F.segment_frames(_frames(14, 5, SMILE), max_seconds=6.0)
    assert len(segments) == 3
    assert all(s["end"] - s["start"] <= 6.0 for s in segments)
