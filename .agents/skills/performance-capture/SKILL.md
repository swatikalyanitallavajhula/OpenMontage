---
name: performance-capture
description: Record actors' body, hand and facial performance from ordinary video with performance_capture, file it in the performance_memory library, and search it by emotion and body language.
license: MIT
---

# Performance Capture

Use this skill before calling `performance_capture` or `performance_memory`.

Realistic character animation comes from real human performance, not from
guessing. These two tools record what an actor does on camera and keep it
in a library a director can search in plain words.

## Shooting a Take

The capture is only as good as the footage. Tell the performer:

- **One continuous take per capture.** The tracker needs ~0.5 s to re-lock
  after a hard cut, and those frames read as wild motion.
- **Whole body in frame** for body work, hips and ankles included, or the
  posture readings (lean, weight shift) are skipped.
- **Face at least ~1/6 of frame height** for facial work. For close emotional
  beats, shoot a second, tighter take and capture `["face"]` only.
- **Even front light, plain background, fitted clothes.** Loose sleeves and
  backlight cost wrist and finger accuracy.
- **Steady camera** on a tripod. Camera motion is not separated from actor
  motion.
- **One performer per take.** The tools track a single person.
- **Hold a neutral pose for one second at the start.** It gives a baseline
  for that performer's resting face.
- Footage from a front-facing selfie camera is usually mirrored: pass
  `mirrored: true` so left and right hands are labelled correctly.

Get the performer's consent to store and reuse their performance, and record
who they are with `performer` on ingest.

## Capture

```json
{"operation": "download_models"}
```

once per machine (about 42 MB, three MediaPipe models), then:

```json
{
  "input_path": "takes/maya_confrontation_01.mp4",
  "capture": ["body", "hands", "face"],
  "sample_fps": 15
}
```

- `sample_fps: 15` suits library building; use `0` (every frame) when the
  motion will drive a character directly.
- Check `coverage` in the result. Below ~0.8 for a part you need means the
  performer left frame or was occluded; reshoot rather than trust it.

The JSON holds, per frame: 33 3D body landmarks in metres, 21 points per
hand, 52 ARKit-compatible face blendshapes, head yaw/pitch/roll, and derived
readings: `expression`, `gaze`, `body_language`, `energy`. ARKit blendshape
names map directly onto MetaHuman, Character Creator and most face rigs.

## Reading the Analysis Honestly

- Expression scores describe **what the face shows**, using FACS action
  units. They do not know what the performer feels. A polite smile and real
  delight both read as `joy`; sarcasm reads as whatever the face does.
- Body tags are geometric rules (wrists across the midline = `arms_crossed`).
  Context they cannot see: holding a mug, adjusting clothes, the cold.
- So **review and annotate**. Watch each beat and correct it with
  `performance_memory` `annotate`: tags like `sarcastic`, `relief`,
  `holding back tears`, and notes. Human tags count in every search.

## Library

```json
{"operation": "ingest", "capture_path": "takes/maya_confrontation_01.performance.json",
 "performer": "Maya R.", "title": "confrontation take 1", "tags": ["drama"]}
```

```json
{"operation": "query", "text": "guarded and nervous, barely moving", "limit": 5}
```

The result shows `interpreted_as` (what the words mapped to) and `why` for
every match. If `interpreted_as` is empty, use the vocabulary below or pass
`expression`, `body_tags` and `energy` directly.

Search vocabulary:

- Expressions: joy, sadness, anger, fear, surprise, disgust, contempt, neutral
  (and everyday synonyms: happy, grieving, furious, anxious, shocked, smug…).
- Body tags: arms_crossed, closed_posture, open_posture, leaning_forward,
  leaning_back, head_down, slumped, shoulders_raised, hand_to_face,
  hands_on_hips, hands_raised, head_tilted, looking_away, gaze_down,
  weight_shifted.
- Energy: low, medium, high.

`get` with `include_frames: true` returns the exact motion for a beat, ready
to retarget.

## Known Limits

- Monocular depth is estimated: forward/back lean and z-motion are the least
  accurate measures.
- Fingers are lost when hands overlap or turn edge-on to the camera.
- Fast motion blurs; shoot at 60 fps for action.
