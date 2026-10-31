# Performance Director

Use this meta skill when a story with human or humanlike characters will be
animated from captured performances (`performance_capture` +
`performance_memory`). Read `.agents/skills/performance-capture/SKILL.md`
before calling either tool.

The writer owns the story and the scene order. The director's job is to turn
each scene into acting beats, cast a real captured performance for every
beat, and say plainly where the library has nothing good enough so the
writer can shoot it.

## 1. Break Scenes into Beats

A beat is one change in what a character wants or feels, usually 1–6
seconds. For every scene the writer gives you, list beats per character:

```json
{
  "scene_id": "s03",
  "character": "Maya",
  "beat": 2,
  "story_moment": "She hears the accusation and refuses to show it hurts",
  "subtext": "hurt, hiding it",
  "visible_performance": "neutral face fighting sadness, arms crossed, very still, eyes down",
  "duration_seconds": 3.5,
  "framing": "medium close-up"
}
```

`visible_performance` is the line that matters: describe what a camera would
see, not the inner state. Subtext belongs in its own field. Searching for
"hurt" finds nothing; searching for "neutral face, arms crossed, still, eyes
down" does.

## 2. Cast Each Beat from the Library

Query `performance_memory` with the `visible_performance` text, a duration
filter around the beat length, and the actor if the character has one
performer for continuity.

- Read `interpreted_as`; rephrase if your key words were dropped.
- Prefer beats whose `why` matches the expression **and** the body cues.
- Same character, same performer across a scene whenever possible.
- Record the chosen `segment_id`, its score and its `why` in the beat.

## 3. Report Gaps, Do Not Paper Over Them

A beat with no match scoring at least ~3, or a match that only fits on energy,
is a **gap**. Never cast a beat whose expression contradicts the script.
Give the writer a shot list for the gaps, phrased as acting direction:

> Beat s03/2 — Maya. Medium close-up, 4 s. Arms crossed, almost no
> movement, eyes drop to the floor and stay there, mouth pressed. She is
> holding back tears; nothing should leak except the swallow at the end.

After the take is captured, ingest and annotate it, then cast again.

## 4. Shot File

The output of directing is one shot file per scene, which the render stage
reads:

```json
{
  "scene_id": "s03",
  "setting": "kitchen, night, single practical light over the table",
  "shots": [
    {
      "shot_id": "s03_sh02",
      "camera": {"framing": "medium close-up", "lens_mm": 50, "move": "slow push-in", "height": "eye level"},
      "lighting_mood": "low-key, warm practical, cold fill from the window",
      "duration_seconds": 3.5,
      "characters": [
        {"character": "Maya", "segment_id": "maya-confrontation-take-1-ab12cd34ef#4",
         "trim": {"start": 12.4, "end": 15.9}, "blocking": "seated, facing camera left"}
      ],
      "dialogue": null,
      "notes": "Hold on her after the line lands."
    }
  ],
  "gaps": []
}
```

## 5. Checkpoint

Present the beat list, castings with their `why`, and every gap to the writer
before any rendering. The writer approves castings, asks for recasts, or
schedules the gap shoots.
