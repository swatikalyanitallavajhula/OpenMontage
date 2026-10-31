"""Filing, searching and correcting captured performances."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.tools.test_performance_features import SAD, SMILE, _frames
from tools.performance import features as F
from tools.performance.performance_capture import SCHEMA_ID, PerformanceCapture
from tools.performance.performance_memory import PerformanceMemory, parse_query


def write_capture(path: Path, frames: list[dict]) -> Path:
    for f, e in zip(frames, F.motion_energy(frames)):
        f["energy"] = e
    doc = {
        "schema": SCHEMA_ID,
        "source": {"path": str(path.with_suffix(".mp4")), "kind": "video", "fps": 10},
        "captured": ["body", "face"],
        "coverage": {"body": 1.0, "face": 1.0},
        "summary": F.summarize_window(frames),
        "segments": F.segment_frames(frames),
        "frames": frames,
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@pytest.fixture()
def library(tmp_path: Path) -> tuple[PerformanceMemory, dict]:
    tool = PerformanceMemory()
    base = {"library_dir": str(tmp_path / "library")}

    happy_then_sad = write_capture(tmp_path / "take1.performance.json", _frames(2, 10, SMILE) + _frames(2, 10, SAD, start=2.0))
    crossed = _frames(3, 10, SAD)
    for f in crossed:
        f["pose_world"][F.L_WRIST] = [-0.08, -0.2, -0.1, 0.9]
        f["pose_world"][F.R_WRIST] = [0.08, -0.2, -0.1, 0.9]
        f["body_language"] = F.body_language_from_pose(f["pose_world"])
    guarded = write_capture(tmp_path / "take2.performance.json", crossed)

    r1 = tool.execute({**base, "operation": "ingest", "capture_path": str(happy_then_sad), "performer": "Maya", "title": "kitchen take 1"})
    r2 = tool.execute({**base, "operation": "ingest", "capture_path": str(guarded), "performer": "Dev", "title": "hallway take 3"})
    assert r1.success and r2.success, (r1.error, r2.error)
    return tool, {**base, "ids": [r1.data["performance_id"], r2.data["performance_id"]], "paths": [happy_then_sad, guarded]}


def query(tool: PerformanceMemory, lib: dict, **kw) -> list[dict]:
    result = tool.execute({"library_dir": lib["library_dir"], "operation": "query", **kw})
    assert result.success, result.error
    return result.data["results"]


def test_plain_language_is_interpreted() -> None:
    want = parse_query("Guarded and nervous, barely moving near the window")
    assert want["expressions"] == ["fear"]
    assert set(want["tags"]) == {"arms_crossed", "closed_posture"}
    assert want["energy"] == "low"
    assert want["words"] == ["moving", "near", "window"]


def test_query_ranks_by_expression_and_body(library) -> None:
    tool, lib = library
    happy = query(tool, lib, text="happy")
    assert happy[0]["performer"] == "Maya" and happy[0]["start"] == 0.0
    assert "plays joy" in happy[0]["why"]

    defensive = query(tool, lib, text="sad and defensive")
    assert defensive[0]["performer"] == "Dev"
    assert "arms crossed" in defensive[0]["why"]


def test_filters(library) -> None:
    tool, lib = library
    assert all(r["performer"] == "Maya" for r in query(tool, lib, text="sad", performer_filter="maya"))
    assert query(tool, lib, text="sad", min_seconds=2.5)[0]["performer"] == "Dev"


def test_human_annotations_are_searchable(library) -> None:
    tool, lib = library
    beat = query(tool, lib, text="happy")[0]["segment_id"]
    assert query(tool, lib, text="sarcastic") == []

    result = tool.execute({"library_dir": lib["library_dir"], "operation": "annotate", "segment_id": beat,
                           "tags": ["sarcastic"], "notes": "The smile is a cover."})
    assert result.success
    found = query(tool, lib, text="sarcastic")
    assert found and found[0]["segment_id"] == beat and found[0]["notes"] == "The smile is a cover."


def test_get_returns_the_beats_motion(library) -> None:
    tool, lib = library
    beat = query(tool, lib, text="happy")[0]["segment_id"]
    data = tool.execute({"library_dir": lib["library_dir"], "operation": "get", "segment_id": beat, "include_frames": True}).data
    assert data["frames"] and all(0.0 <= f["t"] <= data["segment"]["end"] for f in data["frames"])
    assert len(data["frames"][0]["pose_world"]) == 33


def test_reingest_does_not_duplicate_and_remove_cleans_up(library) -> None:
    tool, lib = library
    base = {"library_dir": lib["library_dir"]}
    before = tool.execute({**base, "operation": "list"}).data["total_beats"]
    again = tool.execute({**base, "operation": "ingest", "capture_path": str(lib["paths"][0]), "performer": "Maya", "title": "kitchen take 1"})
    assert again.data["performance_id"] == lib["ids"][0]
    assert tool.execute({**base, "operation": "list"}).data["total_beats"] == before

    assert tool.execute({**base, "operation": "remove", "performance_id": lib["ids"][0]}).success
    listing = tool.execute({**base, "operation": "list"}).data
    assert [p["performance_id"] for p in listing["performances"]] == [lib["ids"][1]]
    assert not (Path(lib["library_dir"]) / "captures" / f"{lib['ids'][0]}.json").exists()


def test_ingest_rejects_bad_input(tmp_path: Path) -> None:
    tool = PerformanceMemory()
    base = {"library_dir": str(tmp_path / "lib"), "operation": "ingest"}
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"schema": "something-else"}), encoding="utf-8")
    assert "performance_capture" in tool.execute({**base, "capture_path": str(other), "performer": "X"}).error
    assert "performer" in tool.execute({**base, "capture_path": str(other)}).error


def test_capture_reports_missing_models(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENMONTAGE_MEDIAPIPE_MODELS", str(tmp_path / "no-models"))
    clip = tmp_path / "take.mp4"
    clip.write_bytes(b"")
    result = PerformanceCapture().execute({"input_path": str(clip)})
    assert not result.success and "download_models" in result.error
