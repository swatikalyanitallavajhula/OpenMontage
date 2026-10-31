"""Searchable library of captured human performances.

`performance_capture` records a take; this tool files it. Every acting beat
in a take becomes a searchable entry with its expression, body language,
gaze, energy and a plain-language description. A director agent queries it
in ordinary words ("guarded and nervous, barely moving") and gets back beats
with the exact time range of real motion to drive a character with.

Human corrections are first-class: `annotate` adds tags and notes to a beat
("sarcastic", "this is relief, not joy"), and those count in every search.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    ToolResult,
    ToolStability,
    ToolTier,
)
from tools.performance.features import EMOTIONS
from tools.performance.performance_capture import SCHEMA_ID as CAPTURE_SCHEMA_ID

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIBRARY_DIR = REPO_ROOT / "projects" / "performance_library"
INDEX_SCHEMA_ID = "openmontage.performance_library/v1"

# Everyday words a writer or director might use, mapped to what capture measures.
EXPRESSION_WORDS = {
    "joy": "happy joyful joy smiling smile delighted cheerful glad pleased laughing elated",
    "sadness": "sad sadness grief grieving sorrow sorrowful unhappy melancholy crying upset depressed heartbroken mournful",
    "anger": "angry anger furious mad rage irritated annoyed frustrated hostile",
    "fear": "afraid fear scared frightened anxious nervous terrified worried panic panicked uneasy",
    "surprise": "surprised surprise shocked astonished amazed startled",
    "disgust": "disgusted disgust repulsed revolted",
    "contempt": "contempt contemptuous smug sneering disdain scornful dismissive",
    "neutral": "neutral blank composed stoic expressionless",
}
TAG_WORDS = {
    "arms_crossed": "crossed defensive guarded",
    "closed_posture": "closed guarded defensive withdrawn protective",
    "open_posture": "open welcoming expansive confident",
    "leaning_forward": "engaged eager interested intent leaning",
    "leaning_back": "relaxed distant disengaged reclined",
    "head_down": "ashamed dejected defeated shy bowed",
    "slumped": "defeated tired exhausted dejected slouching slumped",
    "shoulders_raised": "tense tension hunched",
    "hand_to_face": "thinking pensive embarrassed contemplative",
    "hands_on_hips": "assertive impatient confrontational",
    "hands_raised": "celebrating surrender exasperated triumphant",
    "head_tilted": "curious quizzical sympathetic flirtatious",
    "looking_away": "avoidant evasive distracted",
    "gaze_down": "avoidant shy guilty downcast",
    "weight_shifted": "restless casual",
}
ENERGY_WORDS = {
    "low": "still quiet subdued calm slow motionless barely",
    "medium": "moderate",
    "high": "energetic frantic animated agitated excited explosive fast",
}
STOP_WORDS = set("a an and the of with in on to is are be but very slightly really just some who that while".split())


def _lexicon(table: dict[str, str]) -> dict[str, list[str]]:
    """Word -> every key it signals ("guarded" is both crossed arms and closed posture)."""
    lexicon: dict[str, list[str]] = {}
    for key, words in table.items():
        for word in words.split():
            lexicon.setdefault(word, []).append(key)
    return lexicon


EXPRESSION_LEXICON = _lexicon(EXPRESSION_WORDS)
TAG_LEXICON = _lexicon(TAG_WORDS)
ENERGY_LEXICON = _lexicon(ENERGY_WORDS)


def library_dir(inputs: dict[str, Any]) -> Path:
    override = inputs.get("library_dir") or os.environ.get("OPENMONTAGE_PERFORMANCE_LIBRARY")
    return Path(override) if override else DEFAULT_LIBRARY_DIR


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "take"


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]+", text.lower()) if t not in STOP_WORDS]


def parse_query(text: str) -> dict[str, Any]:
    """Map free text onto expressions, body tags, energy and leftover words."""
    expressions: list[str] = []
    tags: list[str] = []
    energy = None
    leftover: list[str] = []
    for tok in _tokens(text):
        hit = False
        if tok in EXPRESSION_LEXICON:
            expressions.extend(EXPRESSION_LEXICON[tok])
            hit = True
        if tok in TAG_LEXICON:
            tags.extend(TAG_LEXICON[tok])
            hit = True
        if tok in ENERGY_LEXICON:
            energy = ENERGY_LEXICON[tok][0]
            hit = True
        if not hit:
            leftover.append(tok)
    return {
        "expressions": sorted(set(expressions)),
        "tags": sorted(set(tags)),
        "energy": energy,
        "words": leftover,
    }


def score_segment(segment: dict[str, Any], want: dict[str, Any]) -> tuple[float, list[str]]:
    """Rank one beat against a parsed query. Returns (score, reasons)."""
    score = 0.0
    why: list[str] = []
    scores = segment.get("expression_scores") or {}
    for expr in want["expressions"]:
        if segment.get("dominant_expression") == expr:
            score += 3.0
            why.append(f"plays {expr}")
        elif scores.get(expr, 0.0) >= 0.1:
            score += 3.0 * scores[expr]
            why.append(f"some {expr} ({scores[expr]:.2f})")

    have = set(segment.get("tags", [])) | set(segment.get("user_tags", []))
    for tag in want["tags"]:
        if tag in have:
            score += 1.5
            why.append(tag.replace("_", " "))

    if want["energy"]:
        if segment.get("energy") == want["energy"]:
            score += 1.0
            why.append(f"{want['energy']} energy")
        else:
            score -= 0.5

    text = " ".join([
        segment.get("description", ""),
        " ".join(segment.get("user_tags", [])),
        segment.get("notes", ""),
        segment.get("title", ""),
        segment.get("performer", ""),
    ]).lower()
    text_tokens = set(_tokens(text))
    for word in want["words"]:
        if word in text_tokens:
            score += 0.75
            why.append(f"mentions '{word}'")

    # Prefer beats where the model actually saw the face and body.
    coverage = max(segment.get("face_coverage", 0.0), segment.get("body_coverage", 0.0))
    return round(score * (0.5 + 0.5 * coverage), 3), why


class PerformanceMemory(BaseTool):
    name = "performance_memory"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "analysis"
    provider = "local"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = []
    install_instructions = "No dependencies. Feed it captures from performance_capture."
    agent_skills = ["performance-capture"]

    capabilities = [
        "performance_library",
        "performance_search",
        "emotion_search",
        "body_language_search",
    ]
    best_for = [
        "Finding a real captured performance that matches a scene beat",
        "Keeping a growing library of how emotions and body language are played",
    ]

    input_schema = {
        "type": "object",
        "required": ["operation"],
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["ingest", "query", "list", "get", "annotate", "remove"],
            },
            "library_dir": {"type": "string"},
            "capture_path": {"type": "string", "description": "ingest: JSON written by performance_capture"},
            "performer": {"type": "string", "description": "ingest: who performed the take"},
            "title": {"type": "string", "description": "ingest: short name for the take"},
            "notes": {"type": "string", "description": "ingest/annotate: free-text notes"},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "ingest/annotate: extra tags"},
            "text": {"type": "string", "description": "query: what the beat should look like, in plain words"},
            "expression": {"type": "string", "enum": EMOTIONS + ["neutral"]},
            "body_tags": {"type": "array", "items": {"type": "string"}},
            "energy": {"type": "string", "enum": ["low", "medium", "high"]},
            "performer_filter": {"type": "string"},
            "min_seconds": {"type": "number"},
            "max_seconds": {"type": "number"},
            "limit": {"type": "integer", "default": 5},
            "segment_id": {"type": "string", "description": "get/annotate: '<performance_id>#<beat index>'"},
            "performance_id": {"type": "string", "description": "get/remove"},
            "include_frames": {"type": "boolean", "default": False, "description": "get: include the beat's frame data"},
        },
    }

    output_schema = {
        "type": "object",
        "properties": {
            "results": {"type": "array"},
            "performance_id": {"type": "string"},
            "segments_indexed": {"type": "integer"},
        },
    }

    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=500)
    side_effects = ["ingest/annotate/remove write to library_dir"]
    user_visible_verification = ["Watch the matched beat in the source video before casting it in a scene"]

    # ---- storage ----

    def _load(self, root: Path) -> dict[str, Any]:
        index_path = root / "index.json"
        if index_path.exists():
            return json.loads(index_path.read_text(encoding="utf-8"))
        return {"schema": INDEX_SCHEMA_ID, "performances": {}, "segments": []}

    def _save(self, root: Path, index: dict[str, Any]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        tmp = root / "index.json.tmp"
        tmp.write_text(json.dumps(index, indent=1), encoding="utf-8")
        tmp.replace(root / "index.json")

    # ---- operations ----

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        op = inputs.get("operation")
        handler = {
            "ingest": self._ingest,
            "query": self._query,
            "list": self._list,
            "get": self._get,
            "annotate": self._annotate,
            "remove": self._remove,
        }.get(op)
        if handler is None:
            return ToolResult(success=False, error=f"Unknown operation: {op}")
        return handler(inputs, library_dir(inputs))

    def _ingest(self, inputs: dict[str, Any], root: Path) -> ToolResult:
        capture_path = Path(inputs.get("capture_path") or "")
        if not capture_path.is_file():
            return ToolResult(success=False, error=f"capture_path not found: {capture_path}")
        if not inputs.get("performer"):
            return ToolResult(success=False, error="performer is required so every take is credited")

        raw = capture_path.read_bytes()
        capture = json.loads(raw)
        if capture.get("schema") != CAPTURE_SCHEMA_ID:
            return ToolResult(success=False, error=f"Not a performance_capture file: {capture_path}")

        digest = hashlib.sha256(raw).hexdigest()[:10]
        title = inputs.get("title") or Path(capture["source"]["path"]).stem
        perf_id = f"{_slug(title)}-{digest}"

        index = self._load(root)
        # Re-ingesting the same capture refreshes its metadata instead of duplicating it.
        index["segments"] = [s for s in index["segments"] if s["performance_id"] != perf_id]

        captures_dir = root / "captures"
        captures_dir.mkdir(parents=True, exist_ok=True)
        stored = captures_dir / f"{perf_id}.json"
        shutil.copyfile(capture_path, stored)

        extra_tags = list(inputs.get("tags") or [])
        index["performances"][perf_id] = {
            "performance_id": perf_id,
            "title": title,
            "performer": inputs["performer"],
            "notes": inputs.get("notes", ""),
            "tags": extra_tags,
            "source": capture["source"],
            "captured": capture.get("captured", []),
            "coverage": capture.get("coverage", {}),
            "summary": (capture.get("summary") or {}).get("description", ""),
            "capture_file": str(stored.relative_to(root)),
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        for seg in capture.get("segments", []):
            index["segments"].append({
                "segment_id": f"{perf_id}#{seg['index']}",
                "performance_id": perf_id,
                "title": title,
                "performer": inputs["performer"],
                "duration": round(seg["end"] - seg["start"], 3),
                "user_tags": list(extra_tags),
                "notes": "",
                **{k: seg[k] for k in seg if k != "index"},
            })
        self._save(root, index)
        count = len([s for s in index["segments"] if s["performance_id"] == perf_id])
        return ToolResult(
            success=True,
            data={"performance_id": perf_id, "segments_indexed": count, "library_dir": str(root)},
            artifacts=[str(stored)],
        )

    def _query(self, inputs: dict[str, Any], root: Path) -> ToolResult:
        want = parse_query(inputs.get("text", ""))
        if inputs.get("expression") and inputs["expression"] not in want["expressions"]:
            want["expressions"].append(inputs["expression"])
        for tag in inputs.get("body_tags") or []:
            if tag not in want["tags"]:
                want["tags"].append(tag)
        if inputs.get("energy"):
            want["energy"] = inputs["energy"]

        results = []
        for seg in self._load(root)["segments"]:
            if inputs.get("performer_filter") and seg["performer"].lower() != inputs["performer_filter"].lower():
                continue
            if inputs.get("min_seconds") is not None and seg["duration"] < inputs["min_seconds"]:
                continue
            if inputs.get("max_seconds") is not None and seg["duration"] > inputs["max_seconds"]:
                continue
            score, why = score_segment(seg, want)
            if score > 0:
                results.append({
                    "segment_id": seg["segment_id"],
                    "score": score,
                    "why": why,
                    "performer": seg["performer"],
                    "title": seg["title"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "description": seg["description"],
                    "user_tags": seg["user_tags"],
                    "notes": seg["notes"],
                })
        results.sort(key=lambda r: r["score"], reverse=True)
        return ToolResult(success=True, data={"interpreted_as": want, "results": results[: inputs.get("limit", 5)]})

    def _list(self, inputs: dict[str, Any], root: Path) -> ToolResult:
        index = self._load(root)
        performances = []
        for perf in index["performances"].values():
            beats = [s for s in index["segments"] if s["performance_id"] == perf["performance_id"]]
            performances.append({**perf, "beats": len(beats)})
        return ToolResult(success=True, data={"performances": performances, "total_beats": len(index["segments"])})

    def _get(self, inputs: dict[str, Any], root: Path) -> ToolResult:
        index = self._load(root)
        if inputs.get("segment_id"):
            seg = next((s for s in index["segments"] if s["segment_id"] == inputs["segment_id"]), None)
            if seg is None:
                return ToolResult(success=False, error=f"No beat {inputs['segment_id']}")
            data: dict[str, Any] = {"segment": seg, "source": index["performances"][seg["performance_id"]]["source"]}
            if inputs.get("include_frames"):
                capture = json.loads((root / index["performances"][seg["performance_id"]]["capture_file"]).read_text(encoding="utf-8"))
                data["frames"] = [f for f in capture["frames"] if seg["start"] <= f["t"] <= seg["end"]]
            return ToolResult(success=True, data=data)
        perf = index["performances"].get(inputs.get("performance_id", ""))
        if perf is None:
            return ToolResult(success=False, error="get needs a known segment_id or performance_id")
        beats = [s for s in index["segments"] if s["performance_id"] == perf["performance_id"]]
        return ToolResult(success=True, data={"performance": perf, "beats": beats})

    def _annotate(self, inputs: dict[str, Any], root: Path) -> ToolResult:
        index = self._load(root)
        seg = next((s for s in index["segments"] if s["segment_id"] == inputs.get("segment_id")), None)
        if seg is None:
            return ToolResult(success=False, error=f"No beat {inputs.get('segment_id')}")
        for tag in inputs.get("tags") or []:
            if tag not in seg["user_tags"]:
                seg["user_tags"].append(tag)
        if inputs.get("notes"):
            seg["notes"] = (seg["notes"] + "\n" + inputs["notes"]).strip()
        self._save(root, index)
        return ToolResult(success=True, data={"segment": seg})

    def _remove(self, inputs: dict[str, Any], root: Path) -> ToolResult:
        index = self._load(root)
        perf = index["performances"].pop(inputs.get("performance_id", ""), None)
        if perf is None:
            return ToolResult(success=False, error=f"No performance {inputs.get('performance_id')}")
        index["segments"] = [s for s in index["segments"] if s["performance_id"] != perf["performance_id"]]
        (root / perf["capture_file"]).unlink(missing_ok=True)
        self._save(root, index)
        return ToolResult(success=True, data={"removed": perf["performance_id"]})
