"""Inspect the split authority without importing the training framework.

This inventories exact frame metadata. It deliberately does not invent a local
file layout, remap labels or substitute missing annotations.
"""
from __future__ import annotations

from pathlib import Path

from .common import digest, load_json


def inspect_datalink(path: str | Path, splits: list[str] | None = None) -> dict:
    path = Path(path).resolve()
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("Datalink must be an object")
    splits = splits or ["eval"]
    if not splits or any(s not in ("eval", "joint_eval", "vision_eval") for s in splits):
        raise ValueError("Only explicit evaluation splits may be selected")
    records, scene_names, identities, issues = [], set(), set(), []
    for split in splits:
        scenes = data.get(split)
        if not isinstance(scenes, dict):
            raise ValueError(f"Datalink does not contain a scene-object split: {split}")
        for scene, info in scenes.items():
            if not isinstance(info, dict) or not isinstance(info.get("frames"), list):
                raise ValueError(f"{split}/{scene}: expected frames array")
            scene_names.add(scene)
            for frame in info["frames"]:
                if not isinstance(frame, dict) or "frame_index" not in frame:
                    raise ValueError(f"{split}/{scene}: missing frame_index")
                identity = (scene, str(frame["frame_index"]), str(frame.get("time_stamp")))
                if identity in identities:
                    issues.append({"code": "duplicate_frame", "scene": scene, "frame_index": frame["frame_index"]})
                    continue
                identities.add(identity)
                records.append({"split": split, "scene": scene, "frame_index": frame["frame_index"],
                                "time_stamp": frame.get("time_stamp"),
                                "scene_metadata": {k: v for k, v in info.items() if k != "frames"},
                                "frame_metadata": frame})
    return {"schema": "pcm.datalink-inventory/v1", "source": str(path), "sha256": digest(path),
            "splits": splits, "scene_count": len(scene_names), "frame_count": len(records),
            "issues": issues, "frames": records,
            "note": "Metadata inventory only; file availability and V7 preprocessing are not validated."}
