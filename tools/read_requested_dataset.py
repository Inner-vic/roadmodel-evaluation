"""Read the user-specified dataset using the existing personal Asset SDK connection.

No credential values are read by this script, printed, copied, or saved. SDK
authentication uses the bearer-file reference already configured by the user.
Only the exact data_link object is downloaded here, not the training dataset.
"""
import contextlib
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "evidence" / "requested-dataset"
SOURCE = "s3://s3.bj.bcebos.com/smartchassis-roadmodel-data/datasets/DataReorganized-V161-minidataset-test/"
NAME = "data_link_26-10-20-1_train859_eval135_v7.json"


class Quiet:
    def write(self, text):
        return len(text)

    def flush(self):
        pass


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    facts = {"dataset": "smartchassisdata-v3", "version": "26-10-20-1", "source_uri": SOURCE,
             "requested_file": NAME, "cloud_writes": 0, "downloaded_bytes": 0}
    try:
        root = Path("C:/Users/developer/AppData/Local/Packages/OpenAI.Codex_2p2nqsd0c76g0/LocalCache/Local/IterLab/data")
        config = json.loads((root / "integrations.local.json").read_text(encoding="utf-8-sig"))
        connection = next(c for c in config["connections"] if c["id"] == "lpai-sdk-closedloop-20261022")
        print("Reading the specified dataset version through the existing Asset SDK connection", flush=True)
        with contextlib.redirect_stdout(Quiet()), contextlib.redirect_stderr(Quiet()):
            import asset
            resource = asset.resource("datasets/smartchassisdata-v3/versions/26-10-20-1",
                                      config=asset.config(env="prod", bearer_file=connection["settings"]["asset_bearer_file"]))
            detail = resource.get()
            facts["resource_top_keys"] = sorted(detail.keys())
            meta, spec = detail.get("metadata", {}), detail.get("spec", {})
            facts["identity"] = {k: meta.get(k) for k in ("name", "uid", "namespace")}
            facts["declared_path"] = spec.get("dir", {}).get("path")
            if facts["declared_path"].rstrip("/") != SOURCE.rstrip("/"):
                raise ValueError("Dataset version path differs from requested source")
            facts["stage"] = "version_identity_verified"
            (OUT / "access-facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
            head = resource.head_file(s3_path=SOURCE + NAME)
            size = head.get("ContentLength")
            facts["data_link_size_bytes"] = size
            if type(size) is not int or not 0 < size <= 64 * 1024 * 1024:
                raise ValueError("Unexpected data_link object size")
            response = resource.get_file(s3_path=SOURCE + NAME)
            try:
                raw = response.read(size + 1)
            finally:
                response.close()
            if len(raw) != size:
                raise ValueError("data_link byte count mismatch")
            data = json.loads(raw)
            facts["split_counts"] = {}
            for split in ("train", "eval", "joint_eval", "vision_eval"):
                scenes = data.get(split)
                if isinstance(scenes, dict):
                    facts["split_counts"][split] = {"scenes": len(scenes), "frames": sum(len(s.get("frames", [])) for s in scenes.values() if isinstance(s, dict))}
            target = OUT / NAME
            if target.exists() and target.read_bytes() != raw:
                raise ValueError("A different data_link already exists at target")
            target.write_bytes(raw)
            facts.update(stage="data_link_downloaded", downloaded_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), local_file=str(target))
    except Exception as error:
        facts.update(stage=facts.get("stage", "connect") + "_failed", error_type=type(error).__name__)
        status = getattr(error, "status", None)
        if type(status) is int:
            facts["http_status"] = status
    (OUT / "access-facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(facts, ensure_ascii=True, indent=2), flush=True)
    return 0 if facts["downloaded_bytes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
