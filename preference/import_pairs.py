"""Copy aligned edits into immutable comparisons, leaving them unrated."""

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps

from .config import file_identity, model_identity, split_for_prompt, utc_now
from .store import PreferenceStore


def import_pairs(config, manifest_path):
    manifest_path = Path(manifest_path).resolve(strict=True)
    entries = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Import manifest must be a nonempty list of source/target/prompt objects")
    identity = model_identity(config["model"])
    store = PreferenceStore(config["dataset_dir"])
    existing = store.list_comparisons()
    if any(r["model"].get("reference_id") != identity["reference_id"] for r in existing):
        raise ValueError("Choose a dataset for this reference checkpoint/adapter combination")
    synthetic = config["training"]["allow_synthetic"]
    if any(r["synthetic"] != synthetic for r in existing):
        raise ValueError("Human reviews and synthetic test labels need separate datasets")
    groups = {}
    for row in existing:
        if row["group_id"] in groups and groups[row["group_id"]] != row["split"]:
            raise ValueError("Dataset contains conflicting prompt splits")
        groups[row["group_id"]] = row["split"]
    # Validate the entire manifest before committing any images or records.
    prepared = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("aligned") is not True:
            raise ValueError("Each edit pair must explicitly declare aligned: true")
        if not isinstance(entry.get("prompt"), str) or not entry["prompt"].strip():
            raise ValueError("Each pair needs the prompt describing the common scene")
        if not isinstance(entry.get("negative_prompt", ""), str):
            raise ValueError("negative_prompt must be text")
        if type(entry.get("seed", 0)) is not int:
            raise ValueError("seed must be an integer, or omit it if unknown")
        paths, sizes, hashes = [], [], []
        for role in ("source", "target"):
            if not isinstance(entry.get(role), str) or not entry[role].strip():
                raise ValueError(f"Each pair needs a {role} image path")
            path = Path(entry[role])
            path = (manifest_path.parent / path).resolve(strict=True)
            with Image.open(path) as image:
                image.load()
                sizes.append(ImageOps.exif_transpose(image).size)
            paths.append(path)
            hashes.append(file_identity(path)["sha256"])
        if sizes[0] != sizes[1] or any(n % 64 or not 256 <= n <= 1536 for n in sizes[0]):
            raise ValueError("Aligned images must have identical dimensions, multiples of 64 from 256 to 1536")
        if hashes[0] == hashes[1]:
            raise ValueError("Source and target are identical files; choose an actual edit")
        key = [identity["reference_id"], hashes, entry["prompt"], entry.get("negative_prompt", "")]
        cid = "edit-" + hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:32]
        prepared.append((cid, entry, paths, sizes[0], hashes))
    known = {r["id"] for r in existing}
    session = uuid4().hex
    added = 0
    for cid, entry, paths, size, hashes in prepared:
        if cid in known:
            continue
        g = config["generation"]
        group, split = split_for_prompt(entry["prompt"], g["validation_fraction"], g["split_seed"])
        split = groups.setdefault(group, split)
        images = []
        for side, path, source_hash in zip(("a", "b"), paths, hashes):
            if file_identity(path)["sha256"] != source_hash:
                raise ValueError("An import source changed during validation; retry after editing is finished")
            relative = f"images/{cid}-{session}-{side}.png"
            destination = store.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as image, destination.open("xb") as handle:
                ImageOps.exif_transpose(image).convert("RGB").save(handle, format="PNG")
            if file_identity(path)["sha256"] != source_hash:
                raise ValueError("An import source changed while copying; no comparison was committed")
            images.append({"id": side, "path": relative, "seed": entry.get("seed", 0),
                           "sha256": file_identity(destination)["sha256"],
                           "import_source_sha256": source_hash, "import_source_path": str(path)})
        record = {"id": cid, "created_at": utc_now(), "session_id": session,
                  "group_id": group, "split": split, "synthetic": synthetic,
                  "prompt": entry["prompt"], "negative_prompt": entry.get("negative_prompt", ""),
                  "model": identity, "images": images, "pair_kind": "aligned_edit_v1",
                  "generation_settings": {"width": size[0], "height": size[1],
                      "engine": "external-aligned-import-v1", "seed_known": "seed" in entry,
                      "source_side": "a", "edited_side": "b", "alignment": "user-declared"}}
        store.add_comparison(record)
        known.add(cid)
        added += 1
    result = {"imported": added, "duplicates_skipped": len(entries) - added,
              "dataset_dir": str(store.root), "next": "Rate each pair; edits are not automatically preferred."}
    print(json.dumps(result, indent=2), flush=True)
    return result
