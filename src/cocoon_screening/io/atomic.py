"""Atomic writes and recoverable, single-writer multi-file commits."""
import json
from pathlib import Path
from uuid import uuid4

import cv2


def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def csv_bytes(table):
    return table.to_csv(index=False).encode("utf-8")


def image_bytes(image, path):
    success, encoded = cv2.imencode(Path(path).suffix, image)
    if not success:
        raise IOError(f"Could not encode image: {path}")
    return encoded.tobytes()


def recover_commit(journal):
    """Finish an interrupted commit before loading/editing its dataset."""
    journal = Path(journal).resolve()
    if not journal.exists():
        return
    root = journal.parent
    entries = json.loads(journal.read_text(encoding="utf-8"))
    paths = []
    for entry in entries:
        target = (root / entry["target"]).resolve()
        stage = (root / entry["stage"]).resolve()
        if (not target.is_relative_to(root) or not stage.is_relative_to(root)
                or stage.parent != target.parent
                or not stage.name.startswith(".pending-")):
            raise ValueError(f"Invalid recovery paths in {journal}")
        if not stage.exists() and not target.exists():
            raise FileNotFoundError(f"Missing recovery output: {target}")
        paths.append((stage, target))
    for stage, target in paths:
        if stage.exists():
            stage.replace(target)
    journal.unlink()


def commit_files(journal, outputs):
    """Stage all outputs, then journal a retryable roll-forward operation."""
    journal = Path(journal).resolve()
    recover_commit(journal)
    root = journal.parent
    stages = []
    entries = []
    try:
        for path, data in outputs.items():
            target = Path(path).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Output outside transaction directory: {target}")
            stage = target.with_name(f".pending-{uuid4().hex}{target.suffix}")
            stages.append(stage)
            write_bytes(stage, data)
            entries.append({"target": str(target.relative_to(root)),
                            "stage": str(stage.relative_to(root))})
        write_bytes(journal, json.dumps(entries).encode("utf-8"))
    except Exception:
        # Once a journal exists, keep its staged files for recovery.
        if not journal.exists():
            for stage in stages:
                stage.unlink(missing_ok=True)
        raise
    recover_commit(journal)
