"""Atomic JSON writes that tolerate short Windows/OneDrive file locks."""

import json
import os
from pathlib import Path
import tempfile
import time


def write_json(path: Path, value: dict, *, required: bool = True) -> bool:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.stem}-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2)
        for attempt in range(9):
            try:
                os.replace(temporary, path)
                return True
            except PermissionError:
                if attempt == 8:
                    raise
                time.sleep(min(0.02 * 2**attempt, 0.2))
    except OSError:
        if required:
            raise
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
