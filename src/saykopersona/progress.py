from __future__ import annotations

import os
import sys

from tqdm.auto import tqdm


def _progress_disabled() -> bool:
    if os.environ.get("SAYKOPERSONA_PROGRESS") == "0":
        return True
    return not sys.stderr.isatty()


def make_progress(*, total: int, desc: str, unit: str, initial: int = 0, leave: bool = True) -> tqdm:
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        initial=initial,
        leave=leave,
        disable=_progress_disabled(),
    )
