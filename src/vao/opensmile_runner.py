from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .errors import OpenSmileNotFoundError, OpenSmileRunError


@dataclass(frozen=True)
class OpenSmilePaths:
    """Resolved paths for calling openSMILE."""

    smileextract: Path


def _as_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path


def find_smileextract(
    *,
    smileextract_path: str | os.PathLike[str] | None = None,
    opensmile_home: str | os.PathLike[str] | None = None,
) -> OpenSmilePaths:
    """Resolve the SMILExtract executable.

    Resolution order:
    1) Explicit `smileextract_path`
    2) env var `OPENSMILE_SMILEEXTRACT`
    3) `opensmile_home` arg or env var `OPENSMILE_HOME` with common build locations
    4) `SMILExtract` on PATH

    Raises:
        OpenSmileNotFoundError: if the executable cannot be found.
    """

    candidates: list[Path] = []

    if smileextract_path is not None:
        candidates.append(_as_path(smileextract_path))

    env_smileextract = os.environ.get("OPENSMILE_SMILEEXTRACT")
    if env_smileextract:
        candidates.append(_as_path(env_smileextract))

    home = opensmile_home or os.environ.get("OPENSMILE_HOME")
    if home:
        home_path = _as_path(home)
        candidates.extend(
            [
                home_path / "build" / "progsrc" / "smilextract" / "SMILExtract",
                home_path / "progsrc" / "smilextract" / "SMILExtract",
                home_path / "bin" / "SMILExtract",
            ]
        )

    which = shutil.which("SMILExtract")
    if which:
        candidates.append(Path(which))

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            continue

        if resolved.is_file() and os.access(resolved, os.X_OK):
            return OpenSmilePaths(smileextract=resolved)

    message = (
        "Could not find SMILExtract. Provide `smileextract_path=...`, "
        "set env var OPENSMILE_SMILEEXTRACT, or set OPENSMILE_HOME (e.g. the openSMILE repo root)."
    )
    raise OpenSmileNotFoundError(message)


def run_smileextract(
    *,
    input_wav: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    smileextract_path: str | os.PathLike[str] | None = None,
    opensmile_home: str | os.PathLike[str] | None = None,
    input_option: str = "-I",
    output_option: str = "-O",
    loglevel: int = 2,
    extra_args: Sequence[str] = (),
    cwd: str | os.PathLike[str] | None = None,
) -> None:
    """Run openSMILE SMILExtract to produce an output file (typically CSV).

    Note: `input_option`/`output_option` depend on the chosen openSMILE config.
    Many example configs support `-I` and `-O`, but some use different flags.
    """

    paths = find_smileextract(smileextract_path=smileextract_path, opensmile_home=opensmile_home)

    cmd: list[str] = [
        str(paths.smileextract),
        "-C",
        str(_as_path(config_path)),
        "-l",
        str(int(loglevel)),
        input_option,
        str(_as_path(input_wav)),
        output_option,
        str(_as_path(output_path)),
        *list(extra_args),
    ]

    try:
        subprocess.run(cmd, check=True, cwd=str(_as_path(cwd)) if cwd else None)
    except subprocess.CalledProcessError as exc:
        raise OpenSmileRunError(f"SMILExtract failed with exit code {exc.returncode}: {cmd}") from exc
