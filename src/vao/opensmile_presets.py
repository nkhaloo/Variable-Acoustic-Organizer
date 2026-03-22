from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class OpenSmilePreset:
    """A VAO-managed openSMILE preset.

    `config_path` points to a VAO-shipped wrapper config.
    `extra_args` contains config-macro overrides that reference the user's
    local openSMILE installation (typically includes under `<OPENSMILE_HOME>/config`).
    """

    config_path: Path
    extra_args: tuple[str, ...]


_PRESET_COMPARE16_LLD_25MS_1MS = "compare16_lld_25ms_1ms"

_PRESET_EGEMAPSV02_LLD_25MS_1MS = "egemapsv02_lld_25ms_1ms"


def preset_egemapsv02_lld_25ms_1ms(*, opensmile_home: str | Path) -> OpenSmilePreset:
    """Frame-level eGeMAPSv02 preset (25ms window, 1ms hop).

    Users still need to install/build openSMILE (for the `SMILExtract` binary),
    but they do NOT need to modify their openSMILE repo.

    This preset:
    - Uses VAO wrapper config: `vao/opensmile_configs/egemapsv02_lld_25ms_1ms.conf`
    - Pulls feature definitions from openSMILE's shipped GeMAPS/eGeMAPS includes.
    """

    opensmile_home = Path(opensmile_home).expanduser().resolve()

    std_wave_input = opensmile_home / "config" / "shared" / "standard_wave_input.conf.inc"
    gemaps_core_lld = opensmile_home / "config" / "gemaps" / "v01b" / "GeMAPSv01b_core.lld.conf.inc"
    egemaps_core_lld = opensmile_home / "config" / "egemaps" / "v02" / "eGeMAPSv02_core.lld.conf.inc"
    compare16_core_lld = opensmile_home / "config" / "compare16" / "ComParE_2016_core.lld.conf.inc"

    buffer_rb = opensmile_home / "config" / "shared" / "BufferModeRb.conf.inc"
    buffer_rblag = opensmile_home / "config" / "shared" / "BufferModeRbLag.conf.inc"
    buffer_conf = opensmile_home / "config" / "shared" / "BufferMode.conf.inc"

    missing = [
        p
        for p in (
            std_wave_input,
            gemaps_core_lld,
            egemaps_core_lld,
            compare16_core_lld,
            buffer_rb,
            buffer_rblag,
            buffer_conf,
        )
        if not p.is_file()
    ]
    if missing:
        missing_str = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            "openSMILE config includes not found under opensmile_home.\n"
            "Expected to find (at least):\n" + missing_str
        )

    config_path = Path(__file__).resolve().parent / "opensmile_configs" / "egemapsv02_lld_25ms_1ms.conf"
    if not config_path.is_file():
        raise FileNotFoundError(f"VAO packaged config missing: {config_path}")

    extra_args: list[str] = [
        "-stdWaveInput",
        str(std_wave_input),
        "-gemapsCoreLld",
        str(gemaps_core_lld),
        "-egemapsCoreLld",
        str(egemaps_core_lld),
        "-compare16CoreLld",
        str(compare16_core_lld),
        "-bufferModeRbConf",
        str(buffer_rb),
        "-bufferModeRbLagConf",
        str(buffer_rblag),
        "-bufferModeConf",
        str(buffer_conf),
    ]

    return OpenSmilePreset(config_path=config_path, extra_args=tuple(extra_args))


def get_preset(name: str, *, opensmile_home: str | Path) -> OpenSmilePreset:
    """Resolve a named VAO preset."""

    key = name.strip().lower()
    if key in {_PRESET_EGEMAPSV02_LLD_25MS_1MS, "egemapsv02", "egemaps", "egemaps_lld"}:
        return preset_egemapsv02_lld_25ms_1ms(opensmile_home=opensmile_home)

    raise ValueError(
        f"Unknown preset {name!r}. Supported presets: {_PRESET_EGEMAPSV02_LLD_25MS_1MS!r}"
    )


def list_presets() -> Sequence[str]:
    return (_PRESET_EGEMAPSV02_LLD_25MS_1MS,)
