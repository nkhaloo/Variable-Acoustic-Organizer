from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class VoicingConfig:
    """Configuration for deriving a stable voiced/unvoiced mask.

    This converts a per-frame voicing probability (e.g. openSMILE's
    `voicingFinalUnclipped`) into a boolean mask using:
    - hysteresis thresholds (separate on/off)
    - minimum voiced duration
    - short unvoiced gap filling

    All durations are interpreted in milliseconds.
    """

    prob_col: str = "voicingFinalUnclipped"

    # Hysteresis: switch on when prob >= on_threshold, switch off when prob <= off_threshold.
    on_threshold: float = 0.6
    off_threshold: float = 0.4

    # Remove very short voiced bursts (often spurious pitch/periodicity detections).
    min_voiced_ms: int = 20

    # Fill brief unvoiced gaps inside a voiced region.
    fill_gap_ms: int = 10

    # Frame hop in seconds (1 ms by default in this project).
    frame_step_s: float = 0.001


def _runs(mask: pd.Series) -> pd.DataFrame:
    """Return run-length encoding boundaries for a boolean series."""

    if mask.empty:
        return pd.DataFrame(columns=["start", "end", "value"])

    # Identify run starts.
    change = mask.ne(mask.shift(1, fill_value=mask.iloc[0]))
    run_id = change.cumsum()
    grouped = mask.groupby(run_id)
    starts = grouped.apply(lambda s: int(s.index[0]))
    ends = grouped.apply(lambda s: int(s.index[-1]))
    values = grouped.first().astype(bool)
    return pd.DataFrame({"start": starts, "end": ends, "value": values}).reset_index(drop=True)


def voicing_mask(df: pd.DataFrame, *, config: VoicingConfig = VoicingConfig()) -> pd.Series:
    """Compute a stable voiced/unvoiced mask from a voicing probability column.

    Returns a boolean Series aligned to `df.index`.

    This is *voicing* (periodicity) detection, not full speech activity detection.
    """

    if config.prob_col not in df.columns:
        raise KeyError(
            f"Missing voicing probability column {config.prob_col!r}. "
            "Ensure your openSMILE config exports it (VAO preset does)."
        )

    prob = pd.to_numeric(df[config.prob_col], errors="coerce")

    # Hysteresis pass.
    voiced = pd.Series(False, index=df.index)
    state = False
    for i, p in enumerate(prob.to_numpy()):
        if pd.isna(p):
            # Treat NaN as unvoiced.
            state = False
        elif not state and p >= config.on_threshold:
            state = True
        elif state and p <= config.off_threshold:
            state = False
        voiced.iat[i] = state

    # Post-processing in frame counts.
    frame_ms = config.frame_step_s * 1000.0
    min_voiced_frames = int(round(config.min_voiced_ms / frame_ms)) if config.min_voiced_ms > 0 else 0
    fill_gap_frames = int(round(config.fill_gap_ms / frame_ms)) if config.fill_gap_ms > 0 else 0

    # Remove short voiced runs.
    if min_voiced_frames > 1:
        r = _runs(voiced)
        for _, row in r.iterrows():
            if bool(row["value"]) is True:
                length = int(row["end"]) - int(row["start"]) + 1
                if length < min_voiced_frames:
                    voiced.iloc[int(row["start"]) : int(row["end"]) + 1] = False

    # Fill short unvoiced gaps between voiced regions.
    if fill_gap_frames > 0:
        r = _runs(voiced)
        for _, row in r.iterrows():
            if bool(row["value"]) is False:
                length = int(row["end"]) - int(row["start"]) + 1
                if length <= fill_gap_frames:
                    # Only fill if surrounded by voiced on both sides.
                    left_ok = int(row["start"]) > 0 and bool(voiced.iat[int(row["start"]) - 1])
                    right_ok = int(row["end"]) < (len(voiced) - 1) and bool(voiced.iat[int(row["end"]) + 1])
                    if left_ok and right_ok:
                        voiced.iloc[int(row["start"]) : int(row["end"]) + 1] = True

    return voiced


def add_voicing_columns(
    df: pd.DataFrame,
    *,
    config: VoicingConfig = VoicingConfig(),
    voiced_col: str = "voiced",
) -> pd.DataFrame:
    """Return a copy of df with a boolean `voiced` column added."""

    out = df.copy()
    out[voiced_col] = voicing_mask(out, config=config)
    return out
