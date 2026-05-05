from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from .audio_preprocess import preprocess_folder
from .features import extract_features_folder
from .opensmile_presets import get_preset


DEFAULT_PRESET = "egemapsv02_lld_25ms_10ms"


def _smooth_segments(
    labels: pd.Series, min_frames: int, min_silence_ms: int, frame_step_s: float
) -> pd.Series:
    """Remove short segment runs by relabeling them to their dominant neighbor.

    Silence runs use a separate (higher) threshold since the gate over-predicts silence.
    Applied per recording so transitions don't bleed across files.
    """
    min_silence_frames = max(1, round(min_silence_ms / (frame_step_s * 1000)))
    labels = labels.copy()
    values = labels.to_numpy()
    n = len(values)

    changed = True
    while changed:
        changed = False
        i = 0
        while i < n:
            run_val = values[i]
            j = i
            while j < n and values[j] == run_val:
                j += 1
            run_len = j - i
            threshold = min_silence_frames if run_val == "silence" else min_frames
            if run_len < threshold:
                left = values[i - 1] if i > 0 else None
                right = values[j] if j < n else None
                replacement = left if right is None else (right if left is None else left)
                values[i:j] = replacement
                changed = True
            i = j

    labels.iloc[:] = values
    return labels


def vao_extract(
    wav_dir: str | Path,
    *,
    opensmile_home: str | Path | None = None,
    opensmile_default: str | Path | None = None,
    preset: str = DEFAULT_PRESET,
    frame_step_s: float = 0.010,
    workers: int | None = None,
    recursive: bool = False,
    preprocess: bool = True,
    apply_gate: bool = True,
    smooth_gate: bool = False,
    min_segment_ms: int = 30,
    min_silence_ms: int = 100,
    mask_features: bool = False,
    normalize: bool = False,
    frame_level: bool = True,
) -> pd.DataFrame:
    """One-call batch extraction for a folder of WAV files.

    This is the minimal user-facing API:
    - User supplies a directory of WAV files
    - User supplies their local openSMILE root folder (where `config/` lives)
    - VAO runs the eGeMAPSv02 frame-level preset (25 ms window, 10 ms hop)
    - Returns a single combined DataFrame; save it yourself with df.to_csv(...)

    Args:
        wav_dir: Folder containing `.wav` files.
        opensmile_home: Path to the openSMILE repo/install root.
        opensmile_default: Alias for `opensmile_home` (kept for user ergonomics).
        preset: Preset name to use. Defaults to `egemapsv02_lld_25ms_10ms`.
        frame_step_s: Hop size in seconds (should match the wrapper config).
        workers: Number of parallel processes. Defaults to all available CPU cores.
        preprocess: If True, convert all audio files to 16 kHz mono WAV before
            extraction. Preprocessed files are written to `<wav_dir>/vao_output/preprocessed`.
            Requires ffmpeg on PATH.
        apply_gate: If True, add a `segment_class` column (silence/obstruent/sonorant).
            Requires the trained gate model to be present. Defaults to True.
        smooth_gate: If True, apply temporal smoothing to segment_class predictions —
            removes short isolated segments and fills brief gaps. Requires apply_gate=True.
        min_segment_ms: Minimum duration in ms for obstruent/sonorant runs. Shorter runs
            get relabeled to their neighbor. Defaults to 30ms (3 frames).
        min_silence_ms: Minimum duration in ms for silence runs. Higher than min_segment_ms
            because the gate over-predicts silence. Defaults to 100ms (10 frames).
        mask_features: If True, NaN out acoustically invalid features per segment class.
            Silence → NaN all features. Obstruent → NaN F0, jitter, shimmer, H1-H2,
            H1-A3, formant frequencies, bandwidths, and amplitudes (+ their deltas).
            Sonorant → no masking (keeps all features).
            Requires apply_gate=True. Defaults to False.
        normalize: If True, apply per-recording z-score normalization to all numeric
            acoustic feature columns (excludes time, recording, and segment_class).
        frame_level: If True (default), return one row per 10ms frame. If False,
            aggregate to one row per recording using mean and std of each feature —
            the traditional OpenSMILE functional approach.
            Useful before PCA or distance-based models. Defaults to False.

    Returns:
        Combined DataFrame with a `recording` column and `segment_class` if apply_gate=True.
    """

    if opensmile_home is None and opensmile_default is None:
        raise ValueError("Missing openSMILE path. Pass `opensmile_home=...` (or `opensmile_default=...`).")
    if opensmile_home is not None and opensmile_default is not None:
        raise ValueError("Pass only one of `opensmile_home` or `opensmile_default`.")

    resolved_opensmile_home = opensmile_home if opensmile_home is not None else opensmile_default
    assert resolved_opensmile_home is not None

    wav_dir_path = Path(wav_dir).expanduser()
    opensmile_home_path = Path(resolved_opensmile_home).expanduser()

    preset_obj = get_preset(preset, opensmile_home=opensmile_home_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        if preprocess:
            preprocess_folder(wav_dir_path, tmp_dir, recursive=recursive)
            wav_dir_path = Path(tmp_dir)

        df = extract_features_folder(
            wav_dir_path,
            config_path=preset_obj.config_path,
            output_dir=Path(tmp_dir),
            workers=workers,
            recursive=recursive,
            opensmile_home=opensmile_home_path,
            output_option="-csvoutput",
            extra_args=preset_obj.extra_args,
            frame_step_s=frame_step_s,
        )

    if apply_gate:
        from .gate.classifier import apply_gate as _apply_gate
        df = _apply_gate(df)

    if smooth_gate:
        if "segment_class" not in df.columns:
            raise ValueError("smooth_gate=True requires apply_gate=True.")
        min_frames = max(1, round(min_segment_ms / (frame_step_s * 1000)))
        df["segment_class"] = (
            df.groupby("recording", group_keys=False)["segment_class"]
            .transform(lambda s: _smooth_segments(s, min_frames, min_silence_ms, frame_step_s))
        )

    if mask_features:
        if "segment_class" not in df.columns:
            raise ValueError("mask_features=True requires apply_gate=True.")

        # Features that are only meaningful for sonorants — NaN for obstruents and silence.
        # Includes base columns and their _de (delta) variants.
        _OBSTRUENT_NAN_PREFIXES = (
            "F0semitoneFrom27.5Hz",
            "jitterLocal",
            "shimmerLocaldB",
            "logRelF0-H1-H2",
            "logRelF0-H1-A3",
            "F1frequency", "F2frequency", "F3frequency", "F4frequency", "F5frequency",
            "F1bandwidth", "F2bandwidth", "F3bandwidth", "F4bandwidth", "F5bandwidth",
            "F1amplitudeLogRelF0", "F2amplitudeLogRelF0", "F3amplitudeLogRelF0",
        )
        obstruent_nan_cols = [
            c for c in df.columns
            if any(c.startswith(p) for p in _OBSTRUENT_NAN_PREFIXES)
        ]

        non_meta_cols = [
            c for c in df.columns
            if c not in {"recording", "segment_class", "time_s", "frameTime",
                         "frame_time", "timestamp", "name", "label"}
        ]

        # Silence: NaN all acoustic features
        silence_mask = df["segment_class"] == "silence"
        df.loc[silence_mask, non_meta_cols] = pd.NA

        # Obstruent: NaN F0, formant frequencies, bandwidths, amplitudes + deltas
        obstruent_mask = df["segment_class"] == "obstruent"
        df.loc[obstruent_mask, obstruent_nan_cols] = pd.NA

    if normalize:
        non_feature_cols = {"recording", "segment_class", "time_s", "frameTime",
                            "frame_time", "timestamp", "name"}
        feature_cols = [c for c in df.columns
                        if c not in non_feature_cols and pd.api.types.is_numeric_dtype(df[c])]
        # z-score per recording so cross-recording level differences don't affect the scale
        df[feature_cols] = (
            df.groupby("recording")[feature_cols]
            .transform(lambda x: (x - x.mean()) / x.std())
        )

    if not frame_level:
        meta_cols = {"recording", "segment_class", "time_s", "frameTime",
                     "frame_time", "timestamp", "name"}
        feature_cols = [c for c in df.columns
                        if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
        mean_df = df.groupby("recording")[feature_cols].mean().add_suffix("_mean")
        std_df = df.groupby("recording")[feature_cols].std().add_suffix("_std")
        df = pd.concat([mean_df, std_df], axis=1).reset_index()

    return df
