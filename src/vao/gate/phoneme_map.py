# create classes for timit phonemes
from __future__ import annotations

SILENCE = "silence"
OBSTRUENT = "obstruent"
SONORANT = "sonorant"

# TIMIT phoneme → segment class
TIMIT_PHONEME_MAP: dict[str, str] = {
    # --- silence & non-speech ---
    "h#": SILENCE,
    "pau": SILENCE,
    "epi": SILENCE,
    # stop closures are acoustically silent
    "bcl": SILENCE,
    "dcl": SILENCE,
    "gcl": SILENCE,
    "pcl": SILENCE,
    "tcl": SILENCE,
    "kcl": SILENCE,

    # --- obstruents ---
    # stops (burst/release)
    "b": OBSTRUENT,
    "d": OBSTRUENT,
    "g": OBSTRUENT,
    "p": OBSTRUENT,
    "t": OBSTRUENT,
    "k": OBSTRUENT,
    "dx": OBSTRUENT,   # flap
    "q": OBSTRUENT,    # glottal stop
    # affricates
    "jh": OBSTRUENT,
    "ch": OBSTRUENT,
    # fricatives
    "s": OBSTRUENT,
    "sh": OBSTRUENT,
    "z": OBSTRUENT,
    "zh": OBSTRUENT,
    "f": OBSTRUENT,
    "th": OBSTRUENT,
    "v": OBSTRUENT,
    "dh": OBSTRUENT,

    # --- sonorants ---
    # nasals
    "m": SONORANT,
    "n": SONORANT,
    "ng": SONORANT,
    "em": SONORANT,
    "en": SONORANT,
    "eng": SONORANT,
    "nx": SONORANT,
    # semivowels & glides
    "l": SONORANT,
    "r": SONORANT,
    "w": SONORANT,
    "y": SONORANT,
    "hh": SONORANT,
    "hv": SONORANT,
    "el": SONORANT,
    # vowels
    "iy": SONORANT,
    "ih": SONORANT,
    "eh": SONORANT,
    "ey": SONORANT,
    "ae": SONORANT,
    "aa": SONORANT,
    "aw": SONORANT,
    "ay": SONORANT,
    "ah": SONORANT,
    "ao": SONORANT,
    "oy": SONORANT,
    "ow": SONORANT,
    "uh": SONORANT,
    "uw": SONORANT,
    "ux": SONORANT,
    "er": SONORANT,
    "ax": SONORANT,
    "ix": SONORANT,
    "axr": SONORANT,
    "ax-h": SONORANT,
}


def timit_label(phoneme: str) -> str | None:
    """Map a TIMIT phoneme symbol to silence/obstruent/sonorant. Returns None if unknown."""
    return TIMIT_PHONEME_MAP.get(phoneme.lower())
