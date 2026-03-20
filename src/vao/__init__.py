"""VAO (Variable Acoustic Organizer).

Feature extraction (openSMILE) + rule-based acoustic labeling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
	"__version__",
	"VAOError",
	"OpenSmileNotFoundError",
	"OpenSmileRunError",
	"vao_extract",
	"VoicingConfig",
	"voicing_mask",
	"add_voicing_columns",
	"get_presets",
	"get_preset",
	"list_presets",
	"extract_features",
	"extract_features_folder",
	"find_smileextract",
	"run_smileextract",
]

__version__ = "0.1.0"


if TYPE_CHECKING:
	from .errors import OpenSmileNotFoundError as OpenSmileNotFoundError
	from .errors import OpenSmileRunError as OpenSmileRunError
	from .errors import VAOError as VAOError
	from .api import vao_extract as vao_extract
	from .voicing import VoicingConfig as VoicingConfig
	from .voicing import add_voicing_columns as add_voicing_columns
	from .voicing import voicing_mask as voicing_mask
	from .features import extract_features as extract_features
	from .features import extract_features_folder as extract_features_folder
	from .opensmile_presets import get_preset as get_preset
	from .opensmile_presets import list_presets as list_presets
	from .opensmile_runner import find_smileextract as find_smileextract
	from .opensmile_runner import run_smileextract as run_smileextract


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
	# errors
	"VAOError": ("vao.errors", "VAOError"),
	"OpenSmileNotFoundError": ("vao.errors", "OpenSmileNotFoundError"),
	"OpenSmileRunError": ("vao.errors", "OpenSmileRunError"),
	# features
	"extract_features": ("vao.features", "extract_features"),
	"extract_features_folder": ("vao.features", "extract_features_folder"),
	"vao_extract": ("vao.api", "vao_extract"),
	# voicing helpers
	"VoicingConfig": ("vao.voicing", "VoicingConfig"),
	"voicing_mask": ("vao.voicing", "voicing_mask"),
	"add_voicing_columns": ("vao.voicing", "add_voicing_columns"),
	# runner
	"find_smileextract": ("vao.opensmile_runner", "find_smileextract"),
	"run_smileextract": ("vao.opensmile_runner", "run_smileextract"),
	# presets
	"get_presets": ("vao.opensmile_presets", "get_preset"),
	"get_preset": ("vao.opensmile_presets", "get_preset"),
	"list_presets": ("vao.opensmile_presets", "list_presets"),
}


def __getattr__(name: str):
	"""Lazy top-level exports.

	Keeps `import vao` lightweight while supporting `from vao import ...`.
	"""

	target = _LAZY_EXPORTS.get(name)
	if target is not None:
		module_name, attr_name = target
		module = __import__(module_name, fromlist=[attr_name])
		return getattr(module, attr_name)
	raise AttributeError(f"module 'vao' has no attribute {name!r}")
