import re
from typing import Any, Callable, Dict, Optional

_SUPPORTED_TRANSFORMERS_MIN = (5, 1, 0)
_SUPPORTED_TRANSFORMERS_MAX_EXCLUSIVE = (5, 2, 0)
_SUPPORTED_TRANSFORMERS_RANGE = ">=5.1.0,<5.2.0"


def _parse_transformers_version(version: str) -> tuple[int, int, int]:
    match = re.match(r"^\s*(\d+)\.(\d+)(?:\.(\d+))?", str(version))
    if match is None:
        raise ValueError(f"Unrecognized transformers version: {version!r}")
    major, minor, patch = match.groups(default="0")
    return int(major), int(minor), int(patch)


def check_vlm_dependencies(
    import_module: Callable[[str], Any] = __import__,
    transformers_version_provider: Optional[Callable[[], str]] = None,
) -> Dict[str, Any]:
    """Collect dependency diagnostics for VLM runtime without direct logging side effects."""
    result: Dict[str, Any] = {
        "transformers_warning": None,
        "missing_deps": [],
        "install_command": None,
    }

    try:
        version = (
            transformers_version_provider()
            if transformers_version_provider is not None
            else import_module("transformers").__version__
        )
        parsed_version = _parse_transformers_version(version)
        if not (_SUPPORTED_TRANSFORMERS_MIN <= parsed_version < _SUPPORTED_TRANSFORMERS_MAX_EXCLUSIVE):
            result["transformers_warning"] = (
                f"transformers {version} is outside the repo's validated range {_SUPPORTED_TRANSFORMERS_RANGE}. "
                "DINOv3 landed in Transformers 4.56.0+, but the maintained baseline is the latest validated 5.1.x "
                "line for the repo's DINOv3 + SAM3 surface. Install: !pip install 'transformers~=5.1.0'"
            )
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        result["transformers_warning"] = f"Could not check transformers version: {exc}"

    optional_packages = {
        "open_clip": "open-clip-torch",
    }
    missing_deps = []
    for package_name, pip_name in optional_packages.items():
        try:
            import_module(package_name)
        except ImportError:
            missing_deps.append(pip_name)

    result["missing_deps"] = missing_deps
    if missing_deps:
        result["install_command"] = f"!pip install {' '.join(missing_deps)}"

    return result
