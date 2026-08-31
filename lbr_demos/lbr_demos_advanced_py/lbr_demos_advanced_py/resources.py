from pathlib import Path


def package_resource(filename: str) -> Path:
    """Return an installed package data file with a clear failure message."""
    resource_path = Path(__file__).with_name(filename)
    if not resource_path.is_file():
        raise FileNotFoundError(
            f"Required package resource '{filename}' is missing: {resource_path}. "
            "Install the package with its data files included."
        )
    return resource_path
