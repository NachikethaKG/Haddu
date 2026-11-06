"""Basic repository scaffolding and package structure tests."""

from pathlib import Path
import haddu


def test_package_version():
    assert haddu.__version__ == "0.1.0"


def test_project_structure():
    repo_root = Path(__file__).resolve().parent.parent
    assert (repo_root / "haddu" / "core").is_dir()
    assert (repo_root / "haddu" / "db").is_dir()
    assert (repo_root / "haddu" / "scripts").is_dir()
    assert (repo_root / "data").is_dir()
    assert (repo_root / "docker-compose.yml").is_file()
    assert (repo_root / "pyproject.toml").is_file()
    assert (repo_root / ".gitignore").is_file()
