"""The version is 1.1.0 and every place that states it agrees."""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from photocure import __version__

ROOT = Path(__file__).parent.parent
EXPECTED = "1.1.0"


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_package_version():
    assert __version__ == EXPECTED


def test_pyproject_matches():
    assert re.search(r'(?m)^version = "([^"]+)"', _read("pyproject.toml")).group(1) == __version__


def test_debian_control_and_build_script_match():
    assert re.search(r"(?m)^Version: (\S+)", _read("packaging/deb/control")).group(1) == __version__
    assert f'VERSION="{__version__}"' in _read("packaging/deb/build_deb.sh")


def test_metainfo_lists_this_release_first():
    releases = ET.parse(ROOT / "packaging/deb/photocure.metainfo.xml").getroot().find("releases")
    assert releases[0].get("version") == __version__


def test_ci_fallback_version_matches():
    ci = _read(".github/workflows/ci-cd.yml")
    assert f'echo "version={__version__}"' in ci and f'echo "tag=v{__version__}"' in ci


def test_web_app_reports_the_same_version():
    assert 'version="0.1.0"' not in _read("photocure/ui/app.py")
    html = _read("photocure/ui/static/index.html")
    assert f"style.css?v={__version__}" in html and f"app.js?v={__version__}" in html
    badge = re.search(r'id="sidebar-version"[^>]*>([^<]*)<', html).group(1)
    assert badge == __version__
