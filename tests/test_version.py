"""Version is variableized and kept in a single canonical place: photocure/__init__.__version__."""
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from fastapi.testclient import TestClient

from photocure import __version__
from photocure.ui.app import create_app

ROOT = Path(__file__).parent.parent


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_package_version_is_valid_semver():
    """Single source of truth is in photocure/__init__.__version__ and is valid semver."""
    assert re.match(r"^\d+\.\d+\.\d+", __version__)


def test_pyproject_uses_dynamic_versioning():
    """pyproject.toml does not hardcode version; it dynamically references photocure.__version__."""
    pyproject = _read("pyproject.toml")
    assert 'dynamic = ["version"]' in pyproject
    assert 'version = { attr = "photocure.__version__" }' in pyproject
    # Static version under [project] must not be present
    assert not re.search(r'(?m)^\[project\](?:(?!^\[)[\s\S])*?^version\s*=', pyproject)


def test_debian_packaging_is_variableized():
    """Debian control and build script extract version dynamically without hardcoded release fallbacks."""
    build_script = _read("packaging/deb/build_deb.sh")
    assert "photocure.__version__" in build_script
    # build_deb.sh must not contain a hardcoded fallback version like VERSION="1.2.1"
    assert not re.search(r'(?m)^\s*VERSION="[0-9]+\.[0-9]+\.[0-9]+"', build_script)


def test_metainfo_lists_valid_releases():
    """metainfo.xml lists historical releases and has a valid semver entry."""
    releases = ET.parse(ROOT / "packaging/deb/photocure.metainfo.xml").getroot().find("releases")
    assert len(releases) > 0
    first_ver = releases[0].get("version")
    assert re.match(r"^\d+\.\d+\.\d+", first_ver)


def test_ci_extracts_version_dynamically():
    """CI/CD workflow extracts version dynamically from photocure.__version__."""
    ci = _read(".github/workflows/ci-cd.yml")
    assert "photocure.__version__" in ci
    # Must not contain hardcoded version fallback assignments
    assert 'echo "version=1.' not in ci
    assert 'echo "tag=v1.' not in ci


def test_web_app_serves_single_source_version():
    """UI root endpoint and API dynamically inject and report photocure.__version__."""
    app = create_app()
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    html = res.text

    # Dynamic cache-busting queries match the current single-source version
    assert f"style.css?v={__version__}" in html
    assert f"app.js?v={__version__}" in html

    # Sidebar version badge matches the single-source version
    badge_match = re.search(r'id="sidebar-version"[^>]*>([^<]*)<', html)
    assert badge_match is not None
    assert badge_match.group(1) == __version__

    # Static index.html template uses placeholders instead of a hardcoded version string
    static_html = _read("photocure/ui/static/index.html")
    assert "{{VERSION}}" in static_html
    assert f"style.css?v={__version__}" not in static_html

    # /api/about endpoint reports __version__
    client.headers["X-PCUR-Token"] = app.state.token  # the about page needs the token
    about_res = client.get("/api/about")
    assert about_res.status_code == 200
    assert about_res.json()["version"] == __version__
