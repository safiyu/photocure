"""C5: UI revamp with self-hosted typography, WCAG tokens, and About card."""
import json
import re
from pathlib import Path


def test_zero_external_urls():
    """index.html and style.css contain zero http/https URLs (except GitHub link)."""
    root = Path(__file__).parent.parent
    index_html = (root / "photocure/ui/static/index.html").read_text()
    style_css = (root / "photocure/ui/static/style.css").read_text()

    # Count external http(s):// URLs, excluding GitHub project links in comments/readme
    index_urls = re.findall(r'https?://(?!github\.com/)', index_html)
    style_urls = re.findall(r'https?://(?!github\.com/)', style_css)

    assert not index_urls, f"index.html has external URLs: {index_urls}"
    assert not style_urls, f"style.css has external URLs: {style_urls}"


def test_font_files_bundled():
    """Plus Jakarta Sans and JetBrains Mono .woff2 files exist."""
    root = Path(__file__).parent.parent
    fonts_dir = root / "photocure/ui/static/fonts"

    assert (fonts_dir / "plus-jakarta-sans-regular.woff2").exists()
    assert (fonts_dir / "plus-jakarta-sans-bold.woff2").exists()
    assert (fonts_dir / "jetbrains-mono-regular.woff2").exists()


def test_wcag_aa_color_tokens():
    """CSS defines WCAG AA token pairs for both light and dark themes."""
    root = Path(__file__).parent.parent
    style_css = (root / "photocure/ui/static/style.css").read_text()

    # Light theme
    assert "--table: #f8fafc;" in style_css, "Light --table token not found"
    assert "--sheet: #ffffff;" in style_css, "Light --sheet token not found"
    assert "--ink: #0f172a;" in style_css, "Light --ink token not found"
    assert "--mut: #475569;" in style_css, "Light --mut token not found"
    assert "--faint: #5f6e84;" in style_css, "Light --faint token not found"

    # Dark theme
    assert "--table: #0c0e12;" in style_css, "Dark --table token not found"
    assert "--sheet: #14171f;" in style_css, "Dark --sheet token not found"
    assert "--ink: #f1f3f7;" in style_css, "Dark --ink token not found"
    assert "--mut: #949eb2;" in style_css, "Dark --mut token not found"
    assert "--faint: #7e8a9f;" in style_css, "Dark --faint token not found"


def test_typography_tokens():
    """CSS defines font-display, font-body, and font-mono tokens."""
    root = Path(__file__).parent.parent
    style_css = (root / "photocure/ui/static/style.css").read_text()

    assert '--font-display: "Plus Jakarta Sans"' in style_css
    assert '--font-body: "Plus Jakarta Sans"' in style_css
    assert '--font-mono: "JetBrains Mono"' in style_css


def test_backdrop_filter_guards():
    """Backdrop filters wrapped in @supports and @media (prefers-reduced-motion)."""
    root = Path(__file__).parent.parent
    style_css = (root / "photocure/ui/static/style.css").read_text()

    assert "@supports (backdrop-filter: blur" in style_css
    assert "@media (prefers-reduced-motion: reduce)" in style_css


def test_version_badge_html_structure():
    """index.html has version badge with id and data-goto."""
    root = Path(__file__).parent.parent
    index_html = (root / "photocure/ui/static/index.html").read_text()

    assert 'id="header-version-badge"' in index_html
    assert 'data-goto="settings"' in index_html
    assert 'id="sidebar-version"' in index_html
    assert 'class="sidebar-footer"' in index_html


def test_licenses_endpoint_reachable():
    """Licenses link in settings points to /licenses route."""
    root = Path(__file__).parent.parent

    # Check that /licenses appears in about card rendering
    app_js = (root / "photocure/ui/static/app.js").read_text()
    assert "/licenses" in app_js


def test_notice_file_includes_font_licenses():
    """NOTICE file credits Plus Jakarta Sans and JetBrains Mono."""
    root = Path(__file__).parent.parent
    notice = (root / "NOTICE").read_text()

    assert "Plus Jakarta Sans" in notice
    assert "JetBrains Mono" in notice
    assert "SIL Open Font License" in notice


def test_about_api_implementation():
    """app.py has /api/about and /licenses endpoints."""
    root = Path(__file__).parent.parent
    app_py = (root / "photocure/ui/app.py").read_text()

    assert '@app.get("/api/about")' in app_py
    assert '@app.get("/licenses")' in app_py
    assert 'def get_about()' in app_py
    assert 'def get_licenses()' in app_py


def test_about_card_renders_models():
    """About card app.js code iterates MANIFEST.json models."""
    root = Path(__file__).parent.parent
    app_js = (root / "photocure/ui/static/app.js").read_text()

    assert "about.models" in app_js
    assert "Active models" in app_js


def test_manifest_format():
    """MANIFEST.json has correct structure with name, file, license."""
    root = Path(__file__).parent.parent
    manifest_path = root / "photocure/models/MANIFEST.json"

    manifest = json.loads(manifest_path.read_text())
    assert "models" in manifest
    assert isinstance(manifest["models"], list)

    for model in manifest["models"]:
        assert "file" in model
        assert "name" in model
        assert "license" in model


def test_theme_persistence_html_structure():
    """index.html has theme toggle and localStorage init script."""
    root = Path(__file__).parent.parent
    index_html = (root / "photocure/ui/static/index.html").read_text()

    # Theme toggle button
    assert 'id="theme-toggle"' in index_html

    # Script that initializes theme from localStorage or prefers-color-scheme
    assert "localStorage" in index_html
    assert "prefers-color-scheme" in index_html
    assert "documentElement.dataset.theme" in index_html


if __name__ == "__main__":
    test_zero_external_urls()
    test_font_files_bundled()
    test_wcag_aa_color_tokens()
    test_typography_tokens()
    test_backdrop_filter_guards()
    test_version_badge_html_structure()
    test_licenses_endpoint_reachable()
    test_notice_file_includes_font_licenses()
    test_about_api_implementation()
    test_about_card_renders_models()
    test_manifest_format()
    test_theme_persistence_html_structure()
    print("✓ All C5 tests passed")
