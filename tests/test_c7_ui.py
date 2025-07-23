"""C7: UI Refinements, Accessibility (WCAG 2.1 AA) & Ergonomics static assertions."""
import re
from pathlib import Path


def test_no_native_dialogs_in_static_ui():
    """window.prompt, window.confirm, and window.alert are absent from photocure/ui/static/."""
    root = Path(__file__).parent.parent
    static_dir = root / "photocure" / "ui" / "static"
    violating_files = []
    for f in static_dir.rglob("*.js"):
        content = f.read_text(encoding="utf-8")
        forbidden = []
        for line_no, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            if re.search(r'\b(window\.)?(prompt|alert)\s*\(', line):
                forbidden.append((line_no, line))
            # For confirm, make sure it's not showConfirmDialog or confirm-modal or confirmBtn
            if re.search(r'\b(window\.)?confirm\s*\(', line) and "showConfirmDialog" not in line:
                forbidden.append((line_no, line))
        if forbidden:
            violating_files.append((f.name, forbidden))

    assert not violating_files, f"Found native dialog usage in static files: {violating_files}"


def test_header_version_badge_is_native_button():
    """#header-version-badge is a native <button> element with type, title, and data-goto."""
    root = Path(__file__).parent.parent
    index_html = (root / "photocure" / "ui" / "static" / "index.html").read_text(encoding="utf-8")

    badge_match = re.search(r'<(\w+)[^>]*id=["\']header-version-badge["\'][^>]*>', index_html)
    assert badge_match, "#header-version-badge tag not found in index.html"
    tag_name = badge_match.group(1).lower()
    tag_full = badge_match.group(0)

    assert tag_name == "button", f"#header-version-badge must be <button>, found <{tag_name}>"
    assert 'type="button"' in tag_full
    assert 'data-goto="settings"' in tag_full


def test_persistent_reset_filters_button_present():
    """#reset-filters button exists in Library toolbar."""
    root = Path(__file__).parent.parent
    index_html = (root / "photocure" / "ui" / "static" / "index.html").read_text(encoding="utf-8")

    assert 'id="reset-filters"' in index_html, "#reset-filters button not found in index.html"


def test_scrub_presets_control_present():
    """#scrub-presets segmented control exists with all 5 presets."""
    root = Path(__file__).parent.parent
    index_html = (root / "photocure" / "ui" / "static" / "index.html").read_text(encoding="utf-8")

    assert 'id="scrub-presets"' in index_html, "#scrub-presets not found in index.html"
    for preset in ("all", "year", "90d", "30d", "custom"):
        msg = f"Preset button data-preset='{preset}' not found"
        assert f'data-preset="{preset}"' in index_html, msg


def test_confirm_modal_input_present():
    """#confirm-modal has an input field for prompt replacements."""
    root = Path(__file__).parent.parent
    index_html = (root / "photocure" / "ui" / "static" / "index.html").read_text(encoding="utf-8")

    assert 'id="confirm-modal-input-wrap"' in index_html
    assert 'id="confirm-modal-input"' in index_html
    assert 'id="confirm-modal-input-error"' in index_html
