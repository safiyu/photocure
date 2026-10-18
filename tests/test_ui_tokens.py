"""Logo-derived blue, pink and amber token system: values, contrast, and no leftovers."""
import re
from pathlib import Path

STATIC = Path(__file__).parent.parent / "photocure" / "ui" / "static"

ROLE_TOKENS = [
    "--primary", "--primary-hover", "--primary-ink",
    "--accent", "--accent-hover", "--accent-ink",
    "--commit", "--commit-hover", "--commit-ink",
    "--focus", "--link",
    "--primary-soft", "--accent-soft", "--commit-soft",
    "--grad-brand", "--canvas-bg", "--warn", "--warn-soft",
]
SCALES = [f"--{h}-{s}" for h in ("blue", "pink", "amber") for s in (50, 100, 400, 500, 600, 700)]


def _css() -> str:
    return (STATIC / "style.css").read_text(encoding="utf-8")


def _block(theme: str) -> dict:
    sel = r':root\[data-theme="dark"\]' if theme == "dark" else r":root"
    m = re.search(r"(?m)^" + sel + r"\s*\{(.*?)\n\}", _css(), re.S)
    assert m, f"{theme} :root block not found"
    body = re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.S)
    return {k: v.strip() for k, v in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", body)}


def _tok(theme: str, name: str) -> str:
    """Resolve a token to a hex value, following var() chains; dark falls back to light."""
    raw = _block(theme).get(name) or _block("light").get(name)
    assert raw, f"{name} not defined for {theme}"
    ref = re.fullmatch(r"var\((--[\w-]+)\)", raw)
    return _tok(theme, ref.group(1)) if ref else raw.lower()


def _lum(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    chans = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in chans]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_role_tokens_defined_in_both_themes():
    light, dark = _block("light"), _block("dark")
    for token in ROLE_TOKENS:
        assert token in light, f"{token} missing from :root"
        assert token in dark, f"{token} missing from dark theme"
    for token in SCALES + ["--navy"]:
        assert token in light, f"{token} missing from :root"


def test_removed_tokens_are_gone():
    css = _css()
    for stale in ("--accent-electric", "--duo-", "--ok-sage", "--danger-crimson",
                  "--viz-bar", "--tp-", "--accent-glow", "--accent-pill"):
        assert stale not in css, f"{stale} still present"


def test_role_values():
    for theme, stop in (("light", "500"), ("dark", "400")):
        for role, hue in (("primary", "blue"), ("accent", "pink"), ("commit", "amber")):
            assert _tok(theme, f"--{role}") == _tok(theme, f"--{hue}-{stop}"), (theme, role)
    assert _tok("light", "--navy") == "#0b1a33"
    assert _block("dark")["--canvas-bg"].lower() == "#0a0f1e"


def test_brand_fills_pass_aa_with_navy():
    for theme in ("light", "dark"):
        for role in ("primary", "accent", "commit"):
            ink = _tok(theme, f"--{role}-ink")
            assert ink == "#0b1a33", (theme, role, ink)
            for suffix in ("", "-hover"):
                ratio = _contrast(_tok(theme, f"--{role}{suffix}"), ink)
                assert ratio >= 4.5, f"{theme} --{role}{suffix} vs navy is {ratio:.2f}"


def test_link_and_warn_pass_aa_on_sheet():
    for theme, sheet in (("light", "#ffffff"), ("dark", "#14171f")):
        for token in ("--link", "--warn"):
            ratio = _contrast(_tok(theme, token), sheet)
            assert ratio >= 4.5, f"{theme} {token} on sheet is {ratio:.2f}"
        soft = _contrast(_tok(theme, "--warn"), _tok(theme, "--warn-soft"))
        assert soft >= 4.5, f"{theme} --warn on --warn-soft is {soft:.2f}"


def test_no_indigo_left():
    css = re.sub(r"(?m)^\s*--c-[\w-]+:[^;]*;", "", _css())
    for stale in ("#4f46e5", "#6366f1", "#8b5cf6", "#7c3aed", "139, 92, 246", "99, 102, 241", "79, 70, 229"):
        assert stale not in css.lower(), f"{stale} still present outside --c-* data colors"
