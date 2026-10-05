import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGO_PARTIAL = ROOT / "overrides" / "partials" / "logo.html"
FAVICON = ROOT / "docs" / "assets" / "favicon.svg"
EXTRA_CSS = ROOT / "docs" / "stylesheets" / "extra.css"
NODESTEP_CSS = ROOT / "src" / "nodestep_sandbox" / "static" / "nodestep.css"
MARK = 'd="M14 50V14L50 50V14"'
HEX_COLOR = re.compile(r"#[0-9a-fA-F]{3,8}\b")


def theme_settings() -> str:
    settings = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    return settings.split("theme:\n", 1)[1].split("\n\n", 1)[0]


def nodestep_colors() -> set[str]:
    pairs = re.findall(
        r"--nodestep-color-[a-z-]+: light-dark\((#[0-9a-f]{6}), (#[0-9a-f]{6})\);",
        NODESTEP_CSS.read_text(encoding="utf-8"),
    )
    return {color for pair in pairs for color in pair}


def header_logo(site: Path) -> str:
    html = (site / "index.html").read_text(encoding="utf-8")
    return html.split('class="md-header__button md-logo"', 1)[1].split("</a>", 1)[0]


def test_the_theme_uses_the_overrides_folder() -> None:
    assert "\n  custom_dir: overrides\n" in theme_settings()


def test_the_logo_partial_draws_the_mark_in_the_header_ink() -> None:
    partial = LOGO_PARTIAL.read_text(encoding="utf-8")
    assert partial.startswith("<svg ")
    assert 'aria-hidden="true"' in partial
    assert MARK in partial
    assert re.search(MARK + r' fill="none" stroke="currentColor"', partial)
    assert partial.count('fill="currentColor"') == 3
    assert partial.count('class="nodestep-logo-accent"') == 1
    assert not HEX_COLOR.findall(partial)


def test_the_logo_accent_uses_the_amber_variable() -> None:
    stylesheet = EXTRA_CSS.read_text(encoding="utf-8")
    assert (
        ".nodestep-logo-accent {\n  fill: var(--nodestep-accent-color);\n}"
        in stylesheet
    )
    for scheme in ("default", "slate"):
        block = re.search(
            r'\[data-md-color-scheme="' + scheme + r'"\]\s*\{([^}]*)\}', stylesheet
        )
        assert block
        assert "--nodestep-accent-color: " in block.group(1)


def test_the_built_header_shows_the_inline_logo(site: Path) -> None:
    logo = header_logo(site)
    assert "<svg " in logo
    assert MARK in logo
    assert "<img" not in logo


def test_the_favicon_is_the_logo_in_the_nodestep_design_colors() -> None:
    assert "\n  favicon: assets/favicon.svg\n" in theme_settings()
    favicon = FAVICON.read_text(encoding="utf-8")
    assert MARK in favicon
    assert "prefers-color-scheme: dark" in favicon
    assert set(HEX_COLOR.findall(favicon)) <= nodestep_colors()


def test_the_built_pages_link_the_svg_favicon(site: Path) -> None:
    html = (site / "index.html").read_text(encoding="utf-8")
    assert '<link rel="icon" href="assets/favicon.svg">' in html
    assert (site / "assets" / "favicon.svg").read_text(encoding="utf-8") == (
        FAVICON.read_text(encoding="utf-8")
    )
