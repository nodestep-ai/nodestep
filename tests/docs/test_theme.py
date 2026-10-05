import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NAV_ITEM = ROOT / "overrides" / "partials" / "nav-item.html"
MATERIAL_INDEXES = '{% if "navigation.indexes" in features %}'
GROUP_INDEXES = '{% if "navigation.indexes" in features or level > 2 %}'
EXTRA_CSS = ROOT / "docs" / "stylesheets" / "extra.css"
NODESTEP_CSS = ROOT / "src" / "nodestep_sandbox" / "static" / "nodestep.css"
MATERIAL_TOKENS = {
    "--md-default-bg-color": "surface",
    "--md-default-fg-color": "text",
    "--md-default-fg-color--light": "text-secondary",
    "--md-default-fg-color--lighter": "text-muted",
    "--md-code-bg-color": "page",
    "--md-code-fg-color": "text",
    "--md-primary-fg-color": "surface",
    "--md-primary-fg-color--light": "sunk",
    "--md-primary-fg-color--dark": "sunk",
    "--md-primary-bg-color": "text",
    "--md-primary-bg-color--light": "text-secondary",
    "--md-accent-fg-color": "accent-text",
    "--md-accent-fg-color--transparent": "accent-wash",
    "--md-accent-bg-color": "on-accent",
    "--md-typeset-a-color": "accent-text",
    "--md-typeset-mark-color": "accent-wash",
    "--md-typeset-ins-color": "accent-wash",
    "--md-typeset-del-color": "danger-wash",
    "--md-typeset-table-color": "border",
    "--md-typeset-table-color--light": "sunk",
    "--md-warning-bg-color": "accent-wash",
    "--md-warning-fg-color": "text",
    "--md-footer-fg-color": "text",
    "--md-footer-fg-color--light": "text-secondary",
    "--md-footer-fg-color--lighter": "text-muted",
    "--md-footer-bg-color": "surface",
    "--md-footer-bg-color--dark": "surface",
    "--md-code-hl-color": "accent",
    "--md-code-hl-color--light": "accent-wash",
    "--md-code-hl-keyword-color": "accent-text",
    "--md-code-hl-string-color": "accent-text",
    "--md-code-hl-number-color": "text-secondary",
    "--md-code-hl-constant-color": "text",
    "--md-code-hl-special-color": "accent-text",
    "--md-code-hl-function-color": "text",
    "--md-code-hl-name-color": "text",
    "--md-code-hl-comment-color": "text-muted",
    "--md-code-hl-operator-color": "text-muted",
    "--md-code-hl-punctuation-color": "text-muted",
    "--md-code-hl-generic-color": "text-secondary",
    "--md-code-hl-variable-color": "text-secondary",
    "--nodestep-border-color": "border",
    "--nodestep-border-strong-color": "border-strong",
    "--nodestep-sunk-color": "sunk",
    "--nodestep-accent-color": "accent",
    "--nodestep-danger-color": "danger",
}
AMBER_HUES = (30, 50)
SHARED_VALUES = (
    "--nodestep-weight-strong",
    "--nodestep-focus-width",
    "--nodestep-focus-offset",
)
WEB_FONT_HOSTS = ("fonts.googleapis.com", "fonts.gstatic.com")
MINIMUM_FOCUS_CONTRAST = 3
SYMBOL_BADGES = {
    "parameter": "parameter",
    "type_parameter": "type parameter",
    "attribute": "attribute",
    "function": "function",
    "method": "method",
    "class": "class",
    "type_alias": "type alias",
    "type_variable": "type variable",
    "constant": "constant",
    "module": "module",
}


def nodestep_tokens() -> dict[str, tuple[str, str]]:
    found = re.findall(
        r"--nodestep-color-([a-z-]+): light-dark\((#[0-9a-f]{6}), (#[0-9a-f]{6})\);",
        NODESTEP_CSS.read_text(encoding="utf-8"),
    )
    return {name: (light, dark) for name, light, dark in found}


def scheme_rules(scheme: str) -> dict[str, str]:
    stylesheet = EXTRA_CSS.read_text(encoding="utf-8")
    block = re.search(
        r'\[data-md-color-scheme="' + scheme + r'"\]\s*\{([^}]*)\}', stylesheet
    )
    assert block, scheme
    return dict(re.findall(r"(--(?:md|nodestep)-[a-z-]+):\s*([^;]+);", block.group(1)))


def root_rules() -> dict[str, str]:
    stylesheet = EXTRA_CSS.read_text(encoding="utf-8")
    block = re.search(r"^:root \{([^}]*)\}", stylesheet, flags=re.MULTILINE)
    assert block
    return dict(re.findall(r"(--(?:md|nodestep)-[a-z-]+):\s*([^;]+);", block.group(1)))


def nodestep_values() -> dict[str, str]:
    return dict(
        re.findall(
            r"^\s*(--nodestep-[a-z-]+): ([^;]+);",
            NODESTEP_CSS.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
    )


def relative_luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(first: str, second: str) -> float:
    lighter, darker = sorted(
        (relative_luminance(first), relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def palette_entries() -> list[str]:
    settings = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    palette = settings.split("  palette:\n", 1)[1].split("  features:\n", 1)[0]
    return palette.split("    - media: ")[1:]


def test_the_palette_switch_toggles_light_and_dark_like_the_apps() -> None:
    entries = palette_entries()
    media = [entry.split("\n", 1)[0] for entry in entries]
    assert media == [
        '"(prefers-color-scheme: light)"',
        '"(prefers-color-scheme: dark)"',
    ]
    icons = [re.findall(r"\n +icon: (.+)", entry) for entry in entries]
    assert icons == [["nodestep/moon"], ["nodestep/sun"]]
    names = [re.findall(r"\n +name: (.+)", entry) for entry in entries]
    assert names == [["Switch to dark theme"], ["Switch to light theme"]]


def test_the_theme_icons_are_the_nodestep_design_moon_and_sun() -> None:
    icons = ROOT / "overrides" / ".icons" / "nodestep"
    moon = (icons / "moon.svg").read_text(encoding="utf-8")
    sun = (icons / "sun.svg").read_text(encoding="utf-8")
    assert 'd="M7.14 2.06A6 6 0 1 0 13.94 8.87 5 5 0 0 1 7.14 2.06z"' in moon
    assert '<circle cx="8" cy="8" r="3"' in sun
    assert 'd="M8 1v2M8 13v2M1 8h2M13 8h2' in sun
    for icon in (moon, sun):
        assert 'viewBox="0 0 16 16"' in icon
        shapes = re.findall(r"<(?:path|circle)\b[^>]*>", icon)
        assert shapes
        for shape in shapes:
            assert 'fill="none"' in shape
            assert 'stroke="currentColor"' in shape
            assert 'stroke-width="1.5"' in shape


def test_the_sidebar_has_no_site_title_on_wide_screens() -> None:
    stylesheet = EXTRA_CSS.read_text(encoding="utf-8")
    assert re.search(
        r"@media screen and \(min-width: 76\.25em\) \{\s*"
        r"\.md-nav--primary > \.md-nav__title \{\s*display: none;\s*\}",
        stylesheet,
    )


def test_the_palette_uses_custom_colors_in_both_schemes() -> None:
    entries = palette_entries()
    schemes = [re.findall(r"\n +scheme: (\w+)", entry) for entry in entries]
    assert schemes == [["default"], ["slate"]]
    for entry in entries:
        assert "      primary: custom\n" in entry
        assert "      accent: custom\n" in entry


def test_the_docs_use_the_nodestep_design_system_fonts() -> None:
    settings = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    theme = settings.split("theme:\n", 1)[1].split("\n\n", 1)[0]
    assert "\n  font: false\n" in theme
    shared = NODESTEP_CSS.read_text(encoding="utf-8")
    stylesheet = EXTRA_CSS.read_text(encoding="utf-8")
    root = re.search(r"^:root \{([^}]*)\}", stylesheet, flags=re.MULTILINE)
    assert root
    for material, nodestep in (("--md-text-font", "sans"), ("--md-code-font", "mono")):
        expected = re.search(rf"--nodestep-font-{nodestep}: ([^;]+);", shared)
        assert expected
        assert f"{material}: {expected.group(1)};" in root.group(1)


def test_the_built_pages_load_no_web_fonts(site: Path) -> None:
    offenders = [
        str(page.relative_to(site))
        for page in sorted(site.rglob("*.html"))
        if any(host in page.read_text(encoding="utf-8") for host in WEB_FONT_HOSTS)
    ]
    assert not offenders, offenders


def test_the_shared_sizes_are_the_nodestep_design_values() -> None:
    shared = nodestep_values()
    own = {
        name: value
        for name, value in root_rules().items()
        if name.startswith("--nodestep-")
    }
    assert set(SHARED_VALUES) <= set(own)
    assert own == {name: shared.get(name) for name in own}


def test_headings_use_the_nodestep_design_heading_weight() -> None:
    assert re.search(
        r"^h1,\nh2,\nh3,\nh4 \{[^}]*\n  font-weight: var\(--nodestep-weight-strong\);\n",
        NODESTEP_CSS.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    assert (
        ".md-typeset :is(h1, h2, h3, h4) {\n"
        "  font-weight: var(--nodestep-weight-strong);\n"
        "}"
    ) in EXTRA_CSS.read_text(encoding="utf-8")


def test_the_material_colors_are_the_nodestep_design_tokens() -> None:
    tokens = nodestep_tokens()
    for scheme, side in (("default", 0), ("slate", 1)):
        rules = scheme_rules(scheme)
        expected = {
            name: tokens[token][side] for name, token in MATERIAL_TOKENS.items()
        }
        assert {name: rules.get(name) for name in MATERIAL_TOKENS} == expected, scheme


def test_the_dark_scheme_hue_is_amber() -> None:
    hue = int(scheme_rules("slate")["--md-hue"])
    assert AMBER_HUES[0] <= hue <= AMBER_HUES[1]


def test_extra_css_has_no_colors_of_its_own() -> None:
    tokens = nodestep_tokens()
    stylesheet = EXTRA_CSS.read_text(encoding="utf-8")
    for scheme, side in (("default", 0), ("slate", 1)):
        rules = scheme_rules(scheme)
        allowed = {values[side] for values in tokens.values()}
        literals = {value for value in rules.values() if value.startswith("#")}
        assert literals <= allowed, scheme
    outside = re.sub(r'\[data-md-color-scheme="\w+"\]\s*\{[^}]*\}', "", stylesheet)
    assert not re.findall(r"#[0-9a-fA-F]{3,8}\b|\b(?:rgba?|hsla?)\(", outside)
    assert not re.findall(
        r":\s*[^;{}]*(?<![\w-])(?:black|silver|gray|white|maroon|red|purple|fuchsia|"
        r"green|lime|olive|yellow|navy|blue|teal|aqua|orange)(?![\w-])",
        outside,
    )


def test_focus_rings_use_the_bright_accent() -> None:
    stylesheet = EXTRA_CSS.read_text(encoding="utf-8")
    assert re.search(
        r":is\(:focus-visible, \.focus-visible\) \{\n"
        r"  outline-color: var\(--nodestep-accent-color\) !important;\n\}",
        stylesheet,
    )


def test_light_focus_rings_have_a_deep_amber_edge_like_nodestep_design() -> None:
    shared = nodestep_values()
    assert shared["--nodestep-focus-edge"] == (
        "light-dark(var(--nodestep-color-accent-text), transparent)"
    )
    stylesheet = EXTRA_CSS.read_text(encoding="utf-8")
    assert (
        '[data-md-color-scheme="default"] :is(:focus-visible, .focus-visible)'
        ":not(input, .md-search-result__link, .md-search-result__more > summary) {\n"
        "  outline-style: solid;\n"
        "  outline-width: var(--nodestep-focus-width);\n"
        "  outline-offset: var(--nodestep-focus-offset);\n"
        "  box-shadow:\n"
        "    0 0 0 var(--nodestep-focus-offset) var(--md-default-bg-color),\n"
        "    0 0 0 calc(var(--nodestep-focus-offset) + var(--nodestep-focus-width) + 1px)"
        " var(--md-accent-fg-color);\n"
        "}"
    ) in stylesheet
    rules = scheme_rules("default")
    edge = rules["--md-accent-fg-color"]
    assert edge == nodestep_tokens()["accent-text"][0]
    assert contrast(edge, rules["--md-default-bg-color"]) >= MINIMUM_FOCUS_CONTRAST


def test_only_the_light_scheme_adds_a_focus_edge() -> None:
    stylesheet = EXTRA_CSS.read_text(encoding="utf-8")
    assert not re.search(r'\[data-md-color-scheme="slate"\][^{]*focus', stylesheet)
    rules = re.findall(r"[^{}]*focus-visible[^{}]*\{[^}]*box-shadow", stylesheet)
    assert len(rules) == 1


def test_wide_tables_scroll_inside_their_own_box() -> None:
    stylesheet = EXTRA_CSS.read_text(encoding="utf-8")
    assert (
        "[data-md-color-scheme] .md-typeset .md-typeset__scrollwrap {\n"
        "  width: fit-content;\n"
        "  max-width: 100%;\n"
        "  margin: 1em 0;\n"
        "  overflow-x: auto;\n"
        "  background-color: var(--md-default-bg-color);\n"
        "  border: 1px solid var(--nodestep-border-color);\n"
        "  border-radius: var(--nodestep-radius);\n"
        "}"
    ) in stylesheet
    assert (
        "[data-md-color-scheme] .doc .md-typeset__scrollwrap {\n  width: auto;\n}"
    ) in stylesheet
    assert (
        "[data-md-color-scheme] .md-typeset .md-typeset__table {\n"
        "  margin-bottom: 0;\n"
        "  padding: 0;\n"
        "}"
    ) in stylesheet
    assert (
        ".md-typeset table:not([class]) {\n  font-size: 0.65rem;\n  border: 0;\n}"
    ) in stylesheet


def test_the_symbol_badges_still_show_full_words() -> None:
    stylesheet = EXTRA_CSS.read_text(encoding="utf-8")
    for symbol, word in SYMBOL_BADGES.items():
        assert (
            f'code.doc-symbol-{symbol}::after {{\n  content: "{word}";\n}}'
            in stylesheet
        )


def test_groups_inside_a_sidebar_section_open_with_their_index_page() -> None:
    material = pytest.importorskip("material")
    original = (
        Path(material.__file__).parent / "templates" / "partials" / "nav-item.html"
    ).read_text(encoding="utf-8")
    assert original.count(MATERIAL_INDEXES) == 1
    assert NAV_ITEM.read_text(encoding="utf-8") == original.replace(
        MATERIAL_INDEXES, GROUP_INDEXES
    )
    assert "navigation.indexes" not in (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
