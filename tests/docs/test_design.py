import json
import re
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
EXTRA_CSS = ROOT / "docs" / "stylesheets" / "extra.css"
HEADER = ROOT / "overrides" / "partials" / "header.html"
DOCS_PALETTES = ROOT / "tests" / "docs" / "fixtures" / "docs-palettes.json"
NODESTEP_CSS = ROOT / "src" / "nodestep_sandbox" / "static" / "nodestep.css"
BROWSER_REM_PX = 16
WIDE = "screen and (min-width: 76.25em)"
TABLET = "screen and (min-width: 60em)"
NARROW = "screen and (max-width: 40em)"
NODESTEP_NARROW = "(max-width: 40rem)"
SECTIONS_HIDDEN = NARROW
COMPACT = "(width < 60rem)"
MATERIAL_DRAWER = "screen and (max-width:76.234375em)"
DOCS_DRAWER = "screen and (max-width: 76.234375em)"
MATERIAL_TABLET = "screen and (min-width:60em)"
MATERIAL_COMPACT = "screen and (max-width:59.984375em)"
MINIMUM_TEXT_CONTRAST = 4.5
SCHEMES = ("default", "slate")
NEUTRAL_SYMBOLS = (
    "attribute",
    "type_alias",
    "type_parameter",
    "type_variable",
    "constant",
    "parameter",
)
CODE_ROLES = {
    "--md-code-hl-keyword-color": ".nodestep-token-keyword",
    "--md-code-hl-function-color": ".nodestep-token-function",
    "--md-code-hl-name-color": ".nodestep-token-identifier",
    "--md-code-hl-constant-color": ".nodestep-token-identifier",
    "--md-code-hl-string-color": ".nodestep-token-string",
    "--md-code-hl-special-color": ".nodestep-token-string",
    "--md-code-hl-number-color": ".nodestep-token-number",
    "--md-code-hl-comment-color": ".nodestep-token-comment",
    "--md-code-hl-operator-color": ".nodestep-token-operator",
    "--md-code-hl-punctuation-color": ".nodestep-token-operator",
}


class Rule(BaseModel):
    media: str
    selector: str
    declarations: dict[str, str]


def parse(text: str) -> list[Rule]:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    rules: list[Rule] = []
    media: list[str] = []
    position = 0
    while True:
        opening = text.find("{", position)
        closing = text.find("}", position)
        if closing != -1 and (opening == -1 or closing < opening):
            media.pop()
            position = closing + 1
            continue
        if opening == -1:
            return rules
        prelude = " ".join(text[position:opening].split())
        if prelude.startswith("@"):
            media.append(prelude.removeprefix("@media "))
            position = opening + 1
            continue
        end = text.index("}", opening)
        declarations = {}
        for declaration in text[opening + 1 : end].split(";"):
            name, _, value = declaration.partition(":")
            if name.strip():
                declarations[name.strip()] = " ".join(value.split())
        rules.append(
            Rule(media=" and ".join(media), selector=prelude, declarations=declarations)
        )
        position = end + 1


def matching(
    rules: list[Rule], source: str, selector: str, media: str
) -> dict[str, str]:
    found: dict[str, str] = {}
    for rule in rules:
        if rule.selector == selector and rule.media == media:
            found |= rule.declarations
    assert found, f"{source} has no rule for {selector!r} in {media!r}"
    return found


def declarations(path: Path, selector: str, media: str = "") -> dict[str, str]:
    return matching(parse(path.read_text(encoding="utf-8")), path.name, selector, media)


def docs(selector: str, media: str = "") -> dict[str, str]:
    return declarations(EXTRA_CSS, selector, media)


def nodestep(selector: str, media: str = "") -> dict[str, str]:
    return declarations(NODESTEP_CSS, selector, media)


@cache
def material_rules() -> tuple[Rule, ...]:
    return tuple(parse(material_stylesheet()))


def material(selector: str, media: str = "") -> dict[str, str]:
    return matching(list(material_rules()), "Material", selector, media)


def docs_rem_px() -> float:
    size = docs(":root")["font-size"]
    assert size.endswith("%")
    return BROWSER_REM_PX * float(size.removesuffix("%")) / 100


def px(value: str, variables: dict[str, str], rem: float) -> float:
    while match := re.fullmatch(r"var\((--[a-z0-9-]+)\)", value):
        value = variables[match.group(1)]
    if value == "0":
        return 0
    number = re.fullmatch(r"(-?[0-9.]+)(rem|px)", value)
    assert number, value
    return float(number.group(1)) * (rem if number.group(2) == "rem" else 1)


def nodestep_px(value: str) -> float:
    return px(value, nodestep(":root"), BROWSER_REM_PX)


def docs_px(value: str) -> float:
    return px(value, docs(":root"), docs_rem_px())


def resolve(value: str, scheme: str) -> str:
    tokens = docs(f'[data-md-color-scheme="{scheme}"]')
    while match := re.fullmatch(r"var\((--[a-z0-9-]+)\)", value):
        value = tokens[match.group(1)]
    return value


def nodestep_color(value: str, scheme: str) -> str:
    name = re.fullmatch(r"var\(--nodestep-color-([a-z-]+)\)", value)
    assert name, value
    pair = re.search(
        rf"--nodestep-color-{name.group(1)}: light-dark\((#[0-9a-f]{{6}}), (#[0-9a-f]{{6}})\);",
        NODESTEP_CSS.read_text(encoding="utf-8"),
    )
    assert pair
    return pair.group(1 if scheme == "default" else 2)


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


def material_stylesheet() -> str:
    material = pytest.importorskip("material")
    stylesheets = Path(material.__file__).parent / "templates/assets/stylesheets"
    return next(stylesheets.glob("main.*.min.css")).read_text(encoding="utf-8")


def test_the_search_field_has_the_docs_size_and_layout() -> None:
    search = nodestep(".nodestep-search")
    inner = material(".md-search__inner", MATERIAL_TABLET)
    assert nodestep_px(search["width"]) == docs_px(inner["width"])
    form = docs(".md-search__form", TABLET)
    assert nodestep_px(search["height"]) == docs_px(form["height"])
    assert search["border-radius"] == form["border-radius"] == "var(--nodestep-radius)"
    padding = material("[dir=ltr] .md-search__input", MATERIAL_TABLET)["padding-left"]
    assert [
        nodestep_px(part)
        for part in nodestep(".nodestep-search-input")["padding"].split()
    ] == [0, docs_px(padding)]
    assert docs_px(docs(".md-search__input", TABLET)["font-size"]) == nodestep_px(
        nodestep(".nodestep-search-input")["font-size"]
    )
    icon = nodestep(".nodestep-search-icon")
    assert nodestep_px(icon["top"]) == docs_px(
        material(".md-search__icon[for=__search]")["top"]
    )
    assert nodestep_px(icon["left"]) == docs_px(
        material("[dir=ltr] .md-search__icon[for=__search]")["left"]
    )
    assert nodestep_px(icon["width"]) == docs_px(material(".md-search__icon")["width"])
    assert icon["pointer-events"] == "none"
    clear = nodestep(".nodestep-search-clear")
    assert nodestep_px(clear["top"]) == docs_px(material(".md-search__options")["top"])
    assert nodestep_px(clear["right"]) == docs_px(
        material("[dir=ltr] .md-search__options")["right"]
    )


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_search_field_has_the_docs_colors(scheme: str) -> None:
    searching = "[data-md-toggle=search]:checked~.md-header"
    pairs = [
        (
            docs(".md-search__form", TABLET)["background-color"],
            nodestep(".nodestep-search")["background"],
        ),
        (
            docs(".md-search__form", TABLET)["border"].split()[-1],
            "var(--nodestep-color-border-strong)",
        ),
        (
            docs(".md-search__form:hover", TABLET)["border-color"],
            nodestep(".nodestep-search:hover")["border-color"],
        ),
        (
            docs(".md-search__form:focus-within", TABLET)["border-color"],
            nodestep(".nodestep-search:focus-within")["border-color"],
        ),
        (
            docs(".md-search__input::placeholder", TABLET)["color"],
            nodestep(".nodestep-search-input::placeholder")["color"],
        ),
        (
            material(".md-search__input+.md-search__icon", MATERIAL_TABLET)["color"],
            nodestep(".nodestep-search-icon")["color"],
        ),
        (
            material(
                f"{searching} .md-search__input+.md-search__icon", MATERIAL_TABLET
            )["color"],
            nodestep(".nodestep-search:focus-within .nodestep-search-icon")["color"],
        ),
        (
            material(".md-search__options>.md-icon")["color"],
            nodestep(".nodestep-search-clear")["color"],
        ),
    ]
    for docs_value, nodestep_value in pairs:
        assert resolve(docs_value, scheme) == nodestep_color(nodestep_value, scheme)
    assert nodestep(".nodestep-search")["border"] == "var(--nodestep-border-strong)"
    assert material(f"{searching} .md-search__input::placeholder", MATERIAL_TABLET) == {
        "color": "#0000"
    }
    assert nodestep(".nodestep-search-input:focus::placeholder") == {
        "color": "transparent"
    }
    assert material(
        f"{searching} .md-search__input:valid~.md-search__options>.md-icon:hover"
    ) == {"opacity": ".7"}
    assert nodestep(".nodestep-search-clear:hover") == {"opacity": "0.7"}


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_folded_search_is_the_docs_search_button(scheme: str) -> None:
    button = docs('.md-header__inner > .md-header__button[for="__search"]')
    folded = nodestep(".nodestep-search", COMPACT)
    icon = nodestep(".nodestep-search-icon", COMPACT)
    assert docs_px(button["width"]) == nodestep_px(folded["width"])
    assert docs_px(button["height"]) == nodestep_px(
        nodestep(".nodestep-search")["height"]
    )
    assert button["border-radius"] == nodestep(".nodestep-search-icon")["border-radius"]
    svg = docs('.md-header__inner > .md-header__button[for="__search"] svg')
    assert docs_px(svg["width"]) == nodestep_px(
        nodestep(".nodestep-search-icon svg", COMPACT)["width"]
    )
    hover = docs('.md-header__inner > .md-header__button[for="__search"]:hover')
    folded_hover = nodestep(".nodestep-search-icon:hover", COMPACT)
    for docs_value, nodestep_value in [
        (button["color"], icon["color"]),
        (hover["color"], folded_hover["color"]),
        (hover["background-color"], folded_hover["background"]),
    ]:
        assert resolve(docs_value, scheme) == nodestep_color(nodestep_value, scheme)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_opened_search_is_the_docs_search_on_a_phone(scheme: str) -> None:
    opened = nodestep(".nodestep-search:focus-within .nodestep-search-input", COMPACT)
    assert docs_px(material(".md-search__input", MATERIAL_COMPACT)["font-size"]) == (
        nodestep_px(opened["font-size"])
    )
    material_padding = material("[dir=ltr] .md-search__input")
    _, end, _, start = opened["padding"].split()
    assert nodestep_px(start) == docs_px(material_padding["padding-left"])
    assert nodestep_px(end) == docs_px(material_padding["padding-right"])
    gutter = nodestep_px(nodestep(":root", NODESTEP_NARROW)["--nodestep-gutter"])
    icon = nodestep(".nodestep-search:focus-within .nodestep-search-icon", COMPACT)
    assert icon["left"] == "var(--nodestep-gutter)"
    assert gutter == docs_px(
        material("[dir=ltr] .md-search__icon[for=__search]", MATERIAL_COMPACT)["left"]
    )
    clear = nodestep(".nodestep-search:focus-within .nodestep-search-clear", COMPACT)
    assert clear["right"] == "var(--nodestep-gutter)"
    assert gutter == docs_px(
        material("[dir=ltr] .md-search__options", MATERIAL_COMPACT)["right"]
    )
    light = material(
        ".md-search__input::placeholder,.md-search__input~.md-search__icon"
    )["color"]
    assert resolve(light, scheme) == nodestep_color(
        nodestep(".nodestep-search-input:focus::placeholder", COMPACT)["color"], scheme
    )
    focused_icon = nodestep(".nodestep-search:focus-within .nodestep-search-icon")
    assert resolve(light, scheme) == nodestep_color(focused_icon["color"], scheme)


def test_one_rem_is_the_same_at_every_width() -> None:
    assert docs(":root")["font-size"] == "125%"
    assert docs_rem_px() == 20


def test_the_header_is_the_nodestep_design_top_bar() -> None:
    bar = nodestep(".nodestep-topbar")
    header = docs(".md-header")
    assert docs_px(header["height"]) == nodestep_px(bar["min-height"])
    assert bar["background"] == "var(--nodestep-color-surface)"
    assert header["background-color"] == "var(--md-default-bg-color)"
    assert bar["border-bottom"] == "var(--nodestep-border)"
    assert header["border-bottom"] == "1px solid var(--nodestep-border-color)"
    assert header["box-shadow"] == "none"
    inner = docs(".md-header__inner")
    assert inner["max-width"] == "none"
    assert inner["height"] == "100%"
    assert docs_px(inner["padding-inline"]) == nodestep_px("var(--nodestep-gutter)")
    narrow = docs(".md-header__inner", NARROW)["padding-inline"]
    gutter = nodestep(":root", NODESTEP_NARROW)["--nodestep-gutter"]
    assert docs_px(narrow) == nodestep_px(gutter)
    assert NARROW.removeprefix("screen and ") == NODESTEP_NARROW.replace("rem", "em")


def test_the_header_ends_with_search_then_repository_then_theme_toggle() -> None:
    header = HEADER.read_text(encoding="utf-8")
    positions = [
        header.index('<ul class="nodestep-sections">'),
        header.index('{% include "partials/search.html" %}'),
        header.index('<div class="md-header__source">'),
        header.index('{% include "partials/palette.html" %}'),
    ]
    assert positions == sorted(positions)
    ordered = [
        rule.selector
        for rule in parse(EXTRA_CSS.read_text(encoding="utf-8"))
        if "order" in rule.declarations
    ]
    assert ordered == []


def test_the_header_controls_are_spaced_like_the_nodestep_design_top_bar() -> None:
    gap = nodestep_px(nodestep(".nodestep-topbar-end")["gap"])
    toggle = docs(".md-header__inner > .md-header__option")
    assert docs_px(toggle["margin-inline-start"]) == gap
    search = docs('.md-header__inner > .md-header__button[for="__search"]')
    assert search["margin"] == "0"
    assert docs(".md-header__option .md-header__button")["margin"] == "0"


def test_the_gaps_before_the_repository_link_and_the_theme_icon_match() -> None:
    source = docs(".md-header__inner > .md-header__source")
    toggle = docs(".md-header__inner > .md-header__option")
    button = docs(".md-header__option .md-header__button")
    assert docs_px(source["margin-inline-start"]) == docs_px(
        toggle["margin-inline-start"]
    ) + docs_px(button["padding"])


@pytest.mark.parametrize("part", ["", ":hover", " svg"])
def test_the_narrow_search_button_matches_the_theme_toggle(part: str) -> None:
    toggle = docs(f".md-header__option .md-header__button{part}")
    search = docs(f'.md-header__inner > .md-header__button[for="__search"]{part}')
    assert {name: search.get(name) for name in toggle} == toggle


def test_the_menu_button_starts_at_the_gutter() -> None:
    menu = docs('.md-header__inner > .md-header__button[for="__drawer"]')
    assert menu == {"margin-inline-start": "0"}


def test_the_theme_toggle_keeps_its_place_while_searching() -> None:
    assert (
        "[data-md-toggle=search]:checked~.md-header .md-header__option{max-width:0;"
        in material_stylesheet()
    )
    assert docs(
        '[data-md-toggle="search"]:checked ~ .md-header .md-header__option'
    ) == {"max-width": "none"}


def test_the_header_brand_is_the_nodestep_design_brand() -> None:
    brand = nodestep(".nodestep-brand")
    assert docs_px(docs(".md-header__title")["font-size"]) == nodestep_px(
        brand["font-size"]
    )
    topic = docs(".md-header__topic:first-child")
    assert topic["font-weight"] == brand["font-weight"]
    assert topic["letter-spacing"] == brand["letter-spacing"]
    mark = nodestep(".nodestep-brand-mark")
    logo = docs(".md-header__button.md-logo svg")
    assert docs_px(logo["width"]) == nodestep_px(mark["width"])
    assert docs_px(logo["height"]) == nodestep_px(mark["height"])
    gap = docs(".md-header__button.md-logo")["margin-inline-end"]
    assert docs_px(gap) == nodestep_px(brand["gap"])


def test_the_header_keeps_the_site_name_after_scrolling() -> None:
    assert docs(".md-header__topic + .md-header__topic")["display"] == "none"
    assert docs(".md-header__title--active .md-header__topic") == {
        "opacity": "1",
        "pointer-events": "auto",
        "transform": "none",
    }


def test_the_theme_toggle_is_the_nodestep_design_theme_button() -> None:
    button = nodestep(".nodestep-theme-button")
    toggle = docs(".md-header__option .md-header__button")
    assert docs_px(toggle["width"]) == nodestep_px(button["width"])
    assert docs_px(toggle["height"]) == nodestep_px(button["height"])
    assert toggle["border-radius"] == "var(--nodestep-radius)"
    assert button["color"] == "var(--nodestep-color-text-secondary)"
    assert toggle["color"] == "var(--md-default-fg-color--light)"
    icon = docs(".md-header__option .md-header__button svg")
    nodestep_icon = nodestep(".nodestep-theme-button svg")
    assert docs_px(icon["width"]) == nodestep_px(nodestep_icon["width"])
    assert docs_px(icon["height"]) == nodestep_px(nodestep_icon["height"])
    padding = (docs_px(toggle["width"]) - docs_px(icon["width"])) / 2
    assert docs_px(toggle["padding"]) == padding
    hover = docs(".md-header__option .md-header__button:hover")
    assert nodestep(".nodestep-theme-button:hover") == {
        "color": "var(--nodestep-color-text)",
        "background": "var(--nodestep-color-sunk)",
    }
    assert hover == {
        "color": "var(--md-default-fg-color)",
        "background-color": "var(--nodestep-sunk-color)",
        "opacity": "1",
    }


def test_the_search_field_is_a_nodestep_design_input() -> None:
    field = nodestep(".nodestep-input, .nodestep-select, .nodestep-textarea")
    form = docs(".md-search__form", TABLET)
    assert docs_px(form["height"]) == nodestep_px(field["min-height"])
    assert field["background-color"] == "var(--nodestep-color-surface)"
    assert form["background-color"] == "var(--md-default-bg-color)"
    assert field["border"] == "var(--nodestep-border-strong)"
    assert form["border"] == "1px solid var(--nodestep-border-strong-color)"
    assert form["border-radius"] == field["border-radius"] == "var(--nodestep-radius)"
    assert nodestep(
        ".nodestep-input:hover, .nodestep-select:hover, .nodestep-textarea:hover"
    ) == {"border-color": "var(--nodestep-color-text-muted)"}
    assert docs(".md-search__form:hover", TABLET) == {
        "background-color": "var(--md-default-bg-color)",
        "border-color": "var(--md-default-fg-color--lighter)",
    }
    assert docs(".md-search__form:focus-within", TABLET) == {
        "border-color": "var(--nodestep-accent-color)"
    }
    text = docs(".md-search__input", TABLET)
    assert docs_px(text["font-size"]) == nodestep_px(field["font-size"])
    assert docs(".md-search__input::placeholder", TABLET) == {
        "color": "var(--md-default-fg-color--lighter)"
    }


def test_the_repository_link_is_quiet_and_small() -> None:
    link = nodestep(".nodestep-nav a")
    source = docs(".md-header__source .md-source")
    assert docs_px(source["font-size"]) == nodestep_px(link["font-size"])
    assert link["color"] == "var(--nodestep-color-text-secondary)"
    assert source["color"] == "var(--md-default-fg-color--light)"
    assert docs(".md-header__source .md-source:hover") == {
        "color": "var(--md-default-fg-color)",
        "opacity": "1",
    }


def built_header(page: Path) -> str:
    html = page.read_text(encoding="utf-8")
    return html.split('<header class="md-header', 1)[1].split("</header>", 1)[0]


def header_sections(page: Path) -> list[tuple[str, str, str]]:
    header = built_header(page)
    assert header.count('<ul class="nodestep-sections">') == 1
    listed = header.split('<ul class="nodestep-sections">', 1)[1].split("</ul>", 1)[0]
    return re.findall(
        r'<a href="([^"]*)"( aria-current="true")?>\s*([^<]*?)\s*</a>', listed
    )


def checked_out(*names: str) -> list[str]:
    return [name for name in names if (ROOT.parent / name / "docs").is_dir()]


def header_repository(page: Path) -> tuple[str, str]:
    found = re.search(
        r'<a href="([^"]+)" title="[^"]*" class="md-source" data-md-component="source">'
        r'.*?<div class="md-source__repository">\s*([^<]*?)\s*</div>',
        built_header(page),
        re.DOTALL,
    )
    assert found
    return found.group(1), found.group(2)


def test_the_header_links_the_repository_of_the_section(site: Path) -> None:
    nodestep = ("https://github.com/nodestep-ai/nodestep", "nodestep-ai/nodestep")
    for page in ("index.html", "concepts/graphs/index.html", "404.html"):
        assert header_repository(site / page) == nodestep, page
    for name in checked_out("nodeartifact"):
        for page in (
            site / name / "index.html",
            site / name / "tracing" / "index.html",
        ):
            assert header_repository(page) == (
                f"https://github.com/nodestep-ai/{name}",
                f"nodestep-ai/{name}",
            ), page


def test_each_section_caches_the_facts_of_its_own_repository(site: Path) -> None:
    pages = [(site / "index.html", "nodestep-ai/nodestep")] + [
        (site / name / "index.html", f"nodestep-ai/{name}")
        for name in checked_out("nodeartifact")
    ]
    for page, repository in pages:
        head = page.read_text(encoding="utf-8").split("</head>", 1)[0]
        assert f'const repository = "{repository}";' in head, page


def test_material_draws_no_tabs_bar(site: Path) -> None:
    main = (ROOT / "overrides" / "main.html").read_text(encoding="utf-8")
    assert main.startswith('{% extends "base.html" %}\n')
    assert main.endswith("\n{% block tabs %}{% endblock %}\n")
    for page in (site / "index.html", site / "concepts" / "graphs" / "index.html"):
        assert 'class="md-tabs"' not in page.read_text(encoding="utf-8")


def page_title(page: Path) -> str:
    found = re.search(r"<title>([^<]*)</title>", page.read_text(encoding="utf-8"))
    assert found
    return found.group(1)


def test_pages_are_titled_page_first_then_the_docs_they_belong_to(
    site: Path,
) -> None:
    assert page_title(site / "index.html") == "nodestep docs"
    assert page_title(site / "install" / "index.html") == "Install · nodestep docs"
    assert (
        page_title(site / "reference" / "core" / "index.html") == "core · nodestep docs"
    )
    assert page_title(site / "404.html") == "Not found · nodestep docs"
    for name in checked_out("nodeartifact"):
        assert page_title(site / name / "index.html") == f"Overview · {name} docs"


def test_the_docs_share_the_theme_choice_with_the_apps(site: Path) -> None:
    script = (NODESTEP_CSS.parent / "nodestep-theme.js").read_text(encoding="utf-8")
    key = re.search(r'const key = "([^"]+)";', script)
    assert key
    assert re.search(r'stored === "light" \|\| stored === "dark"', script)
    head = (site / "index.html").read_text(encoding="utf-8").split("</head>", 1)[0]
    bridge = head.split("const key = ", 1)[1]
    assert bridge.startswith(f'"{key.group(1)}";')
    assert 'const schemes = { light: "default", dark: "slate" };' in bridge
    assert settings()["theme"]["palette"] == json.loads(
        DOCS_PALETTES.read_text(encoding="utf-8")
    )
    assert "__md_get = (name, ...options) =>" in bridge
    assert "__md_set = (name, ...options) =>" in bridge
    assert '__md_set("__palette", ' not in bridge
    assert "localStorage.setItem(key, " in bridge
    for listener in (
        'addEventListener("pageshow", follow);',
        'addEventListener("storage", follow);',
        'matchMedia("(prefers-color-scheme: dark)").addEventListener("change", follow);',
    ):
        assert listener in bridge
    assert head.index("__md_set=") < head.index("const key = ")
    body = (site / "index.html").read_text(encoding="utf-8").split("<body", 1)[1]
    assert 'var palette=__md_get("__palette")' in body


def test_the_sidebar_shows_only_the_current_section(site: Path) -> None:
    assert "navigation.tabs" in settings()["theme"]["features"]
    html = (site / "index.html").read_text(encoding="utf-8")
    assert 'class="md-nav md-nav--primary md-nav--lifted"' in html


def test_the_header_lists_the_sections_right_after_the_brand(site: Path) -> None:
    header = built_header(site / "index.html")
    positions = [
        header.index('class="md-header__title"'),
        header.index('<ul class="nodestep-sections">'),
        header.index('for="__search"'),
        header.index('class="md-header__source"'),
    ]
    assert positions == sorted(positions)
    siblings = checked_out("nodeartifact")
    assert header_sections(site / "index.html") == [
        (".", ' aria-current="true"', "nodestep"),
        *[(f"{name}/", "", name) for name in siblings],
    ]


def test_a_page_deep_in_the_docs_marks_the_docs_section(site: Path) -> None:
    siblings = checked_out("nodeartifact")
    assert header_sections(site / "concepts" / "graphs" / "index.html") == [
        ("../..", ' aria-current="true"', "nodestep"),
        *[(f"../../{name}/", "", name) for name in siblings],
    ]


@pytest.mark.parametrize("sibling", ["nodeartifact"])
def test_a_sibling_page_marks_its_own_section(site: Path, sibling: str) -> None:
    if not checked_out(sibling):
        pytest.skip(f"../{sibling} is not checked out")
    current = [
        label
        for _, marked, label in header_sections(site / sibling / "index.html")
        if marked
    ]
    assert current == [sibling]


def test_the_section_links_are_nodestep_design_nav_links() -> None:
    nav = nodestep(".nodestep-nav")
    sections = docs(".nodestep-sections")
    assert sections["display"] == nav["display"] == "flex"
    assert docs_px(sections["gap"]) == nodestep_px(nav["gap"])
    assert sections["margin"] == "0"
    assert sections["padding"] == "0"
    assert sections["list-style"] == "none"
    topbar_gap = nodestep(".nodestep-topbar")["gap"].split()[1]
    assert docs_px(sections["margin-inline-start"]) == nodestep_px(topbar_gap)
    assert sections["margin-inline-end"] == "auto"
    link = nodestep(".nodestep-nav a")
    section = docs(".nodestep-sections a")
    assert section["display"] == link["display"]
    assert docs_px(section["font-size"]) == nodestep_px(link["font-size"])
    assert [docs_px(part) for part in section["padding"].split()] == [
        nodestep_px(part) for part in link["padding"].split()
    ]
    assert link["color"] == "var(--nodestep-color-text-secondary)"
    assert section["color"] == "var(--md-default-fg-color--light)"
    assert section["text-decoration"] == link["text-decoration"] == "none"
    assert section["border-radius"] == link["border-radius"]
    assert nodestep(".nodestep-nav a:hover") == {
        "color": "var(--nodestep-color-text)",
        "background": "var(--nodestep-color-sunk)",
    }
    assert docs(".nodestep-sections a:hover") == {
        "color": "var(--md-default-fg-color)",
        "background-color": "var(--nodestep-sunk-color)",
    }


def test_the_current_section_is_marked_like_the_current_nav_link() -> None:
    assert nodestep('.nodestep-nav a[aria-current="page"]') == {
        "font-weight": "var(--nodestep-weight-strong)",
        "color": "var(--nodestep-color-text)",
        "background": "var(--nodestep-color-accent-wash)",
    }
    assert docs('.nodestep-sections a[aria-current="true"]') == {
        "font-weight": "var(--nodestep-weight-strong)",
        "color": "var(--md-default-fg-color)",
        "background-color": "var(--md-accent-fg-color--transparent)",
    }


def test_a_focused_section_link_has_the_nodestep_design_ring_in_both_themes() -> None:
    assert nodestep(":focus-visible")["outline"] == "var(--nodestep-focus-ring)"
    assert nodestep(":root")["--nodestep-focus-ring"] == (
        "var(--nodestep-focus-width) solid var(--nodestep-color-accent)"
    )
    assert docs(".nodestep-sections a:focus-visible") == {
        "outline-style": "solid",
        "outline-width": "var(--nodestep-focus-width)",
        "outline-offset": "var(--nodestep-focus-offset)",
    }


def test_the_brand_keeps_its_own_width_so_the_links_follow_it() -> None:
    assert docs(".md-header__title")["flex-grow"] == "0"
    assert docs('[dir="ltr"] .md-header__title') == {"margin-right": "0"}
    assert docs(".md-header__topic:first-child")["position"] == "relative"


def test_the_section_links_hide_where_they_do_not_fit() -> None:
    assert docs(".nodestep-sections", SECTIONS_HIDDEN) == {"display": "none"}
    assert docs(".md-header__title", SECTIONS_HIDDEN) == {"flex-grow": "1"}


def test_the_open_search_takes_the_place_of_the_section_links() -> None:
    searching = '[data-md-toggle="search"]:checked ~ .md-header'
    assert docs(f"{searching} .nodestep-sections") == {"display": "none"}
    assert docs(f"{searching} .md-header__title") == {"flex-grow": "1"}


@pytest.mark.parametrize("scheme", SCHEMES)
def test_section_labels_are_legible(scheme: str) -> None:
    surface = resolve("var(--md-default-bg-color)", scheme)
    label = resolve(docs(".nodestep-sections a")["color"], scheme)
    assert contrast(label, surface) >= MINIMUM_TEXT_CONTRAST
    current = docs('.nodestep-sections a[aria-current="true"]')
    text = resolve(current["color"], scheme)
    wash = resolve(current["background-color"], scheme)
    assert contrast(text, wash) >= MINIMUM_TEXT_CONTRAST


def test_body_text_uses_the_nodestep_design_base_size() -> None:
    typeset = docs(".md-typeset")
    assert docs_px(typeset["font-size"]) == nodestep_px(nodestep("body")["font-size"])
    assert typeset["line-height"] == "var(--nodestep-leading)"
    assert nodestep("html")["line-height"] == "var(--nodestep-leading)"


@pytest.mark.parametrize("level", ["h1", "h2", "h3"])
def test_headings_use_the_nodestep_design_sizes(level: str) -> None:
    heading = docs(f".md-typeset {level}")
    assert docs_px(heading["font-size"]) == nodestep_px(nodestep(level)["font-size"])
    assert heading["line-height"] == "var(--nodestep-leading-tight)"
    assert nodestep("h1, h2, h3, h4")["line-height"] == "var(--nodestep-leading-tight)"


def test_the_page_title_has_the_text_color() -> None:
    assert docs(".md-typeset h1")["color"] == "var(--md-default-fg-color)"


def test_the_sidebars_stick_where_they_start() -> None:
    assert docs(".md-main__inner")["margin-top"] == "0"
    assert docs(".md-sidebar")["top"] == docs(".md-header")["height"]
    page = nodestep(".nodestep-main")["padding-block"].split()[0]
    content = docs(".md-content__inner")["padding-top"]
    before = material_stylesheet().split(".md-content__inner:before{", 1)[1]
    spacer = re.search(r"height:([0-9.]+rem)", before.split("}", 1)[0])
    assert spacer
    assert docs_px(content) + docs_px(spacer.group(1)) == nodestep_px(page)


def test_the_folded_navigation_ends_at_the_bottom_of_the_screen() -> None:
    assert material(".md-sidebar--primary", MATERIAL_DRAWER)["position"] == "fixed"
    assert material(".md-sidebar--primary", MATERIAL_DRAWER)["height"] == "100%"
    header = docs(".md-header")["height"]
    assert docs(".md-sidebar")["top"] == header
    assert docs(".md-sidebar--primary", DOCS_DRAWER) == {
        "height": f"calc(100% - {header})"
    }


def test_navigation_labels_are_small_muted_capitals() -> None:
    label = docs(".md-nav--primary .md-nav__item--section > .md-nav__link", WIDE)
    assert label["color"] == "var(--md-default-fg-color--lighter)"
    assert label["text-transform"] == "uppercase"
    assert label["letter-spacing"] == "0.06em"
    assert label["font-weight"] == "var(--nodestep-weight-strong)"
    assert docs_px(label["font-size"]) == nodestep_px("var(--nodestep-text-xs)")
    section = docs(".md-nav--primary .md-nav__item--section", WIDE)
    assert docs_px(section["margin-top"]) == nodestep_px("var(--nodestep-space-5)")
    toc = docs(".md-sidebar--secondary .md-nav__title", TABLET)
    for name in ("color", "text-transform", "letter-spacing", "font-weight"):
        assert toc[name] == label[name], name
    assert toc["font-size"] == label["font-size"]


def test_navigation_items_hang_from_a_guide_line() -> None:
    for selector, media in (
        (".md-nav--primary .md-nav__item--section > .md-nav > .md-nav__list", WIDE),
        (".md-sidebar--secondary .md-nav--secondary .md-nav__list", TABLET),
    ):
        guide = docs(selector, media)
        assert guide["border-inline-start"] == "1px solid var(--nodestep-border-color)"
        assert docs_px(guide["margin-inline-start"]) > 0
    assert docs_px(docs(".md-nav", TABLET)["font-size"]) == nodestep_px(
        nodestep(".nodestep-panel-item")["font-size"]
    )


def test_a_section_of_groups_leaves_the_guide_lines_to_its_groups() -> None:
    lifted = (
        ".md-nav--lifted > .md-nav__list > .md-nav__item--active > .md-nav"
        " > .md-nav__list"
    )
    assert docs(f"{lifted}:has(> .md-nav__item--section)", WIDE) == {
        "border-inline-start": "0"
    }
    assert docs(f"{lifted} > .md-nav__item--section > .md-nav__link", WIDE) == {
        "margin-inline-start": "0",
        "padding-inline-start": "0",
    }


def test_table_of_contents_rows_keep_their_step_after_a_nested_group() -> None:
    assert "@media screen and (min-width:60em){.md-nav{margin-bottom:-.4rem}" in (
        material_stylesheet()
    )
    link = docs(".md-sidebar--secondary .md-nav--secondary .md-nav__link", TABLET)
    assert link["margin-top"] == "0"
    assert docs(".md-sidebar--secondary .md-nav--secondary .md-nav", TABLET) == {
        "margin-bottom": "0"
    }


def test_the_sticky_navigation_titles_have_no_glow() -> None:
    sticky = [
        rule
        for rule in parse(material_stylesheet())
        if rule.selector.endswith((".md-nav__title", ".md-nav__link"))
        and rule.declarations.get("position") == "sticky"
    ]
    assert {rule.selector for rule in sticky} == {
        ".md-nav--primary .md-nav__title",
        ".md-nav--secondary .md-nav__title",
        ".md-nav--lifted>.md-nav__list>.md-nav__item--active>.md-nav__link",
    }
    for rule in sticky:
        assert rule.declarations["background"] == "var(--md-default-bg-color)"
        assert "var(--md-default-bg-color)" in rule.declarations["box-shadow"]
        media = re.sub(r":(?=\S)", ": ", rule.media)
        selector = re.sub(r"\s*>\s*", " > ", rule.selector)
        assert docs(selector, media) in ({"box-shadow": "none"}, {"display": "none"})


def test_the_sidebar_has_no_label_for_the_current_section() -> None:
    label = ".md-nav--lifted > .md-nav__list > .md-nav__item--active > .md-nav__link"
    assert docs(label, WIDE) == {"display": "none"}


def settings() -> dict[str, Any]:
    config = pytest.importorskip("mkdocs.config")
    return config.load_config(str(ROOT / "mkdocs.yml"))


def nav_section(name: str) -> list[Any]:
    return next(item[name] for item in settings()["nav"] if name in item)


def nav_pages(items: Any) -> list[str]:
    if isinstance(items, str):
        return [items]
    if isinstance(items, dict):
        return [page for value in items.values() for page in nav_pages(value)]
    return [page for item in items for page in nav_pages(item)]


def test_the_site_has_a_section_per_project() -> None:
    assert [next(iter(item)) for item in settings()["nav"]] == [
        "nodestep",
        "nodeartifact",
    ]


def test_the_header_brand_says_docs(site: Path) -> None:
    topic = built_header(site / "index.html").split('class="md-header__topic"', 1)[1]
    assert re.search(r'<span class="md-ellipsis">\s*docs\s*</span>', topic)


def test_the_sibling_sections_have_groups_like_the_nodestep_section() -> None:
    def groups(name: str) -> list[str]:
        return [next(iter(item)) for item in nav_section(name)]

    assert groups("nodeartifact") == ["Get started", "Guides", "Reference"]


def test_the_sibling_sections_list_their_pages() -> None:
    assert nav_pages(nav_section("nodeartifact")) == [
        "nodeartifact/index.md",
        "nodeartifact/tracing.md",
        "nodeartifact/ui.md",
        "nodeartifact/server.md",
    ]


@pytest.mark.parametrize("sibling", ["nodeartifact"])
def test_every_page_of_a_checked_out_sibling_is_in_its_section(sibling: str) -> None:
    folder = ROOT.parent / sibling / "docs"
    if not folder.is_dir():
        pytest.skip(f"../{sibling} is not checked out")
    pages = sorted(
        f"{sibling}/{page.relative_to(folder).as_posix()}"
        for page in folder.rglob("*.md")
    )
    assert sorted(nav_pages(nav_section(sibling))) == pages


def test_every_navigation_link_sits_under_a_label() -> None:
    groups = {
        label: nav_pages(entries)
        for item in nav_section("nodestep")
        for label, entries in item.items()
    }
    assert list(groups) == [
        "Get started",
        "Core concepts",
        "Capabilities",
        "How-to",
        "API reference",
        "Changelog",
    ]
    assert groups["Get started"] == ["index.md", "install.md", "quickstart.md"]
    assert groups["Core concepts"] == [
        "concepts/graphs.md",
        "concepts/state.md",
        "concepts/nodes.md",
        "concepts/execution.md",
    ]
    assert groups["Capabilities"] == [
        "concepts/streaming.md",
        "concepts/persistence.md",
        "concepts/interrupts.md",
        "concepts/agents.md",
        "concepts/middleware.md",
        "concepts/sub-agents.md",
        "concepts/workspace.md",
    ]
    assert groups["How-to"] == [
        "guides/branching.md",
        "guides/fan-out.md",
        "guides/approval.md",
        "guides/chat-models.md",
        "guides/testing.md",
        "guides/tracing.md",
        "guides/sandbox.md",
        "examples.md",
    ]
    assert all(page.startswith("reference/") for page in groups["API reference"])
    assert groups["Changelog"] == ["changelog.md"]


NESTED_GROUP = (
    ".md-nav--primary .md-nav__item--section"
    " .md-nav__item--nested:not(.md-nav__item--section)"
)
GROUP_ROW = ".md-nav--primary .md-nav__item--section .md-nav__container"


def test_the_primary_navigation_has_no_negative_bottom_margins() -> None:
    assert "@media screen and (min-width:76.25em){.md-nav{margin-bottom:-.4rem;" in (
        material_stylesheet()
    )
    assert docs(".md-sidebar--primary .md-nav", WIDE) == {"margin-bottom": "0"}


def test_package_groups_hang_from_their_own_guide_line() -> None:
    section_link = docs(
        ".md-nav--primary .md-nav__item--section > .md-nav > .md-nav__list"
        " > .md-nav__item > .md-nav__link",
        WIDE,
    )
    guide = docs(f"{NESTED_GROUP} > .md-nav > .md-nav__list", WIDE)
    assert guide == {
        "margin-inline-start": f"calc({section_link['padding-inline-start']} - 1px)",
        "padding": "0",
        "border-inline-start": "1px solid var(--nodestep-border-color)",
    }
    link = docs(
        f"{NESTED_GROUP} > .md-nav > .md-nav__list > .md-nav__item > .md-nav__link",
        WIDE,
    )
    assert link == section_link
    assert docs(f"{NESTED_GROUP} > .md-nav", WIDE) == {
        "grid-template-rows": "minmax(0, 0fr)"
    }
    assert docs(f"{NESTED_GROUP} > .md-nav__toggle:checked ~ .md-nav", WIDE) == {
        "grid-template-rows": "minmax(0, 1fr)"
    }


def test_a_current_package_page_is_marked_on_its_whole_row() -> None:
    assert docs(f"{GROUP_ROW} > .md-nav__link", WIDE) == {"margin": "0", "padding": "0"}
    assert docs(f"{GROUP_ROW}:has(> .md-nav__link--active)", WIDE) == {
        "box-shadow": "inset 3px 0 0 var(--nodestep-accent-color)"
    }
    assert docs(f"{GROUP_ROW} > .md-nav__link--active", WIDE) == {"box-shadow": "none"}


def test_the_current_page_is_marked_like_a_current_panel_item() -> None:
    current = nodestep(
        '.nodestep-panel-item:is([aria-current="true"], [aria-current="page"])'
    )
    assert current["box-shadow"] == "inset 3px 0 0 var(--nodestep-color-accent)"
    for selector, media in (
        (".md-nav--primary .md-nav__item--section .md-nav__link--active", WIDE),
        (".md-sidebar--secondary .md-nav__link--active", TABLET),
    ):
        assert docs(selector, media) == {
            "color": "var(--md-typeset-a-color)",
            "box-shadow": "inset 3px 0 0 var(--nodestep-accent-color)",
        }


def test_scrollbars_stay_neutral() -> None:
    material = material_stylesheet()
    amber = {
        " ".join(re.sub(r"\s*>\s*", " > ", selector).split())
        for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", material)
        if re.search(
            r"scrollbar-color:var\(--md-accent-fg-color\)"
            r"|background-color:var\(--md-accent-fg-color\)",
            body,
        )
        and "scrollbar" in selectors + body
        for selector in selectors.split(",")
    }
    assert amber
    neutral: set[str] = set()
    for rule in parse(EXTRA_CSS.read_text(encoding="utf-8")):
        values = set(rule.declarations.values())
        if values & {
            "var(--nodestep-border-strong-color) transparent",
            "var(--nodestep-border-strong-color)",
        }:
            neutral |= {part.strip() for part in rule.selector.split(",")}
    assert amber <= neutral
    assert docs(".md-sidebar__scrollwrap")["scrollbar-color"] == (
        "var(--nodestep-border-strong-color) transparent"
    )
    assert nodestep("html")["scrollbar-color"] == (
        "var(--nodestep-color-border-strong) transparent"
    )


def test_code_blocks_are_nodestep_design_code_frames() -> None:
    frame = nodestep(".nodestep-code")
    block = docs(".md-typeset pre > code")
    assert frame["background"] == "var(--nodestep-color-page)"
    assert block["background-color"] == "var(--md-code-bg-color)"
    assert frame["border"] == "var(--nodestep-border)"
    assert block["border"] == "1px solid var(--nodestep-border-color)"
    assert block["border-radius"] == frame["border-radius"] == "var(--nodestep-radius)"
    for scheme in SCHEMES:
        assert resolve("var(--md-code-bg-color)", scheme) == nodestep_color(
            frame["background"], scheme
        )


def test_the_copy_button_is_centered_on_the_first_code_line() -> None:
    material = material_stylesheet()
    assert ".md-typeset code{" in material
    assert "font-size:.85em" in material.split(".md-typeset code{", 1)[1].split("}")[0]
    code = material.split(".md-typeset pre>code{", 1)[1].split("}")[0]
    assert "padding:.7720588235em 1.1764705882em" in code
    assert ".md-typeset pre{display:flow-root;line-height:1.4;" in material
    nav = material.split(".md-code__nav{", 1)[1].split("}")[0]
    assert "padding:.2rem" in nav
    assert "position:absolute" in nav
    assert "height:1.5em" in material.split(".md-code__button{", 1)[1].split("}")[0]
    assert docs(".md-typeset pre > code")["border"].startswith("1px ")
    assert docs(".md-typeset .md-code__nav") == {
        "top": "calc(1px + 0.85em * (0.7720588235 + 1.4 / 2) - (0.4rem + 1.5em) / 2)"
    }


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("variable", list(CODE_ROLES))
def test_syntax_colors_follow_the_nodestep_design_token_roles(
    scheme: str, variable: str
) -> None:
    role = nodestep(CODE_ROLES[variable])
    assert resolve(f"var({variable})", scheme) == nodestep_color(role["color"], scheme)


def test_keywords_and_function_names_are_strong_and_comments_italic() -> None:
    assert (
        nodestep(".nodestep-token-keyword")["font-weight"]
        == "var(--nodestep-weight-strong)"
    )
    assert (
        nodestep(".nodestep-token-function")["font-weight"]
        == "var(--nodestep-weight-strong)"
    )
    assert nodestep(".nodestep-token-comment")["font-style"] == "italic"
    assert docs(
        ".md-typeset .highlight :is(.k, .kc, .kd, .kn, .kp, .kr, .kt, .ow)"
    ) == {
        "color": "var(--md-code-hl-keyword-color)",
        "font-weight": "var(--nodestep-weight-strong)",
    }
    assert docs(".md-typeset .highlight :is(.nf, .fm, .nc)") == {
        "font-weight": "var(--nodestep-weight-strong)"
    }
    assert docs(".md-typeset .highlight :is(.c, .c1, .ch, .cm, .cs, .sd)") == {
        "font-style": "italic"
    }


@pytest.mark.parametrize("scheme", SCHEMES)
def test_inline_code_stands_apart_from_body_text(scheme: str) -> None:
    code = docs(".md-typeset :not(pre) > code:not(.doc-symbol)")
    assert code["background-color"] == "var(--nodestep-sunk-color)"
    assert code["border"] == "1px solid var(--nodestep-border-color)"
    assert code["border-radius"] == "var(--nodestep-radius-small)"
    text = resolve(code["color"], scheme)
    sunk = resolve(code["background-color"], scheme)
    surface = resolve("var(--md-default-bg-color)", scheme)
    assert text == resolve("var(--md-default-fg-color)", scheme)
    assert sunk != surface
    assert contrast(text, sunk) >= MINIMUM_TEXT_CONTRAST
    linked = docs(".md-typeset a > code:not(.doc-symbol)")
    link = resolve(linked["color"], scheme)
    assert link == resolve("var(--md-typeset-a-color)", scheme)
    assert link != text
    assert contrast(link, sunk) >= MINIMUM_TEXT_CONTRAST


def test_links_are_underlined_like_nodestep_design_links() -> None:
    anchor = nodestep("a")
    link = docs(".md-typeset a:not(.headerlink, pre a)")
    assert link["text-decoration-line"] == "underline"
    assert link["text-decoration-thickness"] == anchor["text-decoration-thickness"]
    assert link["text-underline-offset"] == anchor["text-underline-offset"]
    hover = docs(".md-typeset a:not(.headerlink, pre a):hover")
    assert hover == nodestep("a:hover")


def test_tables_are_nodestep_design_tables() -> None:
    wrap = docs("[data-md-color-scheme] .md-typeset .md-typeset__scrollwrap")
    assert nodestep(".nodestep-table-wrap")["border"] == "var(--nodestep-border)"
    assert wrap["border"] == "1px solid var(--nodestep-border-color)"
    assert wrap["border-radius"] == "var(--nodestep-radius)"
    assert wrap["background-color"] == "var(--md-default-bg-color)"
    assert wrap["overflow-x"] == "auto"
    table = docs(".md-typeset table:not([class])")
    assert docs_px(table["font-size"]) == nodestep_px(
        nodestep(".nodestep-table")["font-size"]
    )
    assert table["border"] == "0"
    cell = docs(".md-typeset table:not([class]) :is(th, td)")
    nodestep_cell = nodestep(".nodestep-table th, .nodestep-table td")
    assert [docs_px(part) for part in cell["padding"].split()] == [
        nodestep_px(part) for part in nodestep_cell["padding"].split()
    ]
    assert cell["border-top"] == "0"
    assert cell["border-bottom"] == "1px solid var(--nodestep-border-color)"
    head = docs(".md-typeset table:not([class]) th")
    nodestep_head = nodestep(".nodestep-table thead th")
    assert nodestep_head["background"] == "var(--nodestep-color-sunk)"
    assert head["background-color"] == "var(--nodestep-sunk-color)"
    assert nodestep_head["color"] == "var(--nodestep-color-text-secondary)"
    assert head["color"] == "var(--md-default-fg-color--light)"
    assert head["font-weight"] == "var(--nodestep-weight-strong)"
    assert docs(".md-typeset table:not([class]) tbody tr:last-child > *") == {
        "border-bottom": "0"
    }


def test_notes_are_nodestep_design_messages() -> None:
    message = nodestep(".nodestep-message")
    note = docs("[data-md-color-scheme] .md-typeset :is(.admonition, details)")
    assert docs_px(note["font-size"]) == nodestep_px(message["font-size"])
    assert [docs_px(part) for part in note["padding"].split()] == [
        nodestep_px(part) for part in message["padding"].split()
    ]
    assert note["border"] == "0"
    assert (
        note["border-inline-start"] == "3px solid var(--nodestep-border-strong-color)"
    )
    assert note["border-radius"] == "0 var(--nodestep-radius) var(--nodestep-radius) 0"
    assert note["background-color"] == "var(--nodestep-sunk-color)"
    assert note["box-shadow"] == "none"
    assert nodestep(".nodestep-message-info") == {
        "--nodestep-message-color": "var(--nodestep-color-border-strong)",
        "--nodestep-message-wash": "var(--nodestep-color-sunk)",
    }
    assert nodestep(".nodestep-message-warning") == {
        "--nodestep-message-color": "var(--nodestep-color-accent)",
        "--nodestep-message-wash": "var(--nodestep-color-accent-wash)",
    }
    warning = docs(
        "[data-md-color-scheme] .md-typeset :is(.admonition, details)"
        ":is(.warning, .caution, .attention)"
    )
    assert warning == {
        "background-color": "var(--md-accent-fg-color--transparent)",
        "border-inline-start-color": "var(--nodestep-accent-color)",
    }
    title = docs(
        "[data-md-color-scheme] .md-typeset :is(.admonition, details)"
        " > :is(.admonition-title, summary)"
    )
    assert title["background-color"] == "transparent"
    assert title["font-weight"] == "var(--nodestep-weight-strong)"
    assert docs(
        "[data-md-color-scheme] .md-typeset :is(.admonition, details)"
        " > :is(.admonition-title, summary)::before"
    ) == {"display": "none"}


def test_the_footer_has_no_generator_note(site: Path) -> None:
    config = pytest.importorskip("mkdocs.config")
    assert config.load_config(str(ROOT / "mkdocs.yml"))["extra"]["generator"] is False
    html = (site / "index.html").read_text(encoding="utf-8")
    assert "Made with" not in html
    assert "squidfunk.github.io/mkdocs-material" not in html


def test_the_footer_is_a_quiet_surface_with_a_top_border() -> None:
    footer = docs(".md-footer")
    assert footer["border-top"] == "1px solid var(--nodestep-border-color)"
    direction = docs(".md-footer__direction")
    assert docs_px(direction["font-size"]) == nodestep_px("var(--nodestep-text-xs)")
    assert direction["opacity"] == "1"
    assert direction["color"] == "var(--md-footer-fg-color--lighter)"
    title = docs(".md-footer__title")
    assert docs_px(title["font-size"]) == nodestep_px("var(--nodestep-text-base)")
    assert title["font-weight"] == "var(--nodestep-weight-strong)"
    assert docs(".md-footer__link:is(:focus, :hover)") == {
        "color": "var(--md-typeset-a-color)",
        "opacity": "1",
    }


def test_the_footer_stays_at_the_bottom_of_the_window() -> None:
    footer = docs(".md-footer")
    assert footer["position"] == "sticky"
    assert footer["bottom"] == "0"
    assert 0 < int(footer["z-index"]) < int(material(".md-header")["z-index"])
    inner = docs(".md-footer__inner")
    assert inner["height"] == docs(".md-header")["height"]
    assert inner["padding-block"] == "0"
    assert docs(".md-footer__link") == {"align-items": "center", "margin-block": "0"}
    assert docs(".md-footer__title") == {
        "font-size": "0.75rem",
        "font-weight": "var(--nodestep-weight-strong)",
        "margin-bottom": "0",
    }
    assert docs(".md-footer-meta") == {"display": "none"}


def test_keyboard_focus_scrolls_clear_of_the_footer() -> None:
    footer = docs(".md-footer__inner")["height"]
    footer_border = docs(".md-footer")["border-top"].split()[0]
    assert docs("html") == {
        "scroll-padding-bottom": f"calc({footer} + {footer_border})"
    }


@pytest.mark.parametrize(("side", "media"), [("primary", WIDE), ("secondary", TABLET)])
def test_the_sidebars_scroll_between_the_header_and_the_footer(
    side: str, media: str
) -> None:
    header = docs(".md-header")["height"]
    footer = docs(".md-footer__inner")["height"]
    footer_border = docs(".md-footer")["border-top"].split()[0]
    assert docs(f".md-sidebar--{side}", media) == {"padding-block": "0"}
    assert docs(f".md-sidebar--{side} .md-sidebar__scrollwrap", media) == {
        "max-height": f"calc(100vh - {header} - {footer} - {footer_border})"
    }
    padding = material(".md-sidebar")["padding"].split()[0]
    assert docs(f".md-sidebar--{side} .md-sidebar__inner", media) == {
        "padding-block": padding
    }


SYMBOL_COLORS = {
    "class": "var(--md-typeset-a-color)",
    "function": "var(--nodestep-danger-color)",
    "method": "var(--md-default-fg-color--light)",
    "module": "var(--md-default-fg-color)",
    **dict.fromkeys(NEUTRAL_SYMBOLS, "var(--md-default-fg-color--lighter)"),
}
SYMBOL_SCOPE = "[data-md-color-scheme] :is(.md-typeset, .md-nav) code"
SYMBOL = f"{SYMBOL_SCOPE}.doc-symbol"
PROPERTY_LABEL = "[data-md-color-scheme] .md-typeset .doc-label > code"
PLAIN_TEXT = {
    "padding": "0",
    "font-family": "var(--md-text-font-family)",
    "font-size": "0.6rem",
    "font-weight": "normal",
    "background-color": "transparent",
    "border": "0",
    "border-radius": "0",
}


def symbol_style(kind: str) -> dict[str, str]:
    style = docs(SYMBOL)
    for rule in parse(EXTRA_CSS.read_text(encoding="utf-8")):
        if not rule.media and rule.selector.startswith(SYMBOL_SCOPE):
            kinds = re.findall(r"\.doc-symbol-([a-z_]+)", rule.selector)
            if kind in kinds:
                style = style | rule.declarations
    return style


def test_symbol_kinds_are_plain_text_labels() -> None:
    label = docs(SYMBOL)
    assert {name: label.get(name) for name in PLAIN_TEXT} == PLAIN_TEXT
    assert docs_px(label["font-size"]) == nodestep_px("var(--nodestep-text-xs)")
    assert "vertical-align" not in label
    for kind in SYMBOL_COLORS:
        assert symbol_style(kind).keys() - {"color"} == label.keys() - {"color"}


def test_each_symbol_kind_has_its_own_color_from_the_palette() -> None:
    for kind, color in SYMBOL_COLORS.items():
        assert symbol_style(kind)["color"] == color, kind
    kinds = ("class", "method", "attribute", "function", "module")
    assert len({SYMBOL_COLORS[kind] for kind in kinds}) == len(kinds)


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("kind", list(SYMBOL_COLORS))
def test_symbol_labels_are_legible(scheme: str, kind: str) -> None:
    text = resolve(symbol_style(kind)["color"], scheme)
    for background in ("var(--md-default-bg-color)", "var(--nodestep-sunk-color)"):
        assert contrast(text, resolve(background, scheme)) >= MINIMUM_TEXT_CONTRAST


def test_property_labels_are_plain_muted_text_like_the_symbol_labels() -> None:
    label = docs(PROPERTY_LABEL)
    assert {name: label.get(name) for name in PLAIN_TEXT} == PLAIN_TEXT
    assert label["color"] == "var(--md-default-fg-color--lighter)"
    assert docs("[data-md-color-scheme] .md-typeset .doc-label") == {"opacity": "1"}


def test_symbol_labels_win_over_the_material_navigation_code_colors() -> None:
    assert ".md-nav__link[href]:hover code{background-color:" in material_stylesheet()
    assert ".md-nav__item .md-nav__link--active code{color:" in material_stylesheet()
    assert SYMBOL.startswith("[data-md-color-scheme] :is(.md-typeset, .md-nav) ")


def test_property_labels_win_over_the_inline_code_frame() -> None:
    def specificity(selector: str) -> tuple[int, int]:
        classes = len(re.findall(r"\.[\w-]+|\[[^\]]+\]|:not\(\.", selector))
        elements = len(re.findall(r"(?:^|[\s>(])(?:code|pre)\b", selector))
        return classes, elements

    assert specificity(PROPERTY_LABEL) > specificity(
        ".md-typeset :not(pre) > code:not(.doc-symbol)"
    )


def test_page_titles_show_no_symbol_kind_label() -> None:
    assert docs(".md-typeset h1 code.doc-symbol") == {"display": "none"}


def test_parameters_returns_and_raises_are_compact_lists() -> None:
    options = settings()["plugins"]["mkdocstrings"].config["handlers"]["python"][
        "options"
    ]
    assert options["docstring_section_style"] == "list"


def test_docstring_sections_are_small_labels_over_tight_lists() -> None:
    title = docs(".md-typeset .doc-section-title")
    assert title["font-size"] == "0.7rem"
    assert title["color"] == "var(--md-default-fg-color--light)"
    assert docs(".md-typeset .doc-section-item .doc-md-description") == {
        "display": "inline"
    }
    assert docs(".md-typeset .doc-section-item .doc-md-description > p") == {
        "display": "inline"
    }
    assert docs(".md-typeset .doc-contents p:has(> .doc-section-title)")["margin"] == (
        "0.8em 0 0.2em"
    )


def test_a_signature_parameter_stays_whole_unless_it_is_long() -> None:
    assert docs(".md-typeset .ref-param") == {"white-space": "nowrap"}
    assert docs(".md-typeset .ref-param-long") == {"white-space": "normal"}


METHOD_ROW = (
    "[data-md-color-scheme] .md-typeset .ref-methods > details.ref-method > summary"
)


def test_a_method_row_keeps_its_text_clear_of_the_fold_chevron() -> None:
    chevron = material(".md-typeset summary:after")
    inset = material("[dir=ltr] .md-typeset summary:after")["right"]
    assert chevron["position"] == "absolute"
    room = docs_px(docs(METHOD_ROW)["padding-inline-end"])
    assert room >= docs_px(inset) + docs_px(chevron["width"]) + docs_px("0.2rem")


def test_the_overview_rows_are_tighter_than_page_tables() -> None:
    cell = docs(".md-typeset table:not([class]) :is(th, td)")["padding"].split()
    row = docs(".md-typeset .ref-overview table:not([class]) :is(th, td)")
    assert docs_px(row["padding-block"]) < docs_px(cell[0])
    group = docs(
        "[data-md-color-scheme] .md-typeset .ref-overview .ref-overview-group th"
    )
    assert docs_px(group["padding-top"]) <= 2 * docs_px(row["padding-block"])
    assert docs_px(group["padding-bottom"]) <= docs_px(row["padding-block"])
    assert float(group["line-height"]) < 1.5


def test_an_entry_heading_has_one_size_at_any_level() -> None:
    assert docs(".md-typeset h2.ref-entry-heading") == {
        "font-size": docs(".md-typeset h3")["font-size"]
    }


def test_an_entry_right_after_the_intro_keeps_the_space_between_entries() -> None:
    assert docs(".md-typeset .doc-module + .ref-entry") == {
        "margin-top": docs(".md-typeset .ref-entry + .ref-entry")["margin-top"]
    }
