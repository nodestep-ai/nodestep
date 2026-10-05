import colorsys
import re
from importlib.resources import files
from typing import Any

from nodestep_sandbox import demo
from nodestep_sandbox.web import MERMAID_URL

PACKAGE = files("nodestep_sandbox")
STATIC = PACKAGE.joinpath("static")
TEMPLATES = PACKAGE.joinpath("templates")
COLOR_ROLES = {
    "neutral": (
        "page",
        "surface",
        "sunk",
        "border",
        "border-strong",
        "text",
        "text-secondary",
        "text-muted",
        "on-accent",
        "on-danger",
    ),
    "accent": ("accent", "accent-hover", "accent-text", "accent-wash"),
    "danger": ("danger", "danger-hover", "danger-wash"),
}
ACCENT_HUES = (30.0, 50.0)
DANGER_HUES = ((0.0, 12.0), (350.0, 360.0))
WARM_HUES = ((0.0, 60.0), (340.0, 360.0))
NEUTRAL_CHROMA = 16
GRAY_CHROMA = 2
NAMED_COLORS = (
    "black|silver|gray|white|maroon|red|purple|fuchsia|green|lime|olive|yellow|"
    "navy|blue|teal|aqua|orange"
)
THEME_BUTTON = (
    '<button class="nodestep-theme-button" type="button" aria-label="Dark theme" '
    'aria-pressed="false"><svg class="nodestep-theme-moon" viewBox="0 0 16 16" '
    'aria-hidden="true" focusable="false"><path d="M7.14 2.06A6 6 0 1 0 13.94 8.87 '
    '5 5 0 0 1 7.14 2.06z"/></svg><svg class="nodestep-theme-sun" viewBox="0 0 16 16" '
    'aria-hidden="true" focusable="false"><circle cx="8" cy="8" r="3"/><path '
    'd="M8 1v2M8 13v2M1 8h2M13 8h2M3.05 3.05l1.41 1.41M11.54 11.54l1.41 1.41'
    'M3.05 12.95l1.41-1.41M11.54 4.46l1.41-1.41"/></svg></button>'
)


def color_literals(text: str) -> list[str]:
    hexadecimal = r"(?<![\w&/])#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3,4})\b"
    functional = r"\b(?:rgba?|hsla?)\([^)]*\)"
    return re.findall(rf"{hexadecimal}|{functional}", text)


def named_colors(stylesheet: str) -> list[str]:
    values = re.findall(r":\s*([^;{}]+);", re.sub(r"/\*.*?\*/", "", stylesheet))
    return [
        name
        for value in values
        for name in re.findall(rf"(?<![\w-])(?:{NAMED_COLORS})(?![\w-])", value)
    ]


def hue_and_chroma(literal: str) -> tuple[float, int]:
    if literal.startswith("#"):
        digits = literal[1:]
        if len(digits) in (3, 4):
            digits = "".join(digit * 2 for digit in digits)
        red, green, blue = (int(digits[index : index + 2], 16) for index in (0, 2, 4))
    else:
        function, arguments = literal.split("(", 1)
        parts = re.split(r"[\s,/]+", arguments.rstrip(")").strip())
        if function.startswith("hsl"):
            hue = float(parts[0].removesuffix("deg")) / 360
            saturation = float(parts[1].rstrip("%")) / 100
            lightness = float(parts[2].rstrip("%")) / 100
            channels = colorsys.hls_to_rgb(hue, lightness, saturation)
            red, green, blue = (round(channel * 255) for channel in channels)
        else:
            red, green, blue = (
                round(float(part.rstrip("%")) * 2.55)
                if part.endswith("%")
                else round(float(part))
                for part in parts[:3]
            )
    hue, _, _ = colorsys.rgb_to_hls(red / 255, green / 255, blue / 255)
    return hue * 360, max(red, green, blue) - min(red, green, blue)


def within(hue: float, ranges: tuple[tuple[float, float], ...]) -> bool:
    return any(low <= hue <= high for low, high in ranges)


def color_family(literal: str) -> str | None:
    hue, chroma = hue_and_chroma(literal)
    if chroma <= GRAY_CHROMA:
        return "neutral"
    if chroma <= NEUTRAL_CHROMA:
        return "neutral" if within(hue, WARM_HUES) else None
    if within(hue, (ACCENT_HUES,)):
        return "accent"
    if within(hue, DANGER_HUES):
        return "danger"
    return None


def sandbox_sources() -> dict[str, str]:
    sources: dict[str, str] = {
        f"static/{name}": STATIC.joinpath(name).read_text(encoding="utf-8")
        for name in ("sandbox.css", "nodestep-theme.js", "diagram.js")
    }
    for template in TEMPLATES.iterdir():
        sources[f"templates/{template.name}"] = template.read_text(encoding="utf-8")
    return sources


def page_paths(site: Any) -> list[str]:
    run = site.last_run(site.start_json({"text": "hi"}))
    return [
        "/",
        "/runs/new",
        "/threads",
        f"/threads/{run.thread_id}",
        f"/runs/{run.id}",
        f"/runs/{run.id}/steps/0",
        "/runs/nope",
    ]


def test_the_vendored_stylesheet_is_nodestep_design_0_1_0a1() -> None:
    stylesheet = STATIC.joinpath("nodestep.css").read_text(encoding="utf-8")
    assert '--nodestep-design-version: "0.1.0a1";' in stylesheet
    assert ".nodestep-theme-button" in stylesheet
    assert ".nodestep-theme-toggle" not in stylesheet


def test_the_vendored_palette_is_one_neutral_scale_one_accent_and_one_danger() -> None:
    stylesheet = STATIC.joinpath("nodestep.css").read_text(encoding="utf-8")
    declared = set(re.findall(r"--nodestep-color-([\w-]+):", stylesheet))
    allowed = {name for names in COLOR_ROLES.values() for name in names}
    assert declared == allowed
    for family, names in COLOR_ROLES.items():
        for name in names:
            value = re.search(
                rf"--nodestep-color-{name}:\s*light-dark\((#[0-9a-f]{{6}}), (#[0-9a-f]{{6}})\);",
                stylesheet,
            )
            assert value, name
            assert [color_family(value.group(1)), color_family(value.group(2))] == [
                family,
                family,
            ], name


def test_every_color_in_the_vendored_stylesheet_is_neutral_accent_or_danger() -> None:
    stylesheet = STATIC.joinpath("nodestep.css").read_text(encoding="utf-8")
    literals = color_literals(stylesheet)
    assert literals
    assert not [literal for literal in literals if color_family(literal) is None]
    assert not named_colors(stylesheet)


def test_the_sandbox_files_write_no_colors_of_their_own() -> None:
    for name, source in sandbox_sources().items():
        assert not color_literals(source), name
        assert not named_colors(source), name
        assert "style=" not in source, name
        assert "<style" not in source, name


def test_the_sandbox_uses_only_declared_color_tokens() -> None:
    stylesheet = STATIC.joinpath("nodestep.css").read_text(encoding="utf-8")
    declared = set(re.findall(r"--nodestep-color-([\w-]+):", stylesheet))
    for name, source in sandbox_sources().items():
        used = set(re.findall(r"var\(--nodestep-color-([\w-]+)", source))
        assert used <= declared, (name, sorted(used - declared))


def test_every_page_loads_nodestep_css_before_the_sandbox_stylesheet(
    graphs: Any, open_site: Any
) -> None:
    site = open_site(graphs.echo())
    for path in page_paths(site):
        page = site.client.get(path).text
        theme = page.index('<script src="/static/nodestep-theme.js"></script>')
        shared = page.index('<link rel="stylesheet" href="/static/nodestep.css">')
        own = page.index('<link rel="stylesheet" href="/static/sandbox.css">')
        assert theme < shared < own < page.index("</head>"), path


def test_the_sandbox_stylesheet_builds_on_the_tokens() -> None:
    stylesheet = STATIC.joinpath("sandbox.css").read_text(encoding="utf-8")
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", stylesheet)
    assert not re.search(r"\b(rgb|rgba|hsl|hsla)\(", stylesheet)
    assert ":root" not in stylesheet
    assert not re.search(r"font-family:(?!\s*var\(--nodestep-font-)", stylesheet)
    assert "text-transform" not in stylesheet
    assert not re.search(r"(^|\})\s*(body|html|\*)\s*\{", stylesheet)
    assert "var(--nodestep-" in stylesheet
    assert len(stylesheet.splitlines()) < 120


def test_every_page_has_the_theme_button_in_the_top_bar(
    graphs: Any, open_site: Any
) -> None:
    site = open_site(graphs.echo())
    for path in page_paths(site):
        page = site.client.get(path).text
        top_bar = page[
            page.index('<header class="nodestep-topbar">') : page.index("</header>")
        ]
        end = top_bar[top_bar.index('<div class="nodestep-topbar-end">') :]
        assert end == (
            f'<div class="nodestep-topbar-end">\n{THEME_BUTTON}\n</div>\n'
        ), path
        assert "nodestep-theme-toggle" not in page, path


def test_the_theme_button_switches_between_light_and_dark() -> None:
    script = STATIC.joinpath("nodestep-theme.js").read_text(encoding="utf-8")
    assert 'const key = "nodestep-theme";' in script
    assert 'document.querySelectorAll(".nodestep-theme-button")' in script
    assert 'matchMedia("(prefers-color-scheme: dark)")' in script
    assert 'const theme = isDark() ? "light" : "dark";' in script
    assert "root.dataset.theme = theme;" in script
    assert "localStorage.setItem(key, theme);" in script
    assert 'setAttribute("aria-pressed", String(isDark()))' in script
    assert 'addEventListener("change", show)' in script
    assert 'addEventListener("DOMContentLoaded", wire)' in script
    assert "removeItem" not in script
    assert "system" not in script.replace("systemDark", "")


def test_the_theme_script_is_served_and_guards_local_storage(
    graphs: Any, open_site: Any
) -> None:
    site = open_site(graphs.echo())
    response = site.client.get("/static/nodestep-theme.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    script = response.text
    assert '"nodestep-theme"' in script
    for match in re.finditer("localStorage", script):
        before = script[: match.start()]
        assert before.count("try {") > before.count("} catch")
    for name in ("theme.js", "theme-toggle.js"):
        assert site.client.get(f"/static/{name}").status_code == 404, name


def test_form_fields_span_their_card() -> None:
    capped = [
        f"{name}: {selector.strip()}"
        for name in ("sandbox.css", "nodestep.css")
        for selector, body in re.findall(
            r"([^{}]+)\{([^{}]*)\}", STATIC.joinpath(name).read_text(encoding="utf-8")
        )
        if re.search(r"nodestep-(?:field|input|select|textarea)\b", selector)
        and re.search(r"(?<![\w-])max-(?:width|inline-size):", body)
    ]
    assert capped == []


def test_the_sandbox_stylesheet_keeps_only_sandbox_rules() -> None:
    stylesheet = STATIC.joinpath("sandbox.css").read_text(encoding="utf-8")
    for name in (".muted", ".mono", ".check", ".meta", ".row", ".outcome", "summary"):
        assert name not in stylesheet, name
    assert len(stylesheet.splitlines()) < 30


def test_pages_use_the_layout_classes_of_nodestep_design(
    graphs: Any, open_site: Any
) -> None:
    site = open_site(graphs.echo())
    for path in page_paths(site):
        page = site.client.get(path).text
        assert page.index(
            '<a class="nodestep-skip-link" href="#content">'
        ) < page.index('<header class="nodestep-topbar">'), path
        assert (
            '<main class="nodestep-main nodestep-stack" id="content" tabindex="-1">'
            in page
        ), path
    run = site.last_run(site.start_json({"text": "hi"}))
    page = site.client.get(f"/runs/{run.id}").text
    assert '<dl class="nodestep-pairs">' in page
    assert '<ol class="nodestep-list" aria-label="Events">' in page
    step = site.client.get(f"/runs/{run.id}/steps/0").text
    assert '<ol class="nodestep-breadcrumbs">' in step


def test_pages_have_no_inline_scripts(graphs: Any, open_site: Any) -> None:
    site = open_site(graphs.echo())
    for path in page_paths(site):
        page = site.client.get(path).text
        for tag in re.findall(r"<script[^>]*>", page):
            assert " src=" in tag, (path, tag)
        assert not re.search(r"<script[^>]*>[^<]", page), path


def test_only_the_graph_page_loads_mermaid(graphs: Any, open_site: Any) -> None:
    site = open_site(graphs.echo())
    for path in page_paths(site):
        page = site.client.get(path).text
        assert (MERMAID_URL in page) == (path == "/"), path


def test_the_diagram_sits_in_the_mermaid_frame(graphs: Any, open_site: Any) -> None:
    page = open_site(graphs.echo()).client.get("/").text
    assert '<pre class="mermaid nodestep-mermaid">' in page


def test_statuses_show_as_badges(graphs: Any, open_site: Any) -> None:
    site = open_site(graphs.review())
    run = site.last_run(site.start_json({"draft": "d"}))
    threads = site.client.get("/threads").text
    assert '<span class="nodestep-badge nodestep-badge-paused">Paused</span>' in threads
    thread = site.client.get(f"/threads/{run.thread_id}").text
    assert '<span class="nodestep-badge nodestep-badge-paused">Paused</span>' in thread


def test_form_controls_use_the_shared_components(graphs: Any, open_site: Any) -> None:
    site = open_site(graphs.echo())
    page = site.client.get("/runs/new").text
    assert '<div class="nodestep-field">' in page
    assert '<label class="nodestep-label" for="state.text">' in page
    assert 'name="state.text" value="" class="nodestep-input">' in page
    assert 'name="raw" rows="8" class="nodestep-textarea nodestep-mono">' in page
    assert (
        '<button class="nodestep-button nodestep-button-primary" type="submit">' in page
    )
    refused = site.post("/runs", {"mode": "json", "raw": "[1"})
    assert refused.status_code == 422
    assert (
        'class="nodestep-textarea nodestep-mono" aria-invalid="true">' in refused.text
    )
    assert '<p class="nodestep-field-error">' in refused.text


def test_the_store_note_is_an_info_message(graphs: Any, open_site: Any) -> None:
    page = open_site(graphs.echo()).client.get("/").text
    assert '<p class="nodestep-message nodestep-message-info">' in page


def test_each_stacked_cell_holds_one_value_block(open_site: Any) -> None:
    page = open_site(demo.graph).client.get("/").text
    nodes = page[
        page.index('<h2 class="nodestep-card-title">Nodes</h2>') : page.index(
            '<h2 class="nodestep-card-title">State</h2>'
        )
    ]
    cell = re.search(r'<td data-label="Branches">(.*?)</td>', nodes)
    assert cell
    assert cell.group(1).startswith("<div><div><code>")
    assert cell.group(1).endswith("</code></div></div>")


def test_mermaid_edge_labels_and_stacked_badges_come_from_nodestep_design() -> None:
    shared = STATIC.joinpath("nodestep.css").read_text(encoding="utf-8")
    own = STATIC.joinpath("sandbox.css").read_text(encoding="utf-8")
    assert (
        ".nodestep-mermaid :is(.edgeLabel, .edgeLabel p, .edgeLabel rect, .labelBkg) {"
        in shared
    )
    assert ".nodestep-table-stack .nodestep-badge {" in shared
    assert "edgeLabel" not in own
    assert ".nodestep-badge" not in own


def test_the_graph_page_sets_up_mermaid_after_loading_it(
    graphs: Any, open_site: Any
) -> None:
    page = open_site(graphs.echo()).client.get("/").text
    mermaid = page.index(f'<script src="{MERMAID_URL}"')
    setup = page.index('<script src="/static/diagram.js"></script>')
    assert mermaid < setup
    script = STATIC.joinpath("diagram.js").read_text(encoding="utf-8")
    assert (
        'initialize({ startOnLoad: false, securityLevel: "strict", theme: "neutral" })'
        in script
    )
    assert "run(" in script


def test_every_page_links_the_favicon_file(graphs: Any, open_site: Any) -> None:
    site = open_site(graphs.echo())
    for path in page_paths(site):
        page = site.client.get(path).text
        assert (
            '<link rel="icon" href="/static/favicon.svg" type="image/svg+xml">' in page
        ), path
        assert "data:" not in page, path
    favicon = site.client.get("/static/favicon.svg")
    assert favicon.status_code == 200
    assert favicon.headers["content-type"].startswith("image/svg+xml")


def test_the_favicon_uses_only_token_colors_and_the_light_accent() -> None:
    stylesheet = STATIC.joinpath("nodestep.css").read_text(encoding="utf-8")
    tokens = {
        value
        for pair in re.findall(
            r"--nodestep-color-[\w-]+:\s*light-dark\((#[0-9a-f]{6}), (#[0-9a-f]{6})\);",
            stylesheet,
        )
        for value in pair
    }
    accent = re.search(
        r"--nodestep-color-accent: light-dark\((#[0-9a-f]{6}), ", stylesheet
    )
    assert accent
    favicon = STATIC.joinpath("favicon.svg").read_text(encoding="utf-8")
    colors = {literal.lower() for literal in color_literals(favicon)}
    assert accent.group(1) in colors
    assert colors <= tokens
    assert not named_colors(favicon)
    assert "prefers-color-scheme: dark" in favicon


def test_every_page_has_the_nodestep_mark_in_the_brand(
    graphs: Any, open_site: Any
) -> None:
    site = open_site(graphs.echo())
    for path in page_paths(site):
        page = site.client.get(path).text
        brand = re.search(r'<a class="nodestep-brand" href="/">(.*?)</a>', page, re.S)
        assert brand, path
        mark, label = brand.group(1).split("</svg>")
        assert mark.startswith(
            '<svg class="nodestep-brand-mark" viewBox="0 0 64 64" aria-hidden="true"'
            ' focusable="false">'
        ), path
        assert label == "nodestep sandbox", path
        assert 'stroke="currentColor"' in mark
        assert mark.count('fill="currentColor"') == 3
        assert mark.count('class="nodestep-mark-accent-fill"') == 1
        assert not color_literals(mark)
