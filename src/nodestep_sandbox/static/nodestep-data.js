(() => {
  const searched =
    ".nodestep-data-key, .nodestep-data-value, .nodestep-data-text, .nodestep-data-message-name, .nodestep-data-call-name, .nodestep-data-call-arguments";
  const searches = new WeakMap();
  const labels = new WeakMap();
  document.documentElement.setAttribute("data-nodestep-data", "");

  const pattern = (query) =>
    query ? new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "giu") : null;

  const mark = (leaf, found) => {
    const text = leaf.textContent;
    const parts = [];
    let last = 0;
    if (found) {
      for (const match of text.matchAll(found)) {
        if (match.index > last) parts.push(document.createTextNode(text.slice(last, match.index)));
        const highlight = document.createElement("mark");
        highlight.textContent = match[0];
        parts.push(highlight);
        last = match.index + match[0].length;
      }
    }
    if (parts.length === 0) {
      if (leaf.querySelector("mark")) leaf.textContent = text;
      return;
    }
    if (last < text.length) parts.push(document.createTextNode(text.slice(last)));
    leaf.replaceChildren(...parts);
  };

  const describe = (search) => {
    const total = search.matches.length;
    const index = search.matches.indexOf(search.current);
    if (!search.query) return "";
    if (total === 0) return "No matches";
    return `${index + 1} of ${total}`;
  };

  const show = (viewer, search) => {
    for (const element of viewer.querySelectorAll('mark[aria-current="true"]')) {
      if (element !== search.current) element.removeAttribute("aria-current");
    }
    viewer.querySelector(".nodestep-data-matches").textContent = describe(search);
    if (!search.current) return;
    search.current.setAttribute("aria-current", "true");
    for (let node = search.current.parentElement; node && node !== viewer; node = node.parentElement) {
      if (node.tagName === "DETAILS") node.open = true;
    }
    const tree = viewer.querySelector(".nodestep-data-tree");
    const box = tree.getBoundingClientRect();
    const match = search.current.getBoundingClientRect();
    tree.scrollTop += match.top - box.top - (tree.clientHeight - match.height) / 2;
    if (viewer.querySelector(".nodestep-data-search") === document.activeElement) {
      viewer.scrollIntoView({ block: "nearest" });
    }
  };

  const matchesIn = (viewer) => [...viewer.querySelector(".nodestep-data-tree").querySelectorAll("mark")];

  const search = (viewer, query) => {
    const found = pattern(query);
    for (const leaf of viewer.querySelector(".nodestep-data-tree").querySelectorAll(searched)) {
      mark(leaf, leaf.closest("[data-nodestep-data-echo]") ? null : found);
    }
    const matches = matchesIn(viewer);
    const state = { query, matches, current: matches[0] ?? null };
    searches.set(viewer, state);
    show(viewer, state);
  };

  const step = (viewer, by) => {
    const state = searches.get(viewer);
    if (!state || state.matches.length === 0) return;
    const total = state.matches.length;
    const index = state.matches.indexOf(state.current);
    state.current = state.matches[(index + by + total) % total];
    show(viewer, state);
  };

  const expand = (viewer, open) => {
    const selector = open
      ? ".nodestep-data-node, .nodestep-data-message-row"
      : ".nodestep-data-node, .nodestep-data-message-row, .nodestep-data-raw";
    for (const node of viewer.querySelectorAll(selector)) node.open = open;
  };

  const copy = (viewer, button) => {
    if (!labels.has(button)) labels.set(button, button.textContent);
    const say = (text) => {
      button.textContent = text;
      setTimeout(() => {
        button.textContent = labels.get(button);
      }, 2000);
    };
    const text = viewer.querySelector(".nodestep-data-source").textContent;
    new Promise((resolve) => resolve(navigator.clipboard.writeText(text))).then(
      () => say("Copied"),
      () => say("Copy failed"),
    );
  };

  const viewerOf = (target, selector) => target.closest?.(selector)?.closest(".nodestep-data") ?? null;

  document.addEventListener("input", (event) => {
    const viewer = viewerOf(event.target, ".nodestep-data-search");
    if (viewer) search(viewer, event.target.value);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.isComposing) return;
    const viewer = viewerOf(event.target, ".nodestep-data-search");
    if (!viewer) return;
    event.preventDefault();
    step(viewer, event.shiftKey ? -1 : 1);
  });

  document.addEventListener("click", (event) => {
    const button = event.target.closest?.("[data-nodestep-data-action]");
    const viewer = button?.closest(".nodestep-data");
    if (!viewer) return;
    const action = button.getAttribute("data-nodestep-data-action");
    if (action === "expand") expand(viewer, true);
    if (action === "collapse") expand(viewer, false);
    if (action === "copy") copy(viewer, button);
  });
})();
