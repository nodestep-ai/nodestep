(() => {
  const key = "nodestep-theme";
  const root = document.documentElement;
  const load = () => {
    try {
      const stored = localStorage.getItem(key);
      if (stored === "light" || stored === "dark") root.dataset.theme = stored;
      else delete root.dataset.theme;
    } catch {}
  };
  load();
  const systemDark = matchMedia("(prefers-color-scheme: dark)");
  const isDark = () => (root.dataset.theme ?? (systemDark.matches ? "dark" : "light")) === "dark";
  const buttons = () => document.querySelectorAll(".nodestep-theme-button");
  const show = () => {
    for (const button of buttons()) button.setAttribute("aria-pressed", String(isDark()));
  };
  const reload = () => {
    load();
    show();
  };
  addEventListener("pageshow", reload);
  addEventListener("storage", reload);
  document.addEventListener("mousedown", (event) => {
    if (event.target.closest(".nodestep-search-clear")) event.preventDefault();
  });
  document.addEventListener("reset", (event) => {
    event.target.closest(".nodestep-search")?.querySelector(".nodestep-search-input")?.focus();
  });
  const wire = () => {
    show();
    systemDark.addEventListener("change", show);
    for (const button of buttons()) {
      button.addEventListener("click", () => {
        const theme = isDark() ? "light" : "dark";
        root.dataset.theme = theme;
        try {
          localStorage.setItem(key, theme);
        } catch {}
        show();
      });
    }
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", wire);
  else wire();
})();
