import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";

const ROOT = new URL("../../", import.meta.url);
const read = (path: string): string => readFileSync(new URL(path, ROOT), "utf8");
const TEMPLATE = read("overrides/main.html");
const PALETTE_SETTING = "{{ config.theme.palette | tojson }}";
const REPOSITORY_SETTING =
  "{{ (page.meta.repository if page and page.meta.repository else config.repo_name) | tojson }}";
const PALETTES: {
  media: string;
  scheme: string;
  primary: string;
  accent: string;
}[] = JSON.parse(read("tests/docs/fixtures/docs-palettes.json"));
const MATERIAL_STORAGE =
  '__md_scope=new URL("../",location),' +
  "__md_get=(e,_=localStorage,t=__md_scope)=>JSON.parse(_.getItem(t.pathname+\".\"+e))," +
  "__md_set=(e,_,t=localStorage,a=__md_scope)=>{try{t.setItem(a.pathname+\".\"+e,JSON.stringify(_))}catch(e){}}";
const MATERIAL_PALETTE_KEY = "/nodestep/.__palette";
const THEMES: [string, number][] = [
  ["light", 0],
  ["dark", 1],
];

type ChangeListener = (event: unknown) => void;

class FakeStorage {
  readonly items = new Map<string, string>();

  constructor(initial: Record<string, string>) {
    for (const [key, value] of Object.entries(initial))
      this.items.set(key, value);
  }

  getItem(key: string): string | null {
    return this.items.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.items.set(key, String(value));
  }
}

interface Docs {
  storage: FakeStorage;
  palette: () => unknown;
  scheme: () => string | undefined;
  clicks: string[];
  save: (name: string, value: unknown) => void;
  toggle: (name: string, scheme: string) => void;
  system: (dark: boolean) => void;
  fire: (type: string) => void;
}

function bridge(repository: string): string {
  const block =
    /\{% block extrahead %\}\s*<script>([\s\S]*?)<\/script>\s*\{% endblock %\}/.exec(
      TEMPLATE,
    );
  assert.ok(block);
  assert.ok(block[1].includes(PALETTE_SETTING));
  assert.ok(block[1].includes(REPOSITORY_SETTING));
  return block[1]
    .replace(PALETTE_SETTING, JSON.stringify(PALETTES))
    .replace(REPOSITORY_SETTING, JSON.stringify(repository));
}

function loadDocs({
  stored = {},
  systemDark = false,
  blocked = false,
  repository = "nodestep-ai/nodestep",
}: {
  stored?: Record<string, string>;
  systemDark?: boolean;
  blocked?: boolean;
  repository?: string;
} = {}): Docs {
  const storage = new FakeStorage(stored);
  let dark = systemDark;
  const listeners: ChangeListener[] = [];
  const mediaListeners: (() => void)[] = [];
  const windowListeners: [string, () => void][] = [];
  const clicks: string[] = [];
  const body = { dataset: {} as Record<string, string> };
  const inputs = PALETTES.map(({ scheme }) => ({
    name: "__palette",
    checked: false,
    dataset: { mdColorScheme: scheme },
    click() {
      if (this.checked) return;
      for (const input of inputs) input.checked = input === this;
      clicks.push(scheme);
      body.dataset.mdColorScheme = scheme;
      for (const listener of listeners) listener({ target: this });
    },
  }));
  const context: Record<string, unknown> = {
    URL,
    location: "http://127.0.0.1/nodestep/quickstart/",
    matchMedia(query: string) {
      return {
        get matches() {
          if (query === "(prefers-color-scheme: dark)") return dark;
          if (query === "(prefers-color-scheme: light)") return !dark;
          return false;
        },
        addEventListener(type: string, listener: () => void) {
          if (type === "change") mediaListeners.push(listener);
        },
      };
    },
    addEventListener(type: string, listener: () => void) {
      windowListeners.push([type, listener]);
    },
    document: {
      body,
      addEventListener(type: string, listener: ChangeListener) {
        if (type === "change") listeners.push(listener);
      },
      querySelectorAll(selector: string) {
        assert.equal(selector, 'input[name="__palette"]');
        return inputs;
      },
    },
  };
  Object.defineProperty(context, "localStorage", {
    get: () => {
      if (blocked) throw new Error("SecurityError");
      return storage;
    },
  });
  runInNewContext(MATERIAL_STORAGE, context);
  runInNewContext(bridge(repository), context);
  const current = () =>
    JSON.parse(
      runInNewContext('JSON.stringify(__md_get("__palette"))', context),
    );
  body.dataset.mdColorScheme = current().color.scheme;
  return {
    storage,
    palette: current,
    scheme: () => body.dataset.mdColorScheme,
    clicks,
    save: (name, value) =>
      runInNewContext(
        `__md_set(${JSON.stringify(name)}, ${JSON.stringify(value)})`,
        context,
      ),
    toggle: (name, scheme) => {
      for (const listener of listeners)
        listener({ target: { name, dataset: { mdColorScheme: scheme } } });
    },
    system: (value) => {
      dark = value;
      for (const listener of mediaListeners) listener();
    },
    fire: (type) => {
      for (const [name, listener] of windowListeners)
        if (name === type) listener();
    },
  };
}

function palette(index: number): unknown {
  const { media, scheme, primary, accent } = PALETTES[index];
  return { index, color: { media, scheme, primary, accent } };
}

test("a stored choice is the palette the docs show, whatever the system", () => {
  for (const [theme, index] of THEMES) {
    const docs = loadDocs({
      stored: { "nodestep-theme": theme },
      systemDark: theme === "light",
    });
    assert.deepEqual(docs.palette(), palette(index));
    docs.system(theme === "dark");
    assert.deepEqual(docs.palette(), palette(index));
    assert.equal(docs.scheme(), PALETTES[index].scheme);
    assert.deepEqual(docs.clicks, []);
  }
});

test("with no choice anywhere an open docs page follows the system as it changes", () => {
  const docs = loadDocs({ systemDark: false });
  assert.equal(docs.scheme(), "default");
  docs.system(true);
  assert.equal(docs.scheme(), "slate");
  docs.system(false);
  assert.equal(docs.scheme(), "default");
  assert.deepEqual(docs.clicks, ["slate", "default"]);
  assert.equal(docs.storage.items.size, 0);
});

test("a docs page shown again from the back and forward cache takes the stored choice", () => {
  const docs = loadDocs({
    stored: { "nodestep-theme": "light" },
    systemDark: true,
  });
  assert.equal(docs.scheme(), "default");
  docs.storage.items.set("nodestep-theme", "dark");
  docs.fire("pageshow");
  assert.equal(docs.scheme(), "slate");
  docs.fire("pageshow");
  assert.deepEqual(docs.clicks, ["slate"]);
  assert.equal(docs.storage.items.get("nodestep-theme"), "dark");
});

test("a choice made in another tab is applied to an open docs page at once", () => {
  const docs = loadDocs({ systemDark: false });
  docs.storage.items.set("nodestep-theme", "dark");
  docs.fire("storage");
  assert.equal(docs.scheme(), "slate");
  docs.storage.items.set("nodestep-theme", "light");
  docs.fire("storage");
  assert.equal(docs.scheme(), "default");
  assert.deepEqual(docs.clicks, ["slate", "default"]);
  assert.equal(docs.storage.items.get("nodestep-theme"), "light");
});

test("a palette Material stored earlier becomes the shared choice", () => {
  for (const [theme, index] of THEMES) {
    const docs = loadDocs({
      stored: { [MATERIAL_PALETTE_KEY]: JSON.stringify(palette(index)) },
      systemDark: theme === "light",
    });
    assert.equal(docs.storage.items.get("nodestep-theme"), theme);
    assert.deepEqual(docs.palette(), palette(index));
  }
});

test("a shared choice wins over a palette Material stored earlier", () => {
  const docs = loadDocs({
    stored: {
      "nodestep-theme": "dark",
      [MATERIAL_PALETTE_KEY]: JSON.stringify(palette(0)),
    },
  });
  assert.equal(docs.storage.items.get("nodestep-theme"), "dark");
  assert.deepEqual(docs.palette(), palette(1));
});

test("values other than light and dark count as no choice", () => {
  for (const value of ["sepia", "system", "", "constructor"]) {
    const docs = loadDocs({
      stored: { "nodestep-theme": value },
      systemDark: true,
    });
    assert.deepEqual(docs.palette(), palette(1), value);
  }
});

test("Material no longer stores its palette, other keys are stored as before", () => {
  const docs = loadDocs();
  docs.save("__palette", palette(1));
  assert.ok(!docs.storage.items.has(MATERIAL_PALETTE_KEY));
  assert.deepEqual(docs.palette(), palette(0));
  docs.save("__search", "graph");
  assert.equal(docs.storage.items.get("/nodestep/.__search"), '"graph"');
});

test("the docs toggle stores the shared choice", () => {
  const docs = loadDocs({ systemDark: true });
  docs.toggle("__palette", "default");
  assert.equal(docs.storage.items.get("nodestep-theme"), "light");
  assert.deepEqual(docs.palette(), palette(0));
  docs.toggle("__palette", "slate");
  assert.equal(docs.storage.items.get("nodestep-theme"), "dark");
  docs.toggle("other", "default");
  assert.equal(docs.storage.items.get("nodestep-theme"), "dark");
});

test("blocked storage leaves the docs on the system theme without an error", () => {
  const docs = loadDocs({ blocked: true, systemDark: true });
  assert.deepEqual(docs.palette(), palette(1));
  docs.toggle("__palette", "default");
  docs.save("__palette", palette(0));
  assert.deepEqual(docs.palette(), palette(1));
  docs.system(false);
  docs.fire("pageshow");
  docs.fire("storage");
  assert.equal(docs.scheme(), "default");
});

test("repository facts are kept apart for each repository", () => {
  const nodestep = loadDocs({ repository: "nodestep-ai/nodestep" });
  nodestep.save("__source", { stars: 1 });
  const nodeartifact = loadDocs({
    stored: Object.fromEntries(nodestep.storage.items),
    repository: "nodestep-ai/nodeartifact",
  });
  nodeartifact.save("__source", { stars: 2 });
  assert.equal(nodeartifact.storage.items.get("/nodestep/.__source nodestep-ai/nodestep"), '{"stars":1}');
  assert.equal(nodeartifact.storage.items.get("/nodestep/.__source nodestep-ai/nodeartifact"), '{"stars":2}');
  assert.ok(!nodeartifact.storage.items.has("/nodestep/.__source"));
});
