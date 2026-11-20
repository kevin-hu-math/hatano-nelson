// Module workers work on both GitHub project pages and domain-root pages.
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/pyodide.mjs";
let pythonPromise;
let activeId;
let busy = false;
const progress = (message, value = 0) => self.postMessage({type: "progress", id: activeId, message, value});

async function initialize() {
  progress("Loading browser Python (first run downloads the numerical libraries)", 0.01);
  const {loadPyodide} = await import(PYODIDE_URL);
  const py = await loadPyodide({
    stdout: () => {},
    stderr: (message) => console.debug(message),
  });
  progress("Loading NumPy, SciPy, and NetworkX", 0.02);
  await py.loadPackage(["numpy", "scipy", "networkx"]);
  const files = await Promise.all(["simulation.py", "browser_api.py"].map(async (name) => {
    const response = await fetch(new URL(`./python/${name}`, import.meta.url));
    if (!response.ok) throw new Error(`Cannot load python/${name}: HTTP ${response.status}`);
    return [name, await response.text()];
  }));
  py.FS.mkdirTree("/app");
  for (const [name, code] of files) py.FS.writeFile(`/app/${name}`, code);
  py.globals.set("browser_progress", progress);
  await py.runPythonAsync("import sys\nsys.path.insert(0, '/app')\nfrom browser_api import run_simulation, export_file");
  return py;
}

self.onmessage = async ({data}) => {
  if (busy) return self.postMessage({type: "error", id: data.id, message: "A calculation is already running."});
  busy = true;
  activeId = data.id;
  try {
    pythonPromise ??= initialize();
    const py = await pythonPromise;
    if (data.type === "run") {
      py.globals.set("config_json", JSON.stringify(data.config));
      const serialized = await py.runPythonAsync("run_simulation(config_json, browser_progress)");
      self.postMessage({type: "result", id: data.id, payload: JSON.parse(serialized)});
    } else if (data.type === "export") {
      py.globals.set("export_kind", data.kind);
      const path = py.runPython("export_file(export_kind)");
      const bytes = py.FS.readFile(path);
      py.FS.unlink(path);
      self.postMessage({type: "file", id: data.id, name: path.split("/").pop(), bytes}, [bytes.buffer]);
    } else throw new Error("Unknown worker request.");
  } catch (error) {
    self.postMessage({type: "error", id: data.id, message: String(error.message || error)});
    // Retrying an initialization failure should create a fresh worker.
  } finally {
    busy = false;
  }
};
