// Exercises the actual Python/WASM LAPACK, ARPACK, LU and serialization paths.
import assert from "node:assert/strict";
import {readFile,writeFile} from "node:fs/promises";
import {validatePayload,validateConfig} from "../plots.mjs";
const {loadPyodide}=await import(process.env.PYODIDE_MODULE || "pyodide");
const py=await loadPyodide();
await py.loadPackage(["numpy","scipy","networkx"]);
py.FS.mkdirTree("/app");
for(const name of ["simulation.py","browser_api.py"])py.FS.writeFile(`/app/${name}`,await readFile(new URL(`../python/${name}`,import.meta.url),"utf8"));
py.runPython("import sys\nsys.path.insert(0,'/app')\nfrom browser_api import run_simulation,export_file");
const base={n:80,degree:3,g:.6,disorder_size:2,disorder:"Uniform",orientation:"Random edge directions",seed:42,
  solver:"Full spectrum (dense)",eigenpairs:8,target_real:.3,target_imag:.4,tolerance:1e-9,max_iterations:10000,threads:1,save_eigenvectors:true};
const reports=[];
function run(config) {
  py.globals.set("cfg",JSON.stringify(validateConfig(config)));
  const payload=validatePayload(JSON.parse(py.runPython("run_simulation(cfg)")));
  let residual=0;for(const x of payload.data.residual)residual=Math.max(residual,x);
  assert.ok(residual<1e-7,`Residual ${residual}`);
  reports.push({n:config.n,solver:config.solver,disorder:config.disorder,orientation:config.orientation,
    eigenpairs:payload.data.real.length,seconds:payload.metadata.elapsed_seconds,residual});
  console.log(reports.at(-1));return payload;
}
for(const disorder of ["Uniform","Gaussian","Cauchy","Rademacher"])run({...base,disorder});
const gauge=run({...base,orientation:"Root-distance gauge"});assert.ok(gauge.data.imag.every(x=>x===0));
run({...base,solver:"Near target (sparse)"});
run({...base,n:500,save_eigenvectors:false});
run({...base,n:2000,solver:"Near target (sparse)",eigenpairs:20,save_eigenvectors:false});
run(base);
for(const kind of ["metrics","matrix","graph","vectors"]) {
  py.globals.set("kind",kind);
  const path=py.runPython("export_file(kind)");
  assert.ok(py.FS.readFile(path).byteLength>0);py.FS.unlink(path);
}
py.runPython(""+[
  "import numpy as np, scipy.sparse as sp",
  "v=np.load(export_file('vectors'))",
  "H=sp.load_npz(export_file('matrix'))",
  "d=np.load(export_file('metrics'))",
  "assert np.max(np.abs(H@v-v*d['eigenvalues'][None,:])) < 1e-7",
  "assert np.max(np.abs(np.sum(np.abs(v)**4,axis=0)-d['ipr'])) < 1e-10",
].join("\n"));
if(process.env.HN_WASM_REPORT)await writeFile(process.env.HN_WASM_REPORT,JSON.stringify(reports,null,2));
console.log("WASM solver and ordered binary exports passed.");
