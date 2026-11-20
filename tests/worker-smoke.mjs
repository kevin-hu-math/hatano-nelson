// Runs the production worker body with real Pyodide. Only the remote module
// URL and local Python-file fetches are adapted for this Node-only harness.
import assert from "node:assert/strict";
import {readFile,writeFile,mkdtemp,rm} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {pathToFileURL} from "node:url";
const pyodideModule=process.env.PYODIDE_MODULE || import.meta.resolve("pyodide");
const runtimeURL="https://cdn.jsdelivr.net/pyodide/v0.28.3/full/pyodide.mjs";
const source=(await readFile(new URL("../worker.mjs",import.meta.url),"utf8")).replace(JSON.stringify(runtimeURL),JSON.stringify(pyodideModule));
const directory=await mkdtemp(join(tmpdir(),"hn-worker-test-"));
const messages=[];
const originalFetch=globalThis.fetch;
globalThis.self={postMessage:(message,transfer=[])=>messages.push(structuredClone(message,{transfer}))};
globalThis.fetch=async(url,...args)=>{
  const name=String(url).split("/").pop();
  if(["simulation.py","browser_api.py"].includes(name))return {ok:true,text:()=>readFile(new URL(`../python/${name}`,import.meta.url),"utf8")};
  return originalFetch(url,...args);
};
try {
  await writeFile(join(directory,"worker.mjs"),source);
  await import(pathToFileURL(join(directory,"worker.mjs")));
  const config={n:80,degree:3,g:.6,disorder_size:2,disorder:"Uniform",orientation:"Random edge directions",seed:42,
    solver:"Full spectrum (dense)",eigenpairs:8,target_real:.3,target_imag:.4,tolerance:1e-9,max_iterations:10000,threads:1,save_eigenvectors:true};
  await self.onmessage({data:{type:"run",id:1,config}});
  const error=messages.find(m=>m.type==="error");assert.equal(error,undefined,error?.message);
  assert.ok(messages.some(m=>m.type==="progress" && m.message.includes("Measuring")));
  const result=messages.find(m=>m.type==="result");assert.equal(result.payload.data.real.length,80);
  await self.onmessage({data:{type:"export",id:2,kind:"vectors"}});
  const file=messages.find(m=>m.type==="file");
  assert.ok(file,file?.message);assert.equal(file.name,"hatano-nelson-eigenvectors.npy");
  assert.equal(new TextDecoder().decode(file.bytes.slice(1,6)),"NUMPY");
  console.log("Production worker: Python progress callbacks, result messages and transferable NPY download passed.");
}finally{globalThis.fetch=originalFetch;await rm(directory,{recursive:true,force:true});}
