import {validateConfig, validatePayload, parseCSV, toCSV, figureSpec} from "./plots.mjs";

const $ = id => document.getElementById(id);
const modelForm = $("parameters");
let worker=null, busy=false, workerHasResult=false, payload=null, resultInWorker=false;
let requestId=0, currentId=0, action="", started=0, clock;

const laws = {
  Uniform: "V ~ Uniform[−W/2, W/2]; W is the full box width.",
  Gaussian: "V ~ Normal(0, W²); W is the standard deviation.",
  Cauchy: "V = W × standard Cauchy; W is its scale. Tails are not truncated.",
  Rademacher: "V = −W or +W with equal probability; W is the amplitude.",
};

function configFromForm() {
  const c = {};
  for (const id of ["n","degree","g","disorder_size","seed","eigenpairs","target_real","target_imag","tolerance","max_iterations"]) c[id] = Number($(id).value);
  for (const id of ["disorder","orientation","solver"]) c[id] = $(id).value;
  return {...c, threads:1, save_eigenvectors:$("save_eigenvectors").checked};
}

function updateModelNotes() {
  $("disorder-note").textContent = laws[$("disorder").value];
  $("orientation-note").textContent = $("orientation").value === "Root-distance gauge"
    ? "Finite diagonal similarity: eigenvalues stay real and independent of g. Same-shell hopping is 1."
    : "Each edge independently has a preferred direction, with exp(g) forward and exp(−g) backward hopping.";
  const c=configFromForm(), sparse=c.solver!=="Full spectrum (dense)";
  $("sparse-controls").hidden=!sparse;
  $("target-controls").hidden=c.solver!=="Near target (sparse)";
  const estimate=(c.g===0 || c.orientation==="Root-distance gauge"?40:64)*c.n*c.n/2**20;
  $("memory-note").textContent = sparse
    ? "A selected subset, with O(Nk) eigenvector storage. Shift-invert LU can also need substantial memory."
    : `Dense planning estimate: ${estimate<1024?estimate.toFixed(1)+" MiB":(estimate/1024).toFixed(2)+" GiB"}, plus runtime and plot memory. Time scales roughly as N³.`;
}

function setStatus(message) { $("status-message").textContent=message; }
function showError(error) { $("error").textContent=String(error.message || error); $("error").hidden=false; }
function clearError() { $("error").hidden=true; }

function updateButtons() {
  $("run").disabled=busy; $("cancel").disabled=!busy;
  $("example").disabled=busy; $("import").disabled=busy;
  for (const id of ["metrics","matrix","graph"]) $("download-"+id).disabled=busy || !workerHasResult || !resultInWorker;
  $("download-vectors").disabled=busy || !workerHasResult || !resultInWorker || !payload?.metadata.config?.save_eigenvectors;
  $("release").disabled=busy || !worker;
}

function setBusy(value) {
  busy=value;
  clearInterval(clock);
  $("progress").hidden=!value;
  if (value) {
    started=performance.now();
    clock=setInterval(()=>{$("elapsed").textContent=`Elapsed ${((performance.now()-started)/1000).toFixed(0)} s`;},500);
  } else $("elapsed").textContent="";
  updateButtons();
}

function destroyWorker() { worker?.terminate(); worker=null; workerHasResult=false; resultInWorker=false; }
function ensureWorker() {
  if (worker) return;
  const instance=new Worker(new URL("./worker.mjs",import.meta.url),{type:"module"});
  worker=instance;
  instance.onmessage=({data})=>{
    if (worker!==instance || data.id!==currentId) return;
    if (data.type==="progress") { setStatus(data.message); $("progress").value=data.value; }
    else if (data.type==="result") {
      try {
        payload=validatePayload(data.payload); workerHasResult=true; resultInWorker=true;
        setBusy(false); showResult();
        setStatus(`Calculation complete in ${payload.metadata.elapsed_seconds.toFixed(1)} s (excluding runtime download).`);
      } catch(error) { setBusy(false); showError(error); }
    } else if(data.type==="file") {
      saveBlob(data.bytes,data.name,"application/octet-stream"); setBusy(false); setStatus("Download prepared.");
    } else if(data.type==="error") {
      const message=data.message;
      destroyWorker(); setBusy(false); setStatus("Calculation stopped."); showError(message);
    }
  };
  instance.onerror=event=>{
    if(worker!==instance)return;
    destroyWorker(); setBusy(false); setStatus("The numerical worker stopped.");
    showError(`${event.message || "The browser could not run the worker."}\nCheck your connection, or try a smaller N / fewer eigenpairs. Very large jobs may exceed browser memory.`);
  };
  instance.onmessageerror=()=>{destroyWorker();setBusy(false);showError("The browser could not transfer the result. Try a smaller calculation.");};
}

function plotOptions() { return {metric:$("metric").value, view:$("view").value, coordinates:$("coordinates").value,
  bins:Number($("bins").value), central:Number($("central").value), minimum:Number($("minimum").value)}; }

function renderPlot() {
  if (!payload) return;
  try {
    const spec=figureSpec(payload,plotOptions());
    if (!globalThis.Plotly) throw new Error("Plotly did not load. Check that assets/plotly.min.js was uploaded with this page.");
    Plotly.react($("spectrum-plot"),spec.data,spec.layout,{responsive:true,displaylogo:false,
      toImageButtonOptions:{format:"png",filename:"hatano-nelson-spectrum",scale:2}});
    const b=spec.binned;
    $("plot-note").textContent=`View includes ${b.visible.length.toLocaleString()}/${b.total.toLocaleString()} eigenvalues. Statistic colors always run from 0 to 1. Empty bins are blank; localization bins show arithmetic means. `
      +(b.realView?"One-dimensional density on ℝ.":"Binned density per unit area in ℂ.")
      +(b.physicallyReal && !b.realView?" The computed eigenvalues are numerically real and lie along Im λ = 0.":"");
  } catch(error) { showError(error); }
}

function showResult() {
  $("results").hidden=false; $("empty").hidden=true;
  const m=payload.metadata,c=m.config,n=payload.data.real.length;
  $("run-summary").textContent=c
    ? `N = ${c.n.toLocaleString()}, κ = ${c.degree}, g = ${c.g}, W = ${c.disorder_size}, ${c.disorder} disorder; ${c.orientation}. Seed ${c.seed}; ${n.toLocaleString()} eigenvalues.`
    : `${n.toLocaleString()} eigenvalues imported from ${m.source || "a saved result"}. Graph parameters were not supplied.`;
  $("selection-note").hidden=m.full_spectrum===true;
  $("selection-note").textContent=m.full_spectrum===false
    ? `${c.solver}: ${n.toLocaleString()} of ${c.n.toLocaleString()} eigenvalues. The histogram describes this solver-selected subset.`
    : "The histogram describes the imported eigenvalues. The CSV does not specify whether the complete spectrum was supplied.";
  let largest=0; for(const x of payload.data.residual)largest=Math.max(largest,x);
  $("solver-notes").textContent=`Maximum relative eigenpair residual: ${largest.toExponential(2)}. `+(m.notes || []).join(" ");
  $("metadata").textContent=JSON.stringify(m,null,2);
  updateButtons(); renderPlot();
}

function applyConfig(c) {
  if (!c) return;
  for(const el of modelForm.elements) {
    if(!el.name || !(el.name in c))continue;
    if(el.type==="checkbox")el.checked=!!c[el.name];
    else if(el.tagName==="SELECT") {
      if([...el.options].some(o=>o.value===c[el.name]))el.value=c[el.name];
    } else if(typeof c[el.name]==="number" && Number.isFinite(c[el.name]))el.value=c[el.name];
  }
  updateModelNotes();
}

modelForm.addEventListener("input",updateModelNotes);
modelForm.addEventListener("change",updateModelNotes);
modelForm.addEventListener("submit",event=>{
  event.preventDefault(); if(busy)return;clearError();
  try {
    const config=validateConfig(configFromForm());
    ensureWorker(); currentId=++requestId; action="run";workerHasResult=false;resultInWorker=false;
    $("progress").value=0; setBusy(true);setStatus("Starting numerical worker; the progress bar marks stages, not remaining time.");
    worker.postMessage({type:"run",id:currentId,config});
  } catch(error) { setBusy(false);showError(error); }
});

$("cancel").addEventListener("click",()=>{
  destroyWorker();setBusy(false);setStatus("Calculation cancelled. Any earlier plotted result remains available.");
});
$("release").addEventListener("click",()=>{destroyWorker();updateButtons();setStatus("Numerical runtime released. Plotted data and JSON/CSV downloads remain available.");});

for (const id of ["coordinates","metric","view","bins","central","minimum"])
  $(id).addEventListener("change",()=>{clearError();renderPlot();});

$("example").addEventListener("click",async()=>{
  clearError();$("example").disabled=true;
  try {
    const response=await fetch(new URL("./example.json",import.meta.url));
    if(!response.ok)throw new Error(`Cannot load example.json: HTTP ${response.status}`);
    payload=validatePayload(await response.json());resultInWorker=false;applyConfig(payload.metadata.config);showResult();
    setStatus("Loaded the included N = 1,000 desktop example. No numerical runtime download was needed.");
  }catch(error){showError(error);}finally{updateButtons();}
});

$("import").addEventListener("change",async event=>{
  const file=event.target.files[0];if(!file)return;clearError();
  try {
    const text=await file.text();
    payload=file.name.toLowerCase().endsWith(".csv")?parseCSV(text,file.name):validatePayload(JSON.parse(text));
    resultInWorker=false;applyConfig(payload.metadata.config);showResult();setStatus(`Loaded ${file.name}.`);
  } catch(error){showError(error);}finally{event.target.value="";}
});

function saveBlob(content,name,type) {
  const url=URL.createObjectURL(new Blob([content],{type}));
  const anchor=document.createElement("a");anchor.href=url;anchor.download=name;document.body.append(anchor);anchor.click();anchor.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$("download-json").addEventListener("click",()=>saveBlob(JSON.stringify(payload),"hatano-nelson-run.json","application/json"));
$("download-csv").addEventListener("click",()=>saveBlob(toCSV(payload),"eigenvalues.csv","text/csv"));
$("download-png").addEventListener("click",()=>Plotly.downloadImage($("spectrum-plot"),{format:"png",filename:"hatano-nelson-spectrum",width:1400,height:600,scale:2}).catch(showError));
for(const kind of ["metrics","matrix","graph","vectors"]) $("download-"+kind).addEventListener("click",()=>{
  if(!worker || busy || !workerHasResult || !resultInWorker)return;
  clearError();currentId=++requestId;action="export";setBusy(true);setStatus(`Preparing ${kind} download…`);$("progress").value=.95;
  worker.postMessage({type:"export",id:currentId,kind});
});

updateModelNotes(); updateButtons();
if(location.protocol==="file:")showError("Serve this folder through HTTP(S): use GitHub Pages, or run python -m http.server 8000 in this folder. Opening index.html directly cannot load a module worker.");
