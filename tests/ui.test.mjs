import test from "node:test";
import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {toCSV,FORMAT} from "../plots.mjs";
const {JSDOM}=await import(process.env.JSDOM_MODULE || "jsdom");

test("DOM controls, examples, worker messages, cancellation, imports and raw colors",async()=>{
  const html=await readFile(new URL("../index.html",import.meta.url),"utf8");
  const example=JSON.parse(await readFile(new URL("../example.json",import.meta.url),"utf8"));
  const dom=new JSDOM(html,{url:"https://example.test/project/"});
  const $=id=>dom.window.document.getElementById(id),tick=()=>new Promise(r=>setTimeout(r,0));
  const plots=[],instances=[];
  class FakeWorker {
    constructor(url,options){this.url=url;this.options=options;instances.push(this);}
    postMessage(message){this.message=message;}
    terminate(){this.terminated=true;}
  }
  globalThis.document=dom.window.document;globalThis.location=dom.window.location;
  globalThis.Worker=FakeWorker;
  globalThis.Plotly={react:(_,data,layout)=>{plots.push({data,layout});},downloadImage:async()=>{}};
  globalThis.fetch=async()=>({ok:true,json:async()=>structuredClone(example)});
  await import("../app.mjs");
  assert.equal($("coordinates").value,"complex");
  $("example").click();await tick();
  assert.equal($("results").hidden,false);assert.equal(instances.length,0);
  assert.equal(plots.at(-1).layout.xaxis.title.text,"Re λ");
  assert.deepEqual([plots.at(-1).data[1].zmin,plots.at(-1).data[1].zmax],[0,1]);
  $("metric").value="l4";$("metric").dispatchEvent(new dom.window.Event("change"));
  assert.equal(instances.length,0);
  $("n").value="2";$("degree").value="1";
  $("parameters").dispatchEvent(new dom.window.Event("submit",{cancelable:true}));
  const first=instances[0];assert.equal(first.options.type,"module");assert.equal(first.message.config.n,2);
  assert.equal($("run").disabled,true);
  first.onmessage({data:{type:"progress",id:first.message.id,message:"Computing",value:.3}});
  assert.equal($("status-message").textContent,"Computing");
  const result={format:FORMAT,metadata:{config:first.message.config,full_spectrum:true,elapsed_seconds:.1,notes:[]},
    data:{real:[-1,1],imag:[0,0],l4:[.84,.84],linf:[.71,.71],ipr:[.5,.5],residual:[0,0]}};
  first.onmessage({data:{type:"result",id:first.message.id,payload:result}});
  assert.equal($("run").disabled,false);assert.equal($("download-matrix").disabled,false);
  assert.ok($("plot-note").textContent.includes("Im λ = 0"));
  $("parameters").dispatchEvent(new dom.window.Event("submit",{cancelable:true}));
  $("cancel").click();assert.equal(first.terminated,true);assert.equal($("cancel").disabled,true);
  assert.equal($("results").hidden,false);assert.equal($("download-matrix").disabled,true);
  Object.defineProperty($("import"),"files",{configurable:true,value:[{name:"eigenvalues.csv",text:async()=>toCSV(result)}]});
  $("import").dispatchEvent(new dom.window.Event("change"));await tick();
  assert.ok($("selection-note").textContent.includes("does not specify"));
  assert.equal($("error").hidden,true);
  dom.window.close();
});
