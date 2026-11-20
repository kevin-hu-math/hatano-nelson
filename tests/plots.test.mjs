import test from "node:test";
import assert from "node:assert/strict";
import {binSpectrum,figureSpec,validateConfig,validatePayload,parseCSV,toCSV,FORMAT} from "../plots.mjs";

const sample = (real=[-1,-.9,1,0],imag=[-1,-.9,1,0]) => ({format:FORMAT,
  metadata:{config:{n:4},full_spectrum:true},
  data:{real,imag,l4:[.6,.7,.8,.9],linf:[.4,.5,.6,.7],ipr:[.1,.3,.7,.5],residual:[0,0,0,0]}});
test("density integrates to one; bin means and clipped mass are correct",()=>{
  const p=sample();const b=binSpectrum(p,{bins:2});
  assert.ok(Math.abs(b.density.reduce((a,v)=>a+v,0)*b.dx*b.dy-1)<1e-12);
  assert.ok(Math.abs(b.means[0]-.2)<1e-12);
  const c=binSpectrum(p,{bins:2,central:60});
  assert.ok(Math.abs(c.density.reduce((a,v)=>a+v,0)*c.dx*c.dy-c.visible.length/4)<1e-12);
});
test("real eigenvalues default to the complex plane, with an optional real view",()=>{
  const p=sample([-1,0,1,2],[0,0,0,0]);
  assert.equal(binSpectrum(p).realView,false);
  const b=binSpectrum(p,{coordinates:"auto"});assert.equal(b.realView,true);
  assert.ok(Math.abs(b.density.reduce((a,v)=>a+v,0)*b.dx-1)<1e-12);
  assert.equal(figureSpec(p).layout.xaxis.title.text,"Re λ");
});
test("all metric heat maps and scatters have fixed raw 0 to 1 colors",()=>{
  for(const metric of ["ipr","l4","linf"])for(const view of ["binned","scatter"])for(const real of [false,true]) {
    const p=real?sample([-1,0,1,2],[0,0,0,0]):sample();
    const trace=figureSpec(p,{metric,view,coordinates:"auto"}).data[1];
    if(trace.type==="heatmap")assert.deepEqual([trace.zmin,trace.zmax],[0,1]);
    else assert.deepEqual([trace.marker.cmin,trace.marker.cmax],[0,1]);
  }
});
test("empty localization bins are null, without changing density counts",()=>{
  const b=binSpectrum(sample(),{minimum:100});assert.ok(b.means.every(x=>x===null));
  assert.equal(b.counts.reduce((a,v)=>a+v,0),4);
});
test("desktop CSV round trips and keeps its full-spectrum status unknown",()=>{
  const p=sample(), imported=parseCSV(toCSV(p));
  assert.deepEqual(imported.data,p.data);assert.equal(imported.metadata.full_spectrum,null);
  assert.throws(()=>parseCSV("real,imag\n0,0\n"));
  assert.throws(()=>validatePayload({...p,data:{...p.data,ipr:[NaN,0,0,0]}}));
});
test("graph parity and sparse selection are validated",()=>{
  const config={n:10,degree:3,g:.4,disorder_size:2,disorder:"Uniform",orientation:"Random edge directions",seed:42,
    solver:"Full spectrum (dense)",eigenpairs:4,target_real:0,target_imag:.2,tolerance:1e-9,max_iterations:10000};
  validateConfig(config);assert.throws(()=>validateConfig({...config,n:11}),/even/);
  assert.throws(()=>validateConfig({...config,solver:"Near target (sparse)",eigenpairs:9}),/N − 2/);
});
