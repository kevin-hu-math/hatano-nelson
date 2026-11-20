export const FORMAT = "hatano-nelson-browser/v1";
export const METRICS = {ipr: "IPR", l4: "L⁴", linf: "L∞"};
export const SOLVERS = ["Full spectrum (dense)", "Near target (sparse)", "Largest magnitude (sparse)"];
const COLUMNS = ["real", "imag", "l4", "linf", "ipr", "residual"];

export function validateConfig(c) {
  for (const key of ["n", "degree", "seed", "eigenpairs", "max_iterations"])
    if (!Number.isSafeInteger(c[key])) throw new Error(`${key} must be an integer.`);
  if (c.n < 1 || c.degree < 0 || c.degree >= c.n) throw new Error("Use N ≥ 1 and 0 ≤ κ < N.");
  if ((c.n * c.degree) % 2) throw new Error("N × κ must be even for a regular graph to exist.");
  if (!Number.isFinite(c.g) || Math.abs(c.g) > 700) throw new Error("g must be finite with |g| ≤ 700.");
  if (!Number.isFinite(c.disorder_size) || c.disorder_size < 0) throw new Error("W must be finite and nonnegative.");
  if (c.seed < 0 || c.max_iterations < 1) throw new Error("Use a nonnegative seed and positive iteration count.");
  if (!["Uniform", "Gaussian", "Cauchy", "Rademacher"].includes(c.disorder)) throw new Error("Unknown disorder law.");
  if (!["Random edge directions", "Root-distance gauge"].includes(c.orientation)) throw new Error("Unknown hopping convention.");
  if (!SOLVERS.includes(c.solver)) throw new Error("Unknown solver.");
  if (c.solver !== SOLVERS[0] && !(c.eigenpairs >= 1 && c.eigenpairs <= c.n - 2)) throw new Error("Sparse mode needs 1 ≤ eigenpairs ≤ N − 2.");
  if (![c.target_real, c.target_imag, c.tolerance].every(Number.isFinite) || !(c.tolerance > 0 && c.tolerance < 1))
    throw new Error("Use a finite target and a sparse tolerance strictly between 0 and 1.");
  return c;
}

export function validatePayload(payload) {
  if (payload?.format !== FORMAT || !payload.metadata || !payload.data) throw new Error("This is not a supported saved-run JSON.");
  const n = payload.data.real?.length;
  if (!n) throw new Error("The supplied spectrum is empty.");
  for (const key of COLUMNS) {
    const values = payload.data[key];
    if (!Array.isArray(values) || values.length !== n || !values.every(Number.isFinite))
      throw new Error(`Column ${key} must contain one finite number per eigenvalue.`);
  }
  for (const key of ["l4", "linf", "ipr"])
    if (!payload.data[key].every(x => x > 0 && x <= 1 + 1e-6)) throw new Error(`Column ${key} is inconsistent with normalized eigenvectors.`);
  if (!payload.data.residual.every(x => x >= 0)) throw new Error("Residuals cannot be negative.");
  const size = payload.metadata.config?.n;
  if (payload.metadata.config && (!Number.isSafeInteger(size) || size < n))
    throw new Error("Graph size N must be an integer at least as large as the supplied eigenvalue count.");
  if (payload.metadata.full_spectrum === true && size !== n) throw new Error("The full-spectrum label disagrees with N and the number of eigenvalues.");
  return payload;
}

export function parseCSV(text, filename = "Imported CSV") {
  const lines = text.replace(/^\uFEFF/, "").trim().split(/\r?\n/).filter(x => x.trim());
  const header = lines.shift()?.split(",").map(x => x.trim().replace(/^"|"$/g, "").toLowerCase());
  const indices = ["real", "imag", "l4", "linf", "ipr", "relative_residual"].map(key => header?.indexOf(key) ?? -1);
  if (indices.some(i => i < 0)) throw new Error("CSV needs the desktop columns: real,imag,L4,Linf,IPR,relative_residual.");
  const data = Object.fromEntries(COLUMNS.map(key => [key, []]));
  for (const [row, line] of lines.entries()) {
    const fields = line.split(",");
    for (let j = 0; j < COLUMNS.length; j++) {
      const raw = fields[indices[j]]?.trim();
      if (!raw) throw new Error(`Missing numeric value on CSV row ${row + 2}.`);
      const value = Number(raw);
      if (!Number.isFinite(value)) throw new Error(`Invalid numeric value on CSV row ${row + 2}.`);
      data[COLUMNS[j]].push(value);
    }
  }
  return validatePayload({format: FORMAT, data, metadata: {
    full_spectrum: null, computed_eigenvalues: data.real.length, source: filename,
    notes: ["Imported CSV does not identify graph parameters or whether this is the full spectrum. The histogram describes the supplied eigenvalues."],
  }});
}

export function toCSV(payload) {
  const rows = ["real,imag,L4,Linf,IPR,relative_residual"];
  for (let j = 0; j < payload.data.real.length; j++) rows.push(COLUMNS.map(key => payload.data[key][j]).join(","));
  return rows.join("\n") + "\n";
}

function quantile(sorted, p) {
  const i = (sorted.length - 1) * p, a = Math.floor(i), t = i - a;
  return sorted[a] * (1-t) + sorted[Math.min(a + 1, sorted.length - 1)] * t;
}

function interval(values, percent) {
  let low = Infinity, high = -Infinity;
  if (percent < 100) {
    const sorted = [...values].sort((a,b) => a-b), tail = (100-percent)/200;
    low = quantile(sorted, tail); high = quantile(sorted, 1-tail);
  } else for (const x of values) { low = Math.min(low, x); high = Math.max(high, x); }
  if (low === high) { const pad = Math.max(.05, .025*Math.abs(low)); low -= pad; high += pad; }
  return [low, high];
}

export function binSpectrum(payload, {metric = "ipr", bins = 60, central = 100, minimum = 1, coordinates = "complex"} = {}) {
  if (!METRICS[metric] || !Number.isInteger(bins) || bins < 2 || bins > 400 || !Number.isFinite(central) || central <= 0 || central > 100 || !Number.isInteger(minimum) || minimum < 1)
    throw new Error("Use valid plot controls: 2–400 bins, 0 < percentile ≤ 100, and a positive integer bin count.");
  const {real: x, imag: y} = payload.data, values = payload.data[metric], total = x.length;
  const physicallyReal = y.every((v,i) => Math.abs(v) <= 1e-10 * (1+Math.abs(x[i])));
  const realView = coordinates === "auto" && physicallyReal;
  const xr = interval(x, central), yr = interval(y, central), dx = (xr[1]-xr[0])/bins, dy = (yr[1]-yr[0])/bins;
  const counts = new Float64Array(realView ? bins : bins*bins), sums = new Float64Array(counts.length);
  const visible = [], density = new Array(counts.length), means = new Array(counts.length);
  for (let i=0; i<total; i++) {
    if (x[i] < xr[0] || x[i] > xr[1] || (!realView && (y[i] < yr[0] || y[i] > yr[1]))) continue;
    const ix = Math.min(bins-1, Math.floor((x[i]-xr[0])/dx));
    const iy = realView ? 0 : Math.min(bins-1, Math.floor((y[i]-yr[0])/dy));
    const j = iy*bins + ix;
    counts[j]++; sums[j] += values[i]; visible.push(i);
  }
  for (let j=0; j<counts.length; j++) {
    density[j] = counts[j]/(total*dx*(realView ? 1 : dy));
    means[j] = counts[j] >= minimum ? sums[j]/counts[j] : null;
  }
  const xc = Array.from({length: bins}, (_,i) => xr[0]+(i+.5)*dx), yc = Array.from({length: bins}, (_,i) => yr[0]+(i+.5)*dy);
  return {counts, density, means, visible, realView, physicallyReal, total, bins, xr, yr, dx, dy, xc, yc};
}

export function figureSpec(payload, options = {}) {
  const b = binSpectrum(payload, options), metric = options.metric || "ipr", label = METRICS[metric];
  const scatter = options.view === "scatter";
  const color = x => x;
  const colorTitle = label;
  const matrix = flat => Array.from({length:b.bins}, (_,i) => Array.from(flat.slice(i*b.bins, (i+1)*b.bins)));
  const data = [], isFull = payload.metadata.full_spectrum === true;
  const title = isFull ? "Empirical spectral density" : (payload.metadata.full_spectrum === false ? "Selected-eigenvalue histogram" : "Imported-eigenvalue histogram");
  const localization = scatter ? `Right-eigenvector ${label}` : `Mean ${label} per occupied bin`;
  const base = {showlegend:false, paper_bgcolor:"#fff", plot_bgcolor:"#fff", font:{family:"Arial, sans-serif",size:12,color:"#20252b"},
    margin:{l:55,r:90,t:60,b:55}, height:500, hovermode:"closest", uirevision: JSON.stringify([b.xr,b.yr,b.realView]),
    xaxis:{domain:[0,.405],range:b.xr,title:{text:b.realView?"E":"Re λ"},zerolinecolor:"#ccc"},
    xaxis2:{domain:[.59,1],range:b.xr,title:{text:b.realView?"E":"Re λ"},zerolinecolor:"#ccc"},
    yaxis:{domain:[0,1],title:{text:b.realView?"Density on ℝ":"Im λ"},zerolinecolor:"#ccc"},
    yaxis2:{domain:[0,1],title:{text:b.realView?label:"Im λ"},anchor:"x2",zerolinecolor:"#ccc"},
    annotations:[{text:title,x:.2025,y:1.10,xref:"paper",yref:"paper",showarrow:false},
                 {text:localization,x:.795,y:1.10,xref:"paper",yref:"paper",showarrow:false}]};
  const bar={title:{text:colorTitle},x:1.02,thickness:12,len:.9};
  if (b.realView) {
    data.push({type:"bar",x:b.xc,y:b.density,width:b.dx,marker:{color:"#4b718b"},customdata:Array.from(b.counts),
      hovertemplate:"E=%{x:.5g}<br>density=%{y:.5g}<br>count=%{customdata}<extra></extra>"});
    const x = scatter ? b.visible.map(i=>payload.data.real[i]) : b.xc;
    const values = scatter ? b.visible.map(i=>payload.data[metric][i]) : b.means;
    data.push({type:"scattergl",x,y:values,xaxis:"x2",yaxis:"y2",mode:"markers",marker:{size:5,color:values.map(color),cmin:0,cmax:1,cauto:false,colorscale:"Viridis",showscale:true,colorbar:bar},
      hovertemplate:`E=%{x:.5g}<br>${label}=%{y:.5g}<extra></extra>`});
    base.yaxis2.type = "linear";
    base.bargap=0;
  } else {
    base.yaxis.range=b.yr; base.yaxis2.range=b.yr;
    data.push({type:"heatmap",x:b.xc,y:b.yc,z:matrix(b.density.map((v,i)=>b.counts[i]?v:null)),customdata:matrix(b.counts),
      colorscale:"Cividis",colorbar:{title:{text:"Density"},x:.425,thickness:12,len:.9},hoverongaps:false,
      hovertemplate:"Re=%{x:.5g}<br>Im=%{y:.5g}<br>density=%{z:.5g}<br>count=%{customdata}<extra></extra>"});
    if (scatter) {
      data.push({type:"scattergl",x:b.visible.map(i=>payload.data.real[i]),y:b.visible.map(i=>payload.data.imag[i]),xaxis:"x2",yaxis:"y2",mode:"markers",
        customdata:b.visible.map(i=>payload.data[metric][i]),
        marker:{size:5,color:b.visible.map(i=>color(payload.data[metric][i])),cmin:0,cmax:1,cauto:false,colorscale:"Viridis",showscale:true,colorbar:bar},
        hovertemplate:`Re=%{x:.5g}<br>Im=%{y:.5g}<br>${label}=%{customdata:.5g}<extra></extra>`});
    } else data.push({type:"heatmap",x:b.xc,y:b.yc,z:matrix(b.means.map(color)),zmin:0,zmax:1,zauto:false,xaxis:"x2",yaxis:"y2",colorscale:"Viridis",colorbar:bar,hoverongaps:false,
      customdata:matrix(b.means.map((v,i)=>[v,b.counts[i]])),
      hovertemplate:"Re=%{x:.5g}<br>Im=%{y:.5g}<br>mean=%{customdata[0]:.5g}<br>count=%{customdata[1]}<extra></extra>"});
  }
  return {data,layout:base,binned:b};
}
