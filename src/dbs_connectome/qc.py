"""Self-contained local browser viewer: no server, external scripts or telemetry."""

import base64
from html import escape
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from .masks import mask_qc, validate_binary
from .provenance import digest, image3d, same_grid


def browser_report(reference, mask, output, synthetic=False):
    ref, mi = image3d(reference), image3d(mask)
    same_grid(ref, mi)
    validate_binary(mi)
    if np.prod(ref.shape) > 32_000_000:
        raise ValueError("Browser QC size limit: explicitly prepare a reviewed display grid first")
    # Reorientation is for display only; no data or mask is resampled or changed on disk.
    r = nib.as_closest_canonical(ref)
    m = nib.as_closest_canonical(mi)
    same_grid(r, m)
    data = r.get_fdata()
    positive = data[data > 0]
    if positive.size == 0:
        raise ValueError("Reference contains no positive display signal")
    low, high = np.percentile(positive, [1, 99.5])
    if high <= low:
        high = low + 1
    volume = np.clip((data-low)/(high-low)*255, 0, 255).astype(np.uint8)
    masked = np.asarray(m.dataobj, dtype=np.uint8)
    payload = {
        "shape": list(r.shape), "affine": r.affine.tolist(),
        "image": base64.b64encode(volume.tobytes(order="F")).decode(),
        "mask": base64.b64encode(masked.tobytes(order="F")).decode(),
        "reference_sha256": digest(reference), "mask_sha256": digest(mask),
        "intensity_window": [float(low), float(high)], "qc": mask_qc(masked > 0, r.affine),
    }
    label = "SYNTHETIC DEMO — no patient data" if synthetic else "PRIVATE ANATOMICAL DATA — do not upload or share publicly"
    html = TEMPLATE.replace("__NOTICE__", escape(label)).replace("__DATA__", json.dumps(payload, allow_nan=False))
    with Path(output).open("x", encoding="utf-8") as stream:
        stream.write(html)
    return payload["qc"]


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'">
<title>DBS Native Connectome | mask review</title>
<style>
*{box-sizing:border-box}body{margin:0;background:white;color:#1a2630;font:15px/1.6 system-ui,sans-serif}
main{max-width:1280px;margin:auto;padding:28px}header{padding-bottom:20px;border-bottom:1px solid #ddd}
.eyebrow{font-size:12px;letter-spacing:.13em;color:#697680}h1{font-size:30px;margin:8px 0}h2{font-size:17px;margin:6px 0}
.warning{background:#fff6e8;padding:12px 16px;border-left:4px solid #bb6d18;margin:20px 0}
.controls{display:flex;gap:22px;align-items:center;flex-wrap:wrap;margin:20px 0}input{accent-color:#9b1e36}
.views{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px}canvas{width:100%;background:#080808;image-rendering:pixelated}
.caption{color:#586874;font-size:12px}label{display:block}output{font-variant-numeric:tabular-nums}
.slice{width:100%}.notes{margin-top:28px;display:grid;grid-template-columns:1fr 1fr;gap:32px}code{font-size:11px;word-break:break-all}
@media(max-width:800px){.views,.notes{grid-template-columns:1fr}main{padding:16px}}
</style></head><body><main>
<header><div class="eyebrow">RESEARCH PROTOTYPE / LOCAL QC</div><h1>Electrode-mask review</h1>
<div>Inspect spatial alignment and the entire signal void before approving downstream analysis.</div></header>
<div class="warning">__NOTICE__<br>This page records no approval. Same-grid checks do not establish anatomical registration.</div>
<div class="controls"><label><input id="showMask" type="checkbox" checked> Show mask</label>
<label>Mask opacity <input id="alpha" type="range" min="0" max="100" value="55"></label>
<label>Display window <input id="window" type="range" min="20" max="255" value="255"></label>
<button id="center" type="button">Center on mask</button></div>
<div class="views" id="views"></div>
<div class="notes"><section><h2>Required visual checks</h2><ol>
<li>CT–anatomical MRI–DWI registration and orientation</li><li>b0 void coverage at contacts and along the full shaft</li>
<li>Spurious disconnected components and non-lead dark structures</li><li>Native parcellation alignment in a separate atlas overlay</li>
<li>Registered union coverage at each longitudinal visit</li></ol>
<p class="caption">This viewer shows one b0 and mask, not CT, atlas or multiple timepoints. It does not complete all five checks.
Approval must be recorded separately by the reviewer using the CLI after inspecting the necessary images.</p></section>
<section><h2>Traceability</h2><div id="stats"></div><p class="caption">Mask SHA-256<br><code id="maskHash"></code></p>
<p class="caption">Reference SHA-256<br><code id="refHash"></code></p><p class="caption">Images are reoriented to RAS+ for display only.
Planes follow the image grid; an oblique acquisition is not displayed as resampled anatomical orthogonal planes.
Left is screen-left in the axial/coronal-like views. No connection to external services is made.</p></section></div>
</main><script>
const d=__DATA__;
const bytes=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
const im=bytes(d.image),mask=bytes(d.mask),[nx,ny,nz]=d.shape;
const idx=(x,y,z)=>x+nx*(y+ny*z);
const planes=[{name:'Sagittal-like',axis:0,w:ny,h:nz,labels:'P → A · I → S'},
{name:'Coronal-like',axis:1,w:nx,h:nz,labels:'L → R · I → S'},
{name:'Axial-like',axis:2,w:nx,h:ny,labels:'L → R · P → A'}];
const ui=planes.map((p,i)=>{const box=document.createElement('section');
box.innerHTML=`<h2>${p.name}</h2><canvas width="${p.w}" height="${p.h}" aria-label="${p.name} mask overlay"></canvas>
<label>Slice <output></output><input class="slice" type="range" min="0" max="${d.shape[p.axis]-1}" value="${Math.floor(d.shape[p.axis]/2)}"></label>
<div class="caption">${p.labels} · grid-axis ${p.axis+1}</div>`;
document.getElementById('views').append(box);return {...p,canvas:box.querySelector('canvas'),slider:box.querySelector('input'),output:box.querySelector('output')};});
function draw(){const opacity=Number(document.getElementById('alpha').value)/100,windowWidth=Number(document.getElementById('window').value),overlay=document.getElementById('showMask').checked;
for(const p of ui){const s=Number(p.slider.value),ctx=p.canvas.getContext('2d'),pic=ctx.createImageData(p.w,p.h);p.output.textContent=`${s} / ${d.shape[p.axis]-1}`;
for(let v=0;v<p.h;v++)for(let u=0;u<p.w;u++){let x,y,z;if(p.axis===0){x=s;y=u;z=p.h-1-v}else if(p.axis===1){x=u;y=s;z=p.h-1-v}else{x=u;y=p.h-1-v;z=s}
const j=idx(x,y,z),k=4*(u+p.w*v),c=Math.min(255,im[j]*255/windowWidth),a=overlay&&mask[j]?opacity:0;
pic.data[k]=c*(1-a)+255*a;pic.data[k+1]=c*(1-a)+30*a;pic.data[k+2]=c*(1-a)+50*a;pic.data[k+3]=255;}ctx.putImageData(pic,0,0);}}
function centerMask(){let x=0,y=0,z=0,n=0;for(let k=0;k<mask.length;k++)if(mask[k]){x+=k%nx;y+=Math.floor(k/nx)%ny;z+=Math.floor(k/(nx*ny));n++}
if(n){const c=[x/n,y/n,z/n];let best=null,dist=Infinity;for(let k=0;k<mask.length;k++)if(mask[k]){const v=[k%nx,Math.floor(k/nx)%ny,Math.floor(k/(nx*ny))],r=v.reduce((s,a,i)=>s+(a-c[i])**2,0);if(r<dist){dist=r;best=v}}ui.forEach(p=>p.slider.value=best[p.axis]);}draw();}
ui.forEach(p=>p.slider.addEventListener('input',draw));['showMask','alpha','window'].forEach(id=>document.getElementById(id).addEventListener('input',draw));
document.getElementById('center').addEventListener('click',centerMask);
document.getElementById('stats').textContent=`${d.shape.join(' × ')} voxels · mask ${d.qc.voxels} voxels (${d.qc.volume_mm3.toFixed(1)} mm³) · ${d.qc.components_6_connected} connected component(s)`;
document.getElementById('maskHash').textContent=d.mask_sha256;document.getElementById('refHash').textContent=d.reference_sha256;centerMask();
</script></body></html>'''
