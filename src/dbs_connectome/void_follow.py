"""Frozen study void-follow candidate algorithm, separate from CT/MRI fusion.

This reproduces the study's filled-brain-exit variant on reviewed 2-mm images.
It is a candidate generator, not physical-contact localization. Later manual
repairs and longitudinal mask unions are separate provenance-bearing operations.
"""
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes

from .masks import mask_qc, study_dilate
from .provenance import (assert_inputs_unchanged, digest, freeze_inputs, image3d,
                         record, require_separate_output, same_grid, write_json)
from .qc import browser_report


def _paint(array, centers, affine, voxels, value):
    inverse=np.linalg.inv(affine)
    radius=3.5
    for p in centers:
        vp=inverse[:3,:3]@p+inverse[:3,3]
        extent=int(np.ceil(radius/np.mean(voxels)))+1
        low=[int(max(0,vp[i]-extent)) for i in range(3)]
        high=[int(min(array.shape[i],vp[i]+extent+1)) for i in range(3)]
        if any(a>=b for a,b in zip(low,high)):continue
        grid=np.stack(np.mgrid[low[0]:high[0],low[1]:high[1],low[2]:high[2]],-1).astype(float)
        world=grid@affine[:3,:3].T+affine[:3,3]
        array[low[0]:high[0],low[1]:high[1],low[2]:high[2]][np.linalg.norm(world-p,axis=-1)<=radius]=value


def build(b0, affine, seed_labels, preoperative=False):
    b=np.asarray(b0,dtype=float);ex=np.asarray(seed_labels)
    if b.ndim!=3 or b.shape!=ex.shape or not np.isfinite(b).all() or not np.isin(ex,[0,1,2]).all():
        raise ValueError("Finite 3D b0 and same-grid bilateral seed labels 0/1/2 required")
    if not (b>0).any() or any(np.count_nonzero(ex==h)<3 for h in (1,2)):
        raise ValueError("Both registered shaft seeds and positive brain signal are required")
    # This exact legacy deep-seed rule is voxel-z based. Refuse upside-down grids.
    if nib.aff2axcodes(affine)[2]!='S':
        raise ValueError("Study replay requires increasing voxel-z toward superior; reorient explicitly")
    study_dilate(ex>0,affine,0)
    inverse=np.linalg.inv(affine);vox=np.sqrt((affine[:3,:3]**2).sum(0))
    median=np.median(b[b>0]);dark=.55*median;brain=binary_fill_holes(b>0)
    labeled=np.zeros(b.shape,np.uint8);sides=labeled.copy();info={}
    for h,(contact,shaft,side) in [(1,(1,2,1)),(2,(3,4,2))]:
        idx=np.argwhere(ex==h);world=idx@affine[:3,:3].T+affine[:3,3]
        direction=np.linalg.svd(world-world.mean(0))[2][0]
        direction/=np.linalg.norm(direction)
        if direction[2]<0:direction=-direction
        deep=idx[idx[:,2]<=np.percentile(idx[:,2],30)].mean(0)
        center=deep.copy();void=np.zeros((0,3),int)
        if not preoperative:
            radius=int(18/np.mean(vox))
            low=[int(max(0,deep[i]-radius)) for i in range(3)]
            high=[int(min(b.shape[i],deep[i]+radius+1)) for i in range(3)]
            box=b[low[0]:high[0],low[1]:high[1],low[2]:high[2]]
            candidates=np.argwhere(box<.4*median)+np.array(low)
            if len(candidates):candidates=candidates[np.linalg.norm((candidates-deep)*vox,axis=1)<=18]
            if len(candidates)>=3:void=candidates;center=candidates.mean(0)
        center_world=affine[:3,:3]@center+affine[:3,3]
        path=world
        if not preoperative:
            position=center_world.copy();heading=direction.copy();path=[position.copy()];gap=0
            for _ in range(95):
                predicted=position+heading
                v=(inverse[:3,:3]@predicted+inverse[:3,3]).astype(int)
                if not all(0<=v[i]<b.shape[i] for i in range(3)) or not brain[tuple(v)]:break
                search=int(np.ceil(5/np.mean(vox)))
                # Clipped slices avoid an upstream boundary-indexing error; interior values unchanged.
                sl=tuple(slice(max(0,v[i]-search),min(b.shape[i],v[i]+search+1)) for i in range(3))
                grid=np.stack(np.mgrid[sl[0],sl[1],sl[2]],-1).reshape(-1,3)
                ras=grid@affine[:3,:3].T+affine[:3,3];intensities=b[sl].ravel()
                selected=((ras-position)@heading>0)&(np.linalg.norm(ras-predicted,axis=1)<=5)&(intensities<dark)
                if selected.sum()>=2:
                    weights=np.maximum(dark-intensities[selected],0)
                    if weights.sum()>0:
                        center_ahead=(ras[selected]*weights[:,None]).sum(0)/weights.sum()
                        proposed=center_ahead-position;length=np.linalg.norm(proposed)
                        if length>1e-3 and proposed/length@heading>.1:
                            heading=.6*heading+.4*proposed/length;heading/=np.linalg.norm(heading);gap=0
                else:gap+=1
                position=position+heading;path.append(position.copy())
                if gap>4:break
            path=np.array(path)
        _paint(labeled,path,affine,vox,shaft);_paint(sides,path,affine,vox,side)
        contacts=void@affine[:3,:3].T+affine[:3,3] if len(void)>=3 else np.array([center_world])
        _paint(labeled,contacts,affine,vox,contact);_paint(sides,contacts,affine,vox,side)
        info[str(h)]={'steps':int(len(path)),'end_to_end_length_mm':float(np.linalg.norm(path[-1]-path[0])) if len(path)>1 else 0.,
                      'seed_voxels':int(len(idx)),'void_voxels':int(len(void))}
    return labeled,sides,sides>0,info


def run_void_follow(reference,seed_mask,spec,out):
    out=Path(out)
    for path in (reference,seed_mask):require_separate_output(out,Path(path).parent)
    if spec.get('reference_sha256')!=digest(reference) or spec.get('seed_sha256')!=digest(seed_mask):
        raise ValueError('Seed specification belongs to different input images')
    if spec.get('phase') not in ('pre','post'):
        raise ValueError('Explicit pre (virtual) or post (void-follow) phase required')
    before=freeze_inputs([reference,seed_mask])
    b,s=image3d(reference),image3d(seed_mask);same_grid(b,s)
    labeled,sides,mask,info=build(np.asanyarray(b.dataobj),b.affine,np.asanyarray(s.dataobj),spec['phase']=='pre')
    dilated=study_dilate(mask,b.affine,spec.get('dilation_passes',1))
    paths=[]
    for name,data in [('labels',labeled),('sides',sides),('undilated',mask),('mask_candidate',dilated)]:
        p=out/(name+'.nii.gz');nib.save(nib.Nifti1Image(data.astype(np.uint8),b.affine),p);paths.append(p)
    browser_report(reference,paths[-1],out/'mask_review.html')
    write_json(out/'void_follow_result.json',{'algorithm':'study_void_follow_filled_brain_exit','status':'needs_qc',
        'phase':spec['phase'],'dilation_passes':spec.get('dilation_passes',1),'hemispheres':info,
        'mask_qc':mask_qc(dilated,b.affine),'automatic_qc_approval':False,
        'warning':'Legacy voxel-z seed rule and global b0 threshold. Manual repairs and longitudinal unions are not inferred. Contact labels identify void regions, not physical metal contacts.'})
    assert_inputs_unchanged(before)
    record(out,'study-mask',[reference,seed_mask],spec,[*paths,out/'mask_review.html',out/'void_follow_result.json'])
