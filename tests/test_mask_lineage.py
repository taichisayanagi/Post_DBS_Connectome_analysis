from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from dbs_connectome.mask_lineage import apply_repair, repair_mask, shared_mask_plan
from dbs_connectome.provenance import digest, write_json, read_json


class MaskLineageTests(unittest.TestCase):
    def fixture(self, root):
        source=root/'inputs';source.mkdir();output=root/'outputs';output.mkdir()
        mask=np.zeros((8,8,8),np.uint8);mask[2,2,2:6]=1
        add=np.zeros_like(mask);add[2,2,6]=1
        remove=np.zeros_like(mask);remove[2,2,2]=1
        paths={}
        for name,data in [('reference',np.ones_like(mask)),('mask',mask),('add',add),('remove',remove)]:
            path=source/(name+'.nii.gz');nib.save(nib.Nifti1Image(data,np.diag([2.,2.,2.,1.])),path);paths[name]=str(path)
        return paths,output

    def test_overlay_changes_and_immutable_sources(self):
        with tempfile.TemporaryDirectory() as td:
            paths,out=self.fixture(Path(td));before={p:digest(p) for p in paths.values()}
            recipe=Path(td)/'inputs/recipe.json';spec={**paths,'reason':'synthetic correction'}
            spec.update({k+'_sha256':digest(v) for k,v in paths.items()});write_json(recipe,spec)
            repair_mask(recipe,out)
            result=read_json(out/'mask_lineage.json')
            self.assertEqual(result['changes'],{'added_voxels':1,'removed_voxels':1})
            self.assertEqual(result['status'],'needs_qc')
            self.assertFalse(result['automatic_qc_approval'])
            self.assertEqual(before,{p:digest(p) for p in paths.values()})

    def test_conflict_and_wrong_parent_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p,out=self.fixture(Path(td))
            with self.assertRaisesRegex(ValueError,'conflict'):apply_repair(p['reference'],p['mask'],p['add'],p['add'])
            with self.assertRaisesRegex(ValueError,'absent'):apply_repair(p['reference'],p['mask'],None,p['add'])
            with self.assertRaisesRegex(ValueError,'already'):apply_repair(p['reference'],p['mask'],p['remove'])

    def test_forward_inverse_union_registration_is_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            p,out=self.fixture(Path(td))
            commands=shared_mask_plan(p['reference'],p['reference'],p['mask'],p['mask'],out,12)
            self.assertEqual(commands[0][commands[0].index('-t')+1],'a')
            self.assertIn('NearestNeighbor',commands[1])
            self.assertEqual(commands[2][-1],f'[{out}/post1_to_post2_0GenericAffine.mat,1]')
            self.assertFalse(any('SyN[' in word for c in commands for word in c))


if __name__=='__main__':unittest.main()
