import unittest
import numpy as np
from dbs_connectome.void_follow import build


class VoidTests(unittest.TestCase):
    def phantom(self):
        b=np.ones((36,36,36))*100;b[:2]=0;b[-2:]=0;b[:,:2]=0;b[:,-2:]=0;b[:,:,:2]=0;b[:,:,-2:]=0
        seed=np.zeros(b.shape,np.uint8);seed[12,18,8:29]=1;seed[24,18,8:29]=2
        b[12,18,8:29]=0;b[24,18,8:29]=0
        return b,np.diag([2.,2.,2.,1.]),seed

    def test_virtual_mask_is_intensity_independent(self):
        b,a,s=self.phantom();x=build(b,a,s,True);y=build(np.ones_like(b),a,s,True)
        np.testing.assert_array_equal(x[2],y[2]);self.assertTrue(x[2].any())

    def test_zero_void_does_not_mean_outside_brain(self):
        b,a,s=self.phantom();lab,sides,mask,info=build(b,a,s,False)
        self.assertGreater(info['1']['steps'],2)
        self.assertEqual(set(np.unique(sides)),{0,1,2})

    def test_missing_lead_bad_grid_refused(self):
        b,a,s=self.phantom();s[s==2]=0
        with self.assertRaises(ValueError):build(b,a,s)
        b,a,s=self.phantom();a[2,2]=-2
        with self.assertRaises(ValueError):build(b,a,s)


if __name__=='__main__':unittest.main()
