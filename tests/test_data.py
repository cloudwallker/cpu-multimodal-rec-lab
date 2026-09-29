import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def make_raw(path):
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for u in range(8):
        for item in (u, (u + 1) % 8):
            rows.append((u, item, 5, 0, 0))
        rows.append((u, (u + 2) % 8, 5, 1, 1))
        rows.append((u, (u + 3) % 8, 5, 2, 2))
    pd.DataFrame(rows, columns=['userID','itemID','rating','timestamp','x_label']).to_csv(path/'baby.inter', sep='\t', index=False)
    np.save(path/'image_feat.npy', np.arange(24,dtype=np.float64).reshape(8,3)+1)
    np.save(path/'text_feat.npy', np.arange(16,dtype=np.float32).reshape(8,2)+1)
    pd.DataFrame({'asin':['product'+str(i) for i in range(8)],'itemID':range(8)}).to_csv(path/'i_id_mapping.csv', sep='\t',index=False)
    pd.DataFrame({'user_id':['person'+str(i) for i in range(8)],'userID':range(8)}).to_csv(path/'u_id_mapping.csv', sep='\t',index=False)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = self.root/'raw'
        make_raw(self.raw)

    def api(self):
        try:
            from mmrec_lab.data import prepare_dataset, load_dataset
        except ImportError:
            self.fail('Dataset preparation API is not implemented')
        return prepare_dataset, load_dataset

    def test_original_split_and_feature_row_mapping(self):
        prepare, load = self.api()
        prepare(self.raw, self.root/'out', target_items=8, seed=20260929)
        ds = load(self.root/'out')
        self.assertEqual(ds['train'].shape,(16,2))
        self.assertEqual(ds['valid'].shape,(8,2))
        self.assertEqual(ds['test'].shape,(8,2))
        self.assertEqual(ds['vision'].dtype,np.float32)
        np.testing.assert_array_equal(ds['vision'][:,0],1+3*ds['item_ids'])
        np.testing.assert_array_equal(np.flatnonzero(ds['tail']),[4,5,6,7])
        self.assertEqual(ds['manifest']['split_labels'], {'train':0,'valid':1,'test':2})
        self.assertFalse(ds['manifest']['native_image_availability_verified'])

    def test_cross_split_duplicate_rejected(self):
        prepare,_=self.api()
        table=pd.read_csv(self.raw/'baby.inter',sep='\t')
        duplicate=table.iloc[0].copy(); duplicate['x_label']=2
        pd.concat([table,duplicate.to_frame().T]).to_csv(self.raw/'baby.inter',sep='\t',index=False)
        with self.assertRaisesRegex(ValueError,'duplicate|overlap'):
            prepare(self.raw,self.root/'out',8,20260929)

    def test_invalid_feature_or_mapping_rejected(self):
        prepare,_=self.api()
        feature=np.load(self.raw/'image_feat.npy'); feature[0,0]=np.nan
        np.save(self.raw/'image_feat.npy',feature)
        with self.assertRaisesRegex(ValueError,'finite'):
            prepare(self.raw,self.root/'out',8,20260929)
        make_raw(self.raw)
        mapping=pd.read_csv(self.raw/'i_id_mapping.csv',sep='\t'); mapping.loc[7,'itemID']=6
        mapping.to_csv(self.raw/'i_id_mapping.csv',sep='\t',index=False)
        with self.assertRaisesRegex(ValueError,'mapping'):
            prepare(self.raw,self.root/'out',8,20260929)

    def test_hidden_evaluation_values_cannot_change_subset(self):
        prepare,load=self.api()
        prepare(self.raw,self.root/'a',8,20260929)
        table=pd.read_csv(self.raw/'baby.inter',sep='\t')
        table.loc[table.x_label>0,'rating']=1
        table.to_csv(self.raw/'baby.inter',sep='\t',index=False)
        prepare(self.raw,self.root/'b',8,20260929)
        a,b=load(self.root/'a'),load(self.root/'b')
        for key in ['item_ids','user_ids','train','tail','vision','text']:
            np.testing.assert_array_equal(a[key],b[key])

    def test_prepared_artifact_tampering_detected(self):
        prepare,load=self.api()
        prepare(self.raw,self.root/'out',8,20260929)
        file=self.root/'out'/'dataset.npz'
        with file.open('ab') as stream: stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'hash|checksum'):
            load(self.root/'out')

    def test_mismatched_modality_rows_rejected_before_broadcasting(self):
        prepare,_=self.api()
        np.save(self.raw/'text_feat.npy',np.ones((7,2),dtype=np.float32))
        with self.assertRaisesRegex(ValueError,'mapping'):
            prepare(self.raw,self.root/'out',8,20260929)

    def test_manifest_metadata_tampering_detected(self):
        import json
        prepare,load=self.api()
        prepare(self.raw,self.root/'out',8,20260929)
        path=self.root/'out'/'manifest.json'
        manifest=json.loads(path.read_text(encoding='utf-8'))
        manifest['subset_seed']=9
        path.write_text(json.dumps(manifest),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'manifest'):
            load(self.root/'out')


if __name__=='__main__':
    unittest.main()
