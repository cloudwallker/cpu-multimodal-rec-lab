import unittest
import tempfile
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix


class StudyProtocolTests(unittest.TestCase):
    def api(self):
        try:
            from mmrec_lab.study import experiment_matrix, select_parameter
        except ImportError:
            self.fail('Study protocol API not implemented')
        return experiment_matrix,select_parameter

    def test_matrix_and_pairing(self):
        matrix,_=self.api()
        runs=matrix()
        self.assertEqual(len(runs),45)
        self.assertEqual(len({r['run_id'] for r in runs}),45)
        for seed,mask in [(201,301),(202,302),(203,303)]:
            pair=[r for r in runs if r['seed']==seed]
            self.assertEqual(len(pair),15)
            self.assertEqual({r['mask_seed'] for r in pair if r['scenario']!='clean'},{mask})
            for scenario in ['uniform30','tail30']:
                self.assertEqual(len([r for r in pair if r['scenario']==scenario]),6)

    def test_cross_scenario_selection_and_tie(self):
        _,select=self.api()
        rows=[{'value':v,'scenario':s,'recall':score} for v,s,score in [
            (1,'uniform30',.2),(1,'tail30',.4),
            (4,'uniform30',.3),(4,'tail30',.3),
            (16,'uniform30',.1),(16,'tail30',.1)]]
        self.assertEqual(select(rows)['value'],1)
        rows[0]['recall']=.1
        self.assertEqual(select(rows)['value'],4)

    def test_missing_candidate_scenario_is_error(self):
        _,select=self.api()
        with self.assertRaises(ValueError):
            select([{'value':1,'scenario':'tail30','recall':.4}])

    def test_condition_reuse_preserves_timing_and_rejects_change(self):
        from mmrec_lab.study import prepare_condition
        from mmrec_lab.io import read_json
        ds={'n_items':10,'vision':np.ones((10,2),np.float32),'text':np.ones((10,1),np.float32),
            'tail':np.array([False]*5+[True]*5)}
        condition={'run_id':'one','scenario':'tail30','method':'support_mix','modality':'both','seed':201,'mask_seed':301}
        graph=csr_matrix(np.ones((10,10))-np.eye(10))
        with tempfile.TemporaryDirectory() as path:
            prepare_condition(ds,graph,condition,path,tau=4)
            original=read_json(Path(path)/'condition.json')
            prepare_condition(ds,graph,condition,path,tau=4)
            self.assertEqual(read_json(Path(path)/'condition.json'),original)
            with self.assertRaisesRegex(ValueError,'condition'):
                prepare_condition(ds,graph,condition,path,tau=16)


if __name__=='__main__':
    unittest.main()
