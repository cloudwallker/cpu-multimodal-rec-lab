import unittest


class CompletionGateTests(unittest.TestCase):
    def api(self):
        try:
            from mmrec_lab.completion import audit_development
        except ImportError:
            self.fail('Pre-test development audit is missing')
        return audit_development

    def runs(self):
        return [{'run_id':f'candidate_{i}', 'seed':101, 'test':None,
                 'epochs_completed':70, 'stop_reason':'validation_patience',
                 'config':{'epochs':200,'num_threads':2},
                 'resources':{'total_seconds':700.,'peak_rss_bytes':800_000_000,'device':'cpu'}} for i in range(16)]

    def test_all_candidates_complete_and_cpu_resource_gate(self):
        audit=self.api()
        result=audit(self.runs(),16_000_000_000)
        self.assertEqual(result['decision'],'ready_for_main')
        self.assertEqual(result['candidates'],16)
        self.assertEqual(result['epoch_limit_runs'],0)

    def test_missing_candidate_and_accidental_test_are_rejected(self):
        audit=self.api()
        with self.assertRaisesRegex(ValueError,'16'):
            audit(self.runs()[:15],16_000_000_000)
        runs=self.runs(); runs[0]['test']={'score':.5}
        with self.assertRaisesRegex(ValueError,'test'):
            audit(runs,16_000_000_000)

    def test_majority_hitting_cap_stops_before_main(self):
        audit=self.api(); runs=self.runs()
        for r in runs[:9]:
            r['stop_reason']='epoch_limit'; r['epochs_completed']=200
        with self.assertRaisesRegex(ValueError,'epoch'):
            audit(runs,16_000_000_000)

    def test_resource_excess_and_invalid_device_stop_before_main(self):
        audit=self.api()
        for key,value in [('total_seconds',1801.),('peak_rss_bytes',8_000_000_001),('device','cuda')]:
            runs=self.runs(); runs[0]['resources'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                audit(runs,16_000_000_000)
