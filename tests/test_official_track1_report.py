"""주최측 evaluation-package로 official_track1_report 확인."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from official_track1_report import score_split, publish_comparison, group_analysis, OFFICIAL_COMMIT
from experiment_dashboard import tag

PACKAGE = Path(os.environ.get('ASVSPOOF5_OFFICIAL_REPO', '/private/tmp/asvspoof5-official'))/'evaluation-package'


class OfficialTrack1Tests(unittest.TestCase):
    @unittest.skipUnless((PACKAGE/'evaluation.py').exists(), 'Install the pinned official package first')
    def test_real_cli_full_dev_id_alignment_and_metrics(self):
        # 완전히 분리되는 합성 score, ID 순서는 역순
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/'scores').mkdir(); (output/'protocols').mkdir()
            score = output/'scores'/f'dev_{tag("a1c1")}.tsv'
            protocol = output/'protocols/ASVspoof5.dev.track_1.tsv'
            count = 140950
            with score.open('w') as s, protocol.open('w') as p:
                for i in range(count):
                    label = i % 2
                    key = 'spoof' if label else 'bonafide'
                    p.write(f'S{i} D_{i} M - - - AC A {key} -\n')
                for i in reversed(range(count)):
                    s.write(f'D_{i}\t{-2 if i%2 else 2}\t{i%2}\n')
            score.with_name(score.stem+'_summary.json').write_text(json.dumps(dict(metric_mode='official',sliding=False)))
            report = score_split(output, 'a1c1', 'dev', PACKAGE)
            self.assertEqual(report['rows'], count)
            self.assertEqual(report['eer_pct'], 0.)
            self.assertEqual(report['min_dcf'], 0.)
            self.assertEqual(report['act_dcf'], 0.)
            self.assertEqual(report['official_commit'], OFFICIAL_COMMIT)
            self.assertTrue((output/'scores/official/a1c1/dev/official_stdout.txt').exists())
            self.assertEqual(score_split(output, 'a1c1', 'dev', PACKAGE)['score_sha256'], report['score_sha256'])

    def test_dev_is_not_inserted_as_eval_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); (output/'reports').mkdir()
            rows = publish_comparison(output, [])
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(r['origin'] == 'published challenge ensemble' for r in rows))
            text = (output/'reports/official_eval_comparison.md').read_text()
            self.assertIn('| A1C1 | WavLM-Base | BiLSTM + LCNN | 1 | pending', text)

    @unittest.skipUnless((PACKAGE/'evaluation.py').exists(), 'Install the pinned official package first')
    def test_group_analysis_uses_correct_reference_populations(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ['scores','protocols','reports','figures']: (output/name).mkdir()
            with (output/'scores'/f'eval_{tag("a1c1")}.tsv').open('w') as s, (output/'protocols/ASVspoof5.eval.track_1.tsv').open('w') as p:
                for i in range(6):
                    label = 0 if i%3==0 else 1
                    codec = 'C01' if i<3 else 'C02'
                    attack = 'bonafide' if not label else ('A17' if i%3==1 else 'A18')
                    key = 'spoof' if label else 'bonafide'
                    s.write(f'E_{i}\t{-2 if label else 2}\t{label}\n')
                    p.write(f'S E_{i} M {codec} - - AC {attack} {key} -\n')
            group_analysis(output, dict(combo='a1c1',score_sha256='synthetic-fixture'), PACKAGE)
            rows = json.loads((output/'reports/eval_groups_a1c1.json').read_text())['rows']
            for row in rows:
                self.assertEqual(row['bonafide'], 2 if row['group_type']=='attack' else 1)
                self.assertEqual(row['spoof'], 2)
                self.assertEqual(row['eer_pct'], 0.)
            self.assertTrue((output/'figures/eval_attack_a1c1.png').exists())
            self.assertTrue((output/'figures/eval_codec_a1c1.png').exists())


if __name__ == '__main__': unittest.main()
