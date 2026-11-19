import json
from pathlib import Path
import tempfile
import unittest

from cloud.comparison_metrics import score
from cloud.diagnostic_experiment import (
    build_random_arm, carve_train_split, checkpoint_metrics, dev_size_for_effect,
    encode_records, estimate_gpu_hours, minimum_detectable_effect, plan, power_report,
    refuse_paid, refuse_sealed_path, slice_dose, stage0, stage1, summarize,
)
from cloud.diagnostic_tiny import TinyTokenizer
from cloud.valuation_contract import validate_arm


ROOT = Path(__file__).resolve().parent
SMOKE = ROOT / 'diagnostics' / 'smoke.json'
GRANITE = ROOT / 'diagnostics' / 'granite_stage01.json'


def rows(count, start, split='train'):
    built = []
    for offset in range(count):
        gold = str(4 + offset)
        built.append({
            'id': f'gsm8k:{split}:{start + offset:05d}',
            'prompt': f'Item {start + offset} has {gold} units and a remainder.',
            'response': ('Add the units ' * (2 + offset % 3)) + f'#### {gold}',
            'gold_answer': gold,
            'source_split': split,
        })
    return built


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / 'runs' / 'test_diagnostic_experiment'
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_historical_dev200_is_underpowered_for_a_two_point_effect(self):
        effect = minimum_detectable_effect(200, 9 / 200)
        self.assertGreater(effect['mde_percentage_points'], 4.0)
        self.assertLess(effect['mde_percentage_points'], 4.4)
        self.assertGreater(effect['ci95_halfwidth_percentage_points'], 2.8)
        needed = dev_size_for_effect(0.02, 9 / 200)
        self.assertGreater(needed, 800)
        self.assertLess(needed, 1000)
        report = power_report(json.loads(GRANITE.read_text(encoding='utf-8')))
        self.assertTrue(report['underpowered_for_target'])
        self.assertGreaterEqual(report['suggested_development_questions'], 1000)
        self.assertGreater(report['questions_for_target_at_transfer_rate'], 1800)

    def test_carve_keeps_train_rows_disjoint_and_rejects_the_test_split(self):
        train = rows(30, 0)
        carved = carve_train_split(train, 10, 5, 11, exclude_ids={train[0]['id']})
        groups = [carved['development'], carved['pilot'], carved['train_pool']]
        ids = [row['id'] for group in groups for row in group]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(carved['development']), 10)
        self.assertEqual(len(carved['pilot']), 5)
        self.assertNotIn(train[0]['id'], ids)
        with self.assertRaises(ValueError):
            carve_train_split(rows(4, 0, split='test'), 1, 1, 1)
        with self.assertRaises(ValueError):
            refuse_sealed_path(self.root / 'official_test.jsonl')

    def test_doses_share_one_random_stream_and_match_tokens_and_updates(self):
        corpus = rows(20, 0)
        tokenizer = TinyTokenizer(row['prompt'] + ' ' + row['response'] for row in corpus)
        encoded = encode_records(corpus, tokenizer, prefix='Solve ')
        full = build_random_arm(encoded, 40, 7, 8)
        short = slice_dose(full, 2)
        validate_arm(full)
        validate_arm(short)
        self.assertEqual(full['token_budget'], 40)
        self.assertEqual(full['optimizer_updates'], 5)
        self.assertEqual(short['token_budget'], 16)
        self.assertEqual(short['optimizer_updates'], 2)
        self.assertEqual(sum(token != -100 for row in short['training'] for token in row['labels']), 16)
        self.assertTrue(all(abs(sum(window['loss_weight'] for window in update['microbatches']) - 1) < 1e-9
                            for update in short['updates']))
        long_end = {}
        for update in full['updates']:
            for window in update['microbatches']:
                long_end[window['id']] = window['answer_end']
        for update in short['updates']:
            for window in update['microbatches']:
                self.assertLessEqual(window['answer_end'], long_end[window['id']])
        self.assertTrue(set(row['id'] for row in short['training']).issubset(
            row['id'] for row in full['training']))

    def test_fit_and_fresh_metrics_stay_on_disjoint_questions(self):
        base = [
            {'id': 'd1', 'response': '#### 1', 'gold_answer': '1', 'truncated': False,
             'generated_tokens': 2, 'elapsed_seconds': 0.0},
            {'id': 'd2', 'response': '#### 9', 'gold_answer': '1', 'truncated': False,
             'generated_tokens': 4, 'elapsed_seconds': 0.0},
        ]
        fresh = [
            {'id': 'd1', 'response': '#### 9', 'gold_answer': '1', 'truncated': False,
             'generated_tokens': 3, 'elapsed_seconds': 0.0},
            {'id': 'd2', 'response': '#### 1', 'gold_answer': '1', 'truncated': True,
             'generated_tokens': 8, 'elapsed_seconds': 0.0},
        ]
        taught = [
            {'id': 't1', 'response': '#### 4', 'gold_answer': '4', 'truncated': False,
             'generated_tokens': 2, 'elapsed_seconds': 0.0},
        ]
        metrics = checkpoint_metrics(fresh, taught, base)
        self.assertEqual(metrics['taught_correct'], 1)
        self.assertEqual(metrics['fresh_correct'], 0)
        self.assertEqual(metrics['base_correct_count'], 1)
        self.assertEqual(metrics['retained_base_correct'], 0)
        self.assertEqual(metrics['retention_on_base_correct'], 0.0)
        self.assertEqual(metrics['truncated_but_numeric_match'], 1)
        self.assertTrue(score(fresh[1])['correct'] is False)
        with self.assertRaises(ValueError):
            checkpoint_metrics(fresh, [{'id': 'd1', 'response': '#### 1', 'gold_answer': '1',
                                        'truncated': False, 'generated_tokens': 1, 'elapsed_seconds': 0.0}], base)

    def test_paid_backends_and_the_test_split_are_refused(self):
        with self.assertRaises(RuntimeError):
            refuse_paid({'backend': 'modal', 'launch_modal': True})
        with self.assertRaises(RuntimeError):
            refuse_paid({'backend': 'tiny', 'launch_deepseek': True})
        config = json.loads(GRANITE.read_text(encoding='utf-8'))
        config['launch_modal'] = True
        with self.assertRaises(RuntimeError):
            plan(config, self.root / 'refused.json')
        self.assertFalse((self.root / 'refused.json').exists())

    def test_cpu_smoke_stage0_agrees_and_stage1_logs_the_curve(self):
        config = json.loads(SMOKE.read_text(encoding='utf-8'))
        stage0_path = self.root / 'stage0.json'
        stage1_path = self.root / 'stage1.json'
        summary_path = self.root / 'summary.json'
        first = stage0(config, stage0_path)
        self.assertTrue(first['reproducibility']['exact_agreement'])
        self.assertFalse(first['official_test_used'])
        self.assertTrue(first['power']['underpowered_for_target'])
        measured = stage1(config, stage1_path)
        self.assertEqual(measured['status'], 'measured')
        self.assertEqual(len(measured['runs']), 4)
        self.assertEqual({run['supervised_tokens'] for run in measured['runs']}, {16, 32})
        self.assertEqual({run['optimizer_updates'] for run in measured['runs']}, {2, 4})
        taught_ids = set()
        fresh_ids = set()
        for run in measured['runs']:
            self.assertEqual(len(run['train_losses']), run['optimizer_updates'])
            self.assertTrue(all(math_finite(loss) for loss in run['train_losses']))
            self.assertLess(run['train_losses'][-1], run['train_losses'][0])
            self.assertEqual(run['checkpoints'][0]['updates'], 0)
            self.assertEqual(run['checkpoints'][-1]['updates'], run['optimizer_updates'])
            self.assertIsNotNone(run['checkpoints'][-1]['heldout_teacher_nll'])
            self.assertGreater(run['checkpoints'][-1]['taught_count'], 0)
            taught_ids.update(row['id'] for row in run['taught_generations'])
            fresh_ids.update(row['id'] for row in run['dev_generations'])
            self.assertTrue(set(run['trained_ids']).isdisjoint(run['heldout_teacher_ids']))
        self.assertTrue(taught_ids.isdisjoint(fresh_ids))
        summary = summarize(stage1_path, bootstrap_resamples=200)
        self.assertEqual(len(summary['rows']), 4)
        self.assertIn('paired_bootstrap_95_interval_pp', summary['rows'][0]['versus_base'])
        self.assertEqual(summary['learning_rate_contrasts'], [])
        summary_path.write_text(json.dumps(summary), encoding='utf-8')
        self.assertFalse(json.loads(summary_path.read_text())['official_test_used'])

    def test_plan_names_the_granite_matrix_without_training(self):
        config = json.loads(GRANITE.read_text(encoding='utf-8'))
        written = plan(config, self.root / 'plan.json')
        self.assertEqual(len(written['matrix']['cells']), 12)
        self.assertEqual(written['matrix']['fixed_subset_supervised_tokens'], 262144)
        self.assertEqual(written['matrix']['lora']['target_modules'], ['q_proj', 'v_proj'])
        self.assertIn('recommended_hours_before_stage2', written['gpu_hours'])
        self.assertGreater(written['gpu_hours']['stage1_hours_if_every_checkpoint_uses_suggested_dev'],
                           written['gpu_hours']['stage1_hours_sweep_on_eval_slice'])
        self.assertEqual(written['reference_rows']['rows'][0]['correct'], 194)
        self.assertFalse(written['reference_rows']['comparable_to_a_new_split'])
        self.assertEqual(estimate_gpu_hours(config)['cells'], 12)
        stage0_plan = stage0(config, self.root / 'stage0_plan.json')
        self.assertEqual(stage0_plan['reproducibility']['status'], 'not_run')
        stage1_plan = stage1(config, self.root / 'stage1_plan.json')
        self.assertEqual(stage1_plan['runs'], [])
        self.assertEqual(stage1_plan['status'], 'plan_only')


def math_finite(value):
    return isinstance(value, float) and value == value and value not in (float('inf'), float('-inf'))


if __name__ == '__main__':
    unittest.main()
