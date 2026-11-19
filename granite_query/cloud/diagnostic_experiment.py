"""Unpaid Stage 0 / Stage 1 diagnostic for the pinned Granite recipe.

Stage 0 reports base-eval agreement and the minimum paired effect the dev set
can detect. Stage 1 sweeps learning rate and update count on one fixed random
subset, reusing ``student_valuation._exposure`` so every dose is a prefix of
the same supervised-token stream. The default backend is a tiny CPU bigram.
``backend: plan`` writes the Granite matrix and does not train. Modal and
DeepSeek launches are refused, and the official GSM8K test split is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
for entry in (str(PARENT), str(ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from cloud import student_valuation
from cloud.comparison_metrics import score, summarize as summarize_generations
from cloud.diagnostic_tiny import TinyBigramLoRA, TinyTokenizer
from cloud.evaluation_stats import paired
from cloud.valuation_contract import validate_arm

SEALED_NAMES = {
    'test.jsonl', 'test.json', 'official_test.jsonl', 'official_test.json', 'test_sealed.jsonl',
}
# Measured worker seconds from the completed human-reference dev200 study.
# The 64-update figure includes that arm's own evaluation, so it is an upper
# bound on training rather than a separated train timer.
EVAL_200_WORKER_SECONDS = 592.229
TRAIN_SECONDS_PER_UPDATE = 3.0
PUBLISHED_DEV200 = {
    'question_set': 'human_reference_train_development200',
    'protocol': 'nonthinking greedy, 512 new tokens, truncated generations scored incorrect',
    'comparable_to_a_new_split': False,
    'rows': [
        {'name': 'published_deepseek', 'correct': 194, 'n': 200, 'truncated': 0},
        {'name': 'published_untouched_granite', 'correct': 179, 'n': 200, 'truncated': 8},
        {'name': 'published_random_65536', 'correct': 174, 'n': 200, 'truncated': 2,
         'supervised_tokens': 65536, 'optimizer_updates': 64},
        {'name': 'published_skill_65536', 'correct': 171, 'n': 200, 'truncated': 2,
         'supervised_tokens': 65536, 'optimizer_updates': 64},
    ],
}


def refuse_paid(config):
    """Stop before any provider call. Planning and the CPU smoke stay allowed."""
    if config.get('launch_modal') or config.get('launch_deepseek') or config.get('allow_official_test'):
        raise RuntimeError('This runner does not launch Modal or DeepSeek, and it does not open the official test split')
    backend = config.get('backend', 'tiny')
    if backend not in ('tiny', 'plan'):
        raise RuntimeError('Unknown backend. Use tiny for the CPU smoke or plan for the unpaid Granite matrix')
    return backend


def refuse_sealed_path(path):
    name = Path(path).name.casefold()
    if name in SEALED_NAMES or 'official_test' in name or name.startswith('test_sealed'):
        raise ValueError('Official GSM8K test files stay sealed: ' + name)


def refuse_sealed_rows(rows):
    for row in rows:
        split = str(row.get('source_split', '')).casefold()
        identifier = str(row.get('id', ''))
        if split == 'test' or ':test:' in identifier.casefold() or identifier.casefold().startswith('gsm8k:test'):
            raise ValueError('Official GSM8K test rows cannot enter carving, training, or evaluation')


def _z(probability):
    return statistics.NormalDist().inv_cdf(probability)


def minimum_detectable_effect(n, discordant_rate, *, alpha=0.05, power=0.8):
    """Smallest paired accuracy gap detectable at the given power.

    Under a paired score in {-1, 0, +1}, the null variance of the mean
    difference is ``discordant_rate / n``. Training-seed variance is not in
    this number.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError('Dev size must be an integer of at least 2')
    if not 0 < float(discordant_rate) <= 1:
        raise ValueError('Discordant rate must be in (0, 1]')
    z_effect = _z(1 - alpha / 2) + _z(power)
    standard_error = math.sqrt(float(discordant_rate) / n)
    return {
        'n': n,
        'discordant_rate': float(discordant_rate),
        'alpha': alpha,
        'power': power,
        'standard_error_proportion': standard_error,
        'mde_percentage_points': 100 * z_effect * standard_error,
        'ci95_halfwidth_percentage_points': 100 * _z(1 - alpha / 2) * standard_error,
        'covers': 'question resampling at a fixed training seed',
    }


def dev_size_for_effect(effect, discordant_rate, *, alpha=0.05, power=0.8):
    if not 0 < float(effect) < 1:
        raise ValueError('Target effect must be a proportion in (0, 1)')
    z_effect = _z(1 - alpha / 2) + _z(power)
    return math.ceil(float(discordant_rate) * (z_effect / float(effect)) ** 2)


def power_report(config):
    """MDE at the historical 200 and at the discordant rates those 200 produced."""
    historical_n = int(config.get('historical_dev_questions', 200))
    target = float(config.get('target_effect', 0.02))
    selector_rate = 9 / 200  # skill vs random: 3 corrected + 6 regressed
    transfer_rate = 19 / 200  # random vs base: 7 corrected + 12 regressed
    suggested = max(int(config.get('recommended_dev_questions', 1000)),
                    dev_size_for_effect(target, selector_rate))
    return {
        'historical_dev_questions': historical_n,
        'target_effect_percentage_points': 100 * target,
        'selector_contrast_prior': minimum_detectable_effect(historical_n, selector_rate),
        'transfer_contrast_prior': minimum_detectable_effect(historical_n, transfer_rate),
        'questions_for_target_at_selector_rate': dev_size_for_effect(target, selector_rate),
        'questions_for_target_at_transfer_rate': dev_size_for_effect(target, transfer_rate),
        'suggested_development_questions': suggested,
        'suggested_pilot_questions': int(config.get('pilot_questions', 200)),
        'underpowered_for_target': minimum_detectable_effect(historical_n, selector_rate)['mde_percentage_points'] > 100 * target,
        'source_of_rates': 'progress/HUMAN_STUDY_RESULTS.md paired counts on the sealed-from-training 200',
    }


def carve_train_split(rows, dev_size, pilot_size, seed, exclude_ids=()):
    """Carve disjoint development and selector-pilot sets from GSM8K train rows."""
    refuse_sealed_rows(rows)
    excluded = set(exclude_ids)
    pool = []
    for row in rows:
        if row.get('source_split') != 'train':
            raise ValueError('Only source_split=train rows can be carved')
        if row['id'] in excluded:
            continue
        pool.append(row)
    if len({row['id'] for row in pool}) != len(pool):
        raise ValueError('Train ids must be unique')
    if dev_size <= 0 or pilot_size <= 0 or len(pool) < dev_size + pilot_size:
        raise ValueError('Train pool is smaller than the requested development and pilot sets')
    order = sorted(pool, key=lambda row: row['id'])
    random.Random(seed).shuffle(order)
    development = order[:dev_size]
    pilot = order[dev_size:dev_size + pilot_size]
    remainder = order[dev_size + pilot_size:]
    groups = (development, pilot, remainder)
    if len({row['id'] for group in groups for row in group}) != len(pool):
        raise AssertionError('Carved splits overlap')
    return {'development': development, 'pilot': pilot, 'train_pool': remainder, 'seed': seed}


def load_train_rows(path):
    refuse_sealed_path(path)
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    refuse_sealed_rows(rows)
    return rows


def encode_records(rows, tokenizer, *, prefix):
    refuse_sealed_rows(rows)
    encoded = []
    for row in rows:
        prefix_ids = tokenizer.encode(prefix + row['prompt'])
        answer_ids = tokenizer.encode(row['response']) + [tokenizer.eos_id]
        if not prefix_ids or len(answer_ids) < 2:
            raise ValueError('Each smoke example needs a prefix and an answer')
        encoded.append({
            'id': row['id'], 'prompt': row['prompt'], 'response': row['response'],
            'gold_answer': str(row['gold_answer']), 'source_split': row.get('source_split', 'train'),
            'skills': ['general_arithmetic'],
            'raw_response_sha256': hashlib.sha256(row['response'].encode()).hexdigest(),
            'prefix_ids': prefix_ids, 'answer_ids': answer_ids,
            'response_tokens': len(answer_ids),
        })
    return encoded


def build_random_arm(encoded_rows, budget, data_seed, tokens_per_update):
    """One seeded random subset, then the repository's exact exposure contract."""
    if budget <= 0 or tokens_per_update <= 0 or budget % tokens_per_update:
        raise ValueError('Budget must be a positive multiple of tokens_per_update')
    if sum(row['response_tokens'] for row in encoded_rows) < budget:
        raise ValueError('Pool cannot fill this budget without repeating examples')
    permutation = sorted(encoded_rows, key=lambda row: row['id'])
    random.Random(data_seed).shuffle(permutation)
    selected, total = [], 0
    for row in permutation:
        selected.append(row)
        total += row['response_tokens']
        if total >= budget:
            break
    training, updates = student_valuation._exposure(selected, budget, data_seed, tokens_per_update)
    arm = {'schema_version': 1, 'arm': 'random', 'token_budget': budget,
           'tokens_per_update': tokens_per_update, 'optimizer_updates': len(updates),
           'training': training, 'updates': updates}
    validate_arm(arm)
    return arm


def slice_dose(arm, n_updates):
    """Keep the first ``n_updates`` of an already validated exposure."""
    if not isinstance(n_updates, int) or not 0 < n_updates <= len(arm['updates']):
        raise ValueError('Dose must keep a positive prefix of the fixed update stream')
    per_update = arm['tokens_per_update']
    consumed = {}
    updates = []
    for index, update in enumerate(arm['updates'][:n_updates]):
        current = json.loads(json.dumps(update))
        current['index'] = index
        for window in current['microbatches']:
            consumed[window['id']] = window['answer_end']
        updates.append(current)
    training = []
    by_id = {row['id']: row for row in arm['training']}
    for identifier, count in consumed.items():
        row = json.loads(json.dumps(by_id[identifier]))
        prefix = row['prefix_tokens']
        row['supervised_tokens'] = count
        row['partial_supervised_tail'] = count < row['full_response_tokens']
        row['labels'] = ([-100] * prefix + row['input_ids'][prefix:prefix + count]
                         + [-100] * (len(row['input_ids']) - prefix - count))
        training.append(row)
    sliced = {'schema_version': 1, 'arm': 'random', 'token_budget': n_updates * per_update,
              'tokens_per_update': per_update, 'optimizer_updates': n_updates,
              'training': training, 'updates': updates}
    validate_arm(sliced)
    return sliced


def _indexed(rows):
    indexed = {row['id']: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError('Duplicate generation ids')
    return indexed


def checkpoint_metrics(dev_rows, taught_rows, base_dev_rows):
    """Separate taught-answer fit, fresh-dev accuracy, and base-correct retention."""
    dev = _indexed(dev_rows)
    base = _indexed(base_dev_rows)
    taught = _indexed(taught_rows)
    if set(dev) & set(taught):
        raise ValueError('Taught prompts and fresh development questions overlap')
    if set(dev) != set(base):
        raise ValueError('Checkpoint dev ids differ from the base eval')
    base_correct = {key for key, row in base.items() if score(row)['correct']}
    retained = sum(1 for key in base_correct if score(dev[key])['correct'])
    fresh = summarize_generations(dev_rows)
    taught_summary = summarize_generations(taught_rows) if taught_rows else None
    truncated_match = 0
    for row in list(dev_rows) + list(taught_rows):
        if row.get('truncated') and score(dict(row, truncated=False))['correct']:
            truncated_match += 1
    lengths = [row['generated_tokens'] for row in dev_rows]
    return {
        'fresh_correct': fresh['numeric_correct'],
        'fresh_count': fresh['count'],
        'fresh_truncated': fresh['truncated'],
        'fresh_generated_tokens': fresh['generated_tokens'],
        'mean_fresh_generated_tokens': (sum(lengths) / len(lengths)) if lengths else None,
        'taught_correct': None if taught_summary is None else taught_summary['numeric_correct'],
        'taught_count': 0 if taught_summary is None else taught_summary['count'],
        'taught_truncated': None if taught_summary is None else taught_summary['truncated'],
        'base_correct_count': len(base_correct),
        'retained_base_correct': retained,
        'retention_on_base_correct': (retained / len(base_correct)) if base_correct else None,
        'truncated_but_numeric_match': truncated_match,
        'scoring_policy': fresh['scoring_policy'],
    }


def estimate_gpu_hours(config):
    """Conservative A100-hours for the plan. Eval time scales from the measured dev200 baseline."""
    updates = [int(value) for value in config.get('updates', [64, 256])]
    learning_rates = list(config.get('learning_rates', [2e-5, 1e-4]))
    seeds = list(config.get('train_seeds', [42, 43, 44]))
    every = int(config.get('checkpoint_every_updates', 64))
    sweep_questions = int(config.get('eval_questions_for_sweep', 200))
    full_questions = int(config.get('recommended_dev_questions', 1000))
    repeats = int(config.get('stage0_repeats', 3))

    def eval_seconds(questions):
        return EVAL_200_WORKER_SECONDS * (questions / 200)

    def checkpoints(n_updates):
        points = list(range(every, n_updates + 1, every))
        if not points or points[-1] != n_updates:
            points.append(n_updates)
        return points

    def cell_seconds(n_updates, questions):
        return TRAIN_SECONDS_PER_UPDATE * n_updates + eval_seconds(questions) * len(checkpoints(n_updates))

    cells = len(learning_rates) * len(seeds) * len(updates)
    sweep = sum(cell_seconds(n_updates, sweep_questions) for n_updates in updates) * len(learning_rates) * len(seeds)
    full_sweep = sum(cell_seconds(n_updates, full_questions) for n_updates in updates) * len(learning_rates) * len(seeds)
    stage0_historical = repeats * eval_seconds(int(config.get('historical_dev_questions', 200)))
    stage0_full = repeats * eval_seconds(full_questions)
    chosen_cell_full = max(cell_seconds(n_updates, full_questions) for n_updates in updates)
    return {
        'device': 'one A100-80GB, sequential, from published dev200 worker time',
        'eval_200_worker_seconds': EVAL_200_WORKER_SECONDS,
        'train_seconds_per_update_allowance': TRAIN_SECONDS_PER_UPDATE,
        'allowance_note': ('Three seconds per update is a planning allowance. The observed 64-update '
                           'human200 arm finished in 511.475 worker seconds including its evaluation, '
                           'so the allowance sits inside a job that already happened. It is not a new measurement.'),
        'cells': cells,
        'stage0_hours_on_historical_200': stage0_historical / 3600,
        'stage0_hours_on_suggested_dev': stage0_full / 3600,
        'stage1_hours_sweep_on_eval_slice': sweep / 3600,
        'stage1_hours_if_every_checkpoint_uses_suggested_dev': full_sweep / 3600,
        'recommended_hours_before_stage2': (stage0_full + sweep + chosen_cell_full) / 3600,
        'historical_worker_timeout_seconds': 1800,
        'longest_sweep_cell_seconds_on_eval_slice': max(cell_seconds(n_updates, sweep_questions) for n_updates in updates),
    }


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return value


def _smoke_rows(count, start):
    rows = []
    for offset in range(count):
        gold = str(10 + (offset % 5))
        rows.append({
            'id': f'gsm8k:train:{start + offset:05d}',
            'prompt': f'A basket holds {gold} apples after {offset % 3} are eaten.',
            'response': f'The remaining amount is {gold}\n#### {gold}',
            'gold_answer': gold,
            'source_split': 'train',
        })
    return rows


def _smoke_corpus(config):
    pool_n = int(config.get('n_train_pool', 16))
    dev_n = int(config.get('n_dev', 8))
    held_n = int(config.get('n_heldout_teacher', 6))
    pool = _smoke_rows(pool_n, 0)
    development = _smoke_rows(dev_n, 1000)
    heldout = _smoke_rows(held_n, 2000)
    refuse_sealed_rows(pool + development + heldout)
    texts = [student_valuation.PREFIX + row['prompt'] + ' ' + row['response']
             for row in pool + development + heldout]
    tokenizer = TinyTokenizer(texts)
    return {
        'pool': encode_records(pool, tokenizer, prefix=student_valuation.PREFIX),
        'development': development,
        'heldout': encode_records(heldout, tokenizer, prefix=student_valuation.PREFIX),
        'tokenizer': tokenizer,
    }


def _generations(model, rows, tokenizer, max_new_tokens):
    generated = []
    for row in rows:
        if 'prefix_ids' in row:
            prefix_ids = row['prefix_ids']
        else:
            prefix_ids = tokenizer.encode(student_valuation.PREFIX + row['prompt'])
        token_ids, truncated = model.generate(prefix_ids, max_new_tokens)
        generated.append({
            'id': row['id'], 'gold_answer': str(row['gold_answer']),
            'response': tokenizer.decode(token_ids), 'generated_tokens': len(token_ids),
            'truncated': truncated, 'elapsed_seconds': 0.0,
        })
    return generated


def _taught_rows(arm):
    return list(arm['training'])


def _heldout_nll(model, rows):
    losses = []
    for row in rows:
        start = row['prefix_tokens'] if 'prefix_tokens' in row else len(row['prefix_ids'])
        end = start + row['response_tokens']
        ids = row.get('input_ids') or (row['prefix_ids'] + row['answer_ids'])
        value = model.mean_nll(ids, start, end)
        if value is not None:
            losses.append(value)
    if not losses:
        return None
    return float(sum(losses) / len(losses))


def _run_dose(arm, dev_rows, heldout_rows, tokenizer, *, learning_rate, seed, max_new_tokens, checkpoint_every):
    model = TinyBigramLoRA(tokenizer.vocab_size, seed=seed)
    by_id = {row['id']: row for row in arm['training']}
    base_dev = _generations(model, dev_rows, tokenizer, max_new_tokens)
    train_losses = []
    checkpoints = []

    def record(update_index):
        taught = _generations(model, _taught_rows(arm), tokenizer, max_new_tokens)
        fresh = _generations(model, dev_rows, tokenizer, max_new_tokens)
        metrics = checkpoint_metrics(fresh, taught, base_dev)
        metrics.update({
            'updates': update_index,
            'supervised_tokens': update_index * arm['tokens_per_update'],
            'train_loss': None if not train_losses else train_losses[-1],
            'heldout_teacher_nll': _heldout_nll(model, heldout_rows),
        })
        checkpoints.append(metrics)
        return fresh, taught

    record(0)
    for update in arm['updates']:
        model.zero_grad()
        weighted = 0.0
        for window in update['microbatches']:
            row = by_id[window['id']]
            start = row['prefix_tokens'] + window['answer_start']
            end = row['prefix_tokens'] + window['answer_end']
            weighted += model.accumulate(row['input_ids'], start, end, 1.0 / arm['tokens_per_update'])
        model.optimizer_step(learning_rate)
        train_losses.append(weighted)
        completed = update['index'] + 1
        if completed % checkpoint_every == 0 or completed == len(arm['updates']):
            fresh, taught = record(completed)
    return {
        'learning_rate': learning_rate, 'seed': seed,
        'optimizer_updates': arm['optimizer_updates'],
        'supervised_tokens': arm['token_budget'],
        'tokens_per_update': arm['tokens_per_update'],
        'trained_ids': [row['id'] for row in arm['training']],
        'train_losses': train_losses,
        'checkpoints': checkpoints,
        'base_generations': base_dev,
        'dev_generations': fresh,
        'taught_generations': taught,
        'lora': {'r': 8, 'alpha': 16, 'dropout': 0.0, 'target_modules': ['q_proj', 'v_proj']},
        'optimizer': {'name': 'AdamW', 'betas': [0.9, 0.999], 'eps': 1e-8,
                      'weight_decay': 0.01, 'max_grad_norm': 1.0, 'schedule': 'constant'},
    }


def stage0(config, output):
    backend = refuse_paid(config)
    report = {
        'stage': 0, 'backend': backend, 'scientific_claim': False,
        'official_test_used': False, 'paid_launch': False,
        'power': power_report(config),
    }
    if backend == 'plan':
        report['reproducibility'] = {
            'status': 'not_run',
            'reason': 'GPU repeats of the pinned Granite checkpoint are not launched by this runner',
            'protocol': {
                'decoding': 'greedy, enable_thinking false, max_new_tokens 512, left padding, batch 8',
                'repeats': int(config.get('stage0_repeats', 3)),
                'seeds': list(config.get('train_seeds', [42, 43, 44])),
                'agreement_rule': 'Question-level exact response match across repeats. CUDA SDPA is not pinned deterministic in modal_valuation.execute.',
            },
        }
        return _write(output, report)
    corpus = _smoke_corpus(config)
    max_new = int(config.get('max_new_tokens', 8))
    repeats = []
    for index in range(int(config.get('stage0_repeats', 2))):
        model = TinyBigramLoRA(corpus['tokenizer'].vocab_size, seed=1000 + index)
        repeats.append(_generations(model, corpus['development'], corpus['tokenizer'], max_new))
    reference = {(row['id'], row['response'], row['truncated']) for row in repeats[0]}
    mismatches = []
    for follow in repeats[1:]:
        current = {(row['id'], row['response'], row['truncated']) for row in follow}
        mismatches.append(len(reference.symmetric_difference(current)))
    report['reproducibility'] = {
        'status': 'measured',
        'model': 'tiny-cpu-bigram-lora',
        'repeats': len(repeats),
        'questions': len(repeats[0]),
        'mismatched_questions_against_first_repeat': mismatches,
        'exact_agreement': all(count == 0 for count in mismatches),
        'note': 'B is zero at initialization, so greedy smoke repeats match. The Granite GPU path still needs its own repeats because SDPA is not deterministic.',
    }
    return _write(output, report)


def _matrix(config):
    updates = sorted(int(value) for value in config['updates'])
    rates = [float(value) for value in config['learning_rates']]
    seeds = [int(value) for value in config['train_seeds']]
    per_update = int(config['tokens_per_update'])
    maximum = max(updates)
    return {
        'learning_rates': rates,
        'updates': updates,
        'train_seeds': seeds,
        'data_seed': int(config.get('data_seed', 42)),
        'tokens_per_update': per_update,
        'fixed_subset_supervised_tokens': maximum * per_update,
        'doses_are_prefixes': True,
        'lora': config.get('lora', {'r': 8, 'alpha': 16, 'dropout': 0.0,
                                    'target_modules': ['q_proj', 'v_proj']}),
        'cells': [
            {'learning_rate': rate, 'updates': n_updates, 'seed': seed,
             'supervised_tokens': n_updates * per_update}
            for rate in rates for n_updates in updates for seed in seeds
        ],
    }


def stage1(config, output):
    backend = refuse_paid(config)
    matrix = _matrix(config)
    payload = {
        'stage': 1, 'backend': backend, 'scientific_claim': False,
        'official_test_used': False, 'paid_launch': False,
        'matrix': matrix,
        'reference_rows': PUBLISHED_DEV200 if config.get('include_published_reference') else None,
        'gpu_hours': estimate_gpu_hours(config),
    }
    if backend == 'plan':
        payload['runs'] = []
        payload['status'] = 'plan_only'
        return _write(output, payload)
    corpus = _smoke_corpus(config)
    full = build_random_arm(corpus['pool'], matrix['fixed_subset_supervised_tokens'],
                            matrix['data_seed'], matrix['tokens_per_update'])
    arms = {n_updates: slice_dose(full, n_updates) for n_updates in matrix['updates']}
    full_ids = {row['id'] for row in full['training']}
    for n_updates, arm in arms.items():
        if not set(row['id'] for row in arm['training']).issubset(full_ids):
            raise AssertionError('A shorter dose left the fixed subset')
    heldout_ids = {row['id'] for row in corpus['heldout']}
    if heldout_ids & full_ids or heldout_ids & {row['id'] for row in corpus['development']}:
        raise AssertionError('Held-out teacher rows overlap training or development')
    runs = []
    for rate in matrix['learning_rates']:
        for n_updates in matrix['updates']:
            for seed in matrix['train_seeds']:
                run = _run_dose(
                    arms[n_updates], corpus['development'], corpus['heldout'], corpus['tokenizer'],
                    learning_rate=rate, seed=seed, max_new_tokens=int(config.get('max_new_tokens', 8)),
                    checkpoint_every=int(config.get('checkpoint_every_updates', 2)),
                )
                run['heldout_teacher_ids'] = sorted(heldout_ids)
                runs.append(run)
    payload.update(status='measured', model='tiny-cpu-bigram-lora', runs=runs,
                   fixed_subset_ids=sorted(full_ids))
    return _write(output, payload)


def summarize(path, *, bootstrap_resamples=2000, bootstrap_seed=20261103):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('official_test_used'):
        raise ValueError('Refusing to summarize a payload that used the official test split')
    rows = []
    if payload.get('reference_rows'):
        for reference in payload['reference_rows']['rows']:
            rows.append(dict(reference, kind='published_reference',
                             question_set=payload['reference_rows']['question_set'],
                             comparable_to_these_runs=False))
    comparisons = []
    for run in payload.get('runs', []):
        final = run['checkpoints'][-1]
        comparison = paired(run['base_generations'], run['dev_generations'],
                            seed=bootstrap_seed, resamples=bootstrap_resamples)
        entry = {
            'kind': 'measured', 'learning_rate': run['learning_rate'], 'seed': run['seed'],
            'optimizer_updates': run['optimizer_updates'], 'supervised_tokens': run['supervised_tokens'],
            'final_train_loss': run['train_losses'][-1] if run['train_losses'] else None,
            'heldout_teacher_nll': final['heldout_teacher_nll'],
            'fresh_correct': final['fresh_correct'], 'fresh_count': final['fresh_count'],
            'taught_correct': final['taught_correct'], 'taught_count': final['taught_count'],
            'retention_on_base_correct': final['retention_on_base_correct'],
            'fresh_truncated': final['fresh_truncated'],
            'mean_fresh_generated_tokens': final['mean_fresh_generated_tokens'],
            'versus_base': comparison,
        }
        rows.append(entry)
        comparisons.append(entry)
    by_key = {}
    for entry in comparisons:
        by_key.setdefault((entry['optimizer_updates'], entry['seed']), []).append(entry['learning_rate'])
    matched = []
    for (n_updates, seed), rates in sorted(by_key.items()):
        ordered = sorted(set(rates))
        if len(ordered) < 2:
            continue
        low_run = _find_run(payload, ordered[0], n_updates, seed)
        high_run = _find_run(payload, ordered[-1], n_updates, seed)
        matched.append({
            'updates': n_updates, 'seed': seed,
            'low_learning_rate': ordered[0], 'high_learning_rate': ordered[-1],
            'high_minus_low': paired(low_run['dev_generations'], high_run['dev_generations'],
                                     seed=bootstrap_seed, resamples=bootstrap_resamples),
        })
    return {
        'source': str(path), 'official_test_used': False, 'scientific_claim': False,
        'rows': rows, 'learning_rate_contrasts': matched,
        'reading': ('taught_correct counts greedy numeric matches on prompts that received loss. '
                    'fresh_correct counts the disjoint development questions. '
                    'Published dev200 rows are a different question set and are not paired with these runs.'),
    }


def _find_run(payload, learning_rate, n_updates, seed):
    for run in payload['runs']:
        if run['learning_rate'] == learning_rate and run['optimizer_updates'] == n_updates and run['seed'] == seed:
            return run
    raise KeyError('Missing run for summary contrast')


def plan(config, output):
    refuse_paid(dict(config, backend='plan'))
    value = {
        'stage': 'plan', 'backend': 'plan', 'paid_launch': False, 'official_test_used': False,
        'model': config.get('model', student_valuation.MODEL),
        'revision': config.get('revision', student_valuation.REVISION),
        'matrix': _matrix(config),
        'power': power_report(config),
        'gpu_hours': estimate_gpu_hours(config),
        'reference_rows': PUBLISHED_DEV200,
        'decoding': config.get('decoding', {
            'enable_thinking': False, 'do_sample': False, 'max_new_tokens': 512,
            'evaluation_batch_size': 8,
        }),
        'does_not_launch': ['modal', 'deepseek', 'official_gsm8k_test'],
    }
    return _write(output, value)


def _load_config(path):
    refuse_sealed_path(path)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('stage0', 'stage1', 'plan'):
        command = commands.add_parser(name)
        command.add_argument('--config', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
    summary = commands.add_parser('summarize')
    summary.add_argument('--results', type=Path, required=True)
    summary.add_argument('--output', type=Path, required=True)
    summary.add_argument('--resamples', type=int, default=2000)
    args = parser.parse_args(argv)
    if args.command == 'summarize':
        result = summarize(args.results, bootstrap_resamples=args.resamples)
        _write(args.output, result)
    else:
        config = _load_config(args.config)
        {'stage0': stage0, 'stage1': stage1, 'plan': plan}[args.command](config, args.output)
    print(json.dumps({'command': args.command, 'output': str(args.output), 'paid_launch': False}))


if __name__ == '__main__':
    main()
