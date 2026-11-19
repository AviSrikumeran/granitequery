# Diagnostic experiment plan

The published 65,536-token study did not beat untouched Granite, and skill did not beat random. `docs/AUDIT.md` records the match on supervised tokens and optimizer steps, the mismatch on examples and prompt tokens, and the missing fit curve. This plan measures the training recipe before any new selector is compared.

The official GSM8K test split stays sealed. Stage 0 and Stage 1 are implemented. Stage 2 and Stage 3 are specified and are not executable runners. No command in this document launches Modal or DeepSeek.

## What is held fixed

The student is `ibm-granite/granite-4.2-3b` at revision `4e9c275e232f10211ec88644d106a1f6a1b60e06`. LoRA stays on `q_proj` and `v_proj` only, r=8, alpha=16, dropout 0. The optimizer stays AdamW (betas 0.9/0.999, eps 1e-8, weight decay 0.01, grad clip 1.0) with a constant learning rate. Decoding for the accuracy columns stays greedy, `enable_thinking=False`, 512 new tokens. Truncated generations stay incorrect under `comparison_metrics.py:score`. A separate column counts truncated rows whose text would have matched, and that column does not change the official score.

One random subset is drawn with `diagnostic_experiment.py:build_random_arm`, which calls `student_valuation.py:_exposure`. The longest dose is 256 updates × 1,024 tokens = 262,144 supervised tokens. The 64-update dose is the first 65,536 labels of that same stream (`slice_dose`). Data order uses one `data_seed`. `train_seeds` change initialization and GPU noise. They do not redraw the subset.

## Stage 0 — base eval, seed noise, and power

**Question.** On a frozen development set, how often does greedy untouched Granite disagree with itself, and how large a paired gap can that set detect?

**Run.** Three greedy repeats of the untouched checkpoint on the development questions, same cap and template as the human-200 study. Then one repeat at each of seeds 42, 43, and 44. Log question-level agreement, truncation counts, and length. Compute `minimum_detectable_effect` at the discordant rate those repeats imply. Until the repeats exist, use the published human-200 rates: skill versus random was 9 discordant questions out of 200 (rate 0.045); random versus base was 19 out of 200 (rate 0.095).

**Power, from `diagnostic_experiment.py:power_report`.** At rate 0.045 and n=200, the paired minimum detectable effect is 4.20 percentage points (80 percent power, two-sided 5 percent). The 95 percent interval half-width is 2.94 points. The published skill-minus-random interval was -4.5 to +1.5, which matches that width. A 2-point selector effect needs 883 questions at rate 0.045 and 1,865 questions at rate 0.095.

**Carve.** 200 questions are underpowered for a 2-point effect. Carve 1,000 development questions and a disjoint 200-question pilot from GSM8K **train**, with `carve_train_split`. Pass every id that has already been a development question, a gradient target, or a teaching example in `exclude_ids`. At n=1,000 and rate 0.045 the minimum detectable effect is about 1.9 points. If Stage 0 or Stage 1 shows a discordant rate near 0.095, recompute before treating 1,000 as enough for Stage 3; the 2-point target then wants about 1,865 train questions. Do not take those questions from the official test split.

The 200-question human set can still be the cheap screen inside Stage 1. It should not be the only endpoint for a selector claim.

**Go.** Proceed to the carve when the minimum detectable effect at the working discordant rate is above 2 points. On the published rate, that is now. Proceed to Stage 1 decoding only if three greedy repeats agree on at least 99 percent of questions, or the measured disagreement is written down and is smaller than any effect Stage 1 will claim. The CPU smoke agrees exactly because its adapter starts at zero. The GPU path is unmeasured and is the one that counts.

**No-go.** Do not open the official test split to enlarge the dev set. Do not interpret a GPU disagreement larger than the effect under study as a model difference.

## Stage 1 — dose and learning rate on one random subset

**Question.** At fixed q/v LoRA, which of learning rate {2e-5, 1e-4} and updates {64, 256} actually fit the taught answers, and which of those fits still answer fresh questions?

**Cells.** 2 learning rates × 2 update counts × 3 train seeds = 12 runs, plus the untouched base. DeepSeek is a reference row, not a new call. The published 194/200, 179/200, 174/200, and 171/200 figures are copied into the plan output with `comparable_to_a_new_split: false`. They apply only to the original 200 ids. A new 1,000-question carve has no DeepSeek score until a separately funded collection, which this runner will not start.

**Logs, per checkpoint (every 64 updates, including the start and the end).**

| Column | Meaning |
|---|---|
| `train_losses` | Mean supervised-token NLL of that update, the same weighted mean the worker trains |
| `heldout_teacher_nll` | Mean NLL on teacher answers whose questions are outside the fixed subset |
| `taught_correct` / `taught_count` | Greedy numeric accuracy on prompts that have received loss |
| `fresh_correct` / `fresh_count` | Greedy numeric accuracy on the disjoint development questions |
| `retention_on_base_correct` | Accuracy on development questions the untouched model got right |
| `fresh_truncated`, `mean_fresh_generated_tokens` | Length and cap hits |
| `truncated_but_numeric_match` | Cap hits whose text would have matched. Diagnostic only |

`diagnostic_experiment.py:summarize` adds a paired bootstrap interval of each cell against its own base eval, and of the higher learning rate against the lower one at matched updates and seed.

**Go to Stage 2** when at least one cell (a) lowers train loss across the checkpoint curve and (b) raises `taught_correct` above the untouched model on those same prompts, and (c) the fresh-dev paired interval versus the base is not entirely below zero. Three seeds should agree in sign before that cell is called the recipe.

**A cell beats the base on fresh questions** only when its paired 95 percent interval on the development questions is entirely above zero and the three seeds agree in sign. One favorable seed is a screen, not a result.

**No-go for selection.** If every cell that fits the taught answers has a fresh-dev point estimate below the base, the pattern is the correction study: the adapter memorized and the fresh questions got worse. Stop. Change dose or target. Do not spend a selector comparison on a recipe that forgets.

**No-go for the learning-rate story.** If no cell moves train loss, 1e-4 and 256 updates were still not enough, or the implementation is not training. Selection is uninterpretable until a cell fits.

**Screen versus full dev.** Score every checkpoint of the 12 cells on a frozen 200-question slice (the historical 200, if its ids are excluded from the subset, or the first 200 of the new carve). Score the untouched model on the full 1,000 in Stage 0. After the sweep, rescore only the chosen cell on the full 1,000.

## Stage 2 — target format, at the winning dose

Run only after Stage 1 names a dose and a learning rate. Hold that dose, that learning rate, the LoRA, the optimizer, the token count, and the update count fixed. Three targets, three train seeds:

1. Concise DeepSeek answers, the format already in the 1,930-example human pool.
2. High-thinking DeepSeek answers, complete traces, supervised under the same token budget. The native study prepared this comparison at 16,384 tokens and did not train it. This stage is the first time it would be trained, at the Stage 1 dose rather than at 16 updates.
3. Granite's own greedy answers on train prompts, kept only when `comparison_metrics.py:score` already marks them correct. These are self-generated targets, not a new teacher.

The same `slice_dose` / `_exposure` contract applies, so token totals and update counts match across the three targets. Example counts may differ. Report them. A target that cannot fill the budget without repeating examples is a smaller-budget rerun with its own random control, not a silent repeat.

**Go to Stage 3** when at least one target has a fresh-dev interval versus the untouched base that is not entirely below zero, under the Stage 1 rule. Prefer that target for the selector stage.

**No-go.** Do not compare targets whose supervised-token counts or update counts differ. Do not read the 138/200 or 175/200 open-thinking baselines as this stage; those runs did not train.

Teacher collection for the thinking target is a separate funded API job. This repository's runner refuses `launch_deepseek`.

## Stage 3 — selector versus a matched random arm

Run only after Stage 2 (or Stage 1, if the concise target already transferred) names a recipe that does not lose fresh-dev accuracy.

- Fit the selector on the 200-question **pilot** only. The development questions do not enter the weakness file, the gradient targets, or any early-stop decision.
- Draw the random arm with `build_random_arm` at the same budget and the same `tokens_per_update`. `validate_arm` must pass on both arms with equal `token_budget` and equal `optimizer_updates`.
- Three or more seeds. For the current skill selector, say in the report that `_choose_valued` ignores the seed and the seeds are order replicates plus the boundary example. Random seeds redraw the subset. Those are different variances. Do not label the skill seeds as three independent selections unless the selector is changed so the seed actually redraws it.
- Score development accuracy, retention, taught-fit, held-out teacher NLL, truncation, and length, with paired intervals from `evaluation_stats.py:paired`.

**A selector win** requires the paired interval of selector minus random to lie entirely above zero on the development set, with the token and update counts equal, on a dev set whose minimum detectable effect is at or below the claimed gap. On the published 4.5 percent discordant rate, that means the 1,000-question carve for a 2-point claim, or a still larger carve if the discordant rate is higher.

**No-go.** Unequal supervised tokens or unequal update counts void the comparison. So does a selector that was fit on the same questions it is scored on. The official test split is still closed after Stage 3. A final test eval is a later protocol, with a frozen checkpoint, and is outside this plan.

## GPU-hours

Eval time scales from the measured human-200 baseline, 592.229 worker seconds for 200 questions. Training time is an allowance of 3 seconds per update. The observed 64-update skill arm finished in 511.475 worker seconds **including** its evaluation, so the allowance sits inside a job that already ran. It is not a new measurement. `diagnostic_experiment.py:estimate_gpu_hours` prints the Stage 0 and Stage 1 figures:

| Block | A100-hours |
|---|---:|
| Stage 0, 3 repeats, historical 200 | 0.49 |
| Stage 0, 3 repeats, 1,000 questions | 2.47 |
| Stage 1, 12 cells, checkpoints on a 200-question slice | 6.54 |
| Stage 1 if every checkpoint uses all 1,000 questions | 26.28 |
| Recommended before Stage 2 (Stage 0 on 1,000, sweep on the 200-slice, one chosen 256-update cell rescored on 1,000) | 12.51 |

The longest 256-update cell with four checkpoint evals on 200 questions is about 3,137 seconds. The historical worker timeout is 1,800 seconds. That cell needs a longer timeout or a separate eval job. It does not fit `modal_valuation.py:execute` unchanged.

Stage 2 and Stage 3 use the same two constants. They are conditional on the dose Stage 1 selects.

| Block | 64-update dose | 256-update dose |
|---|---:|---:|
| Stage 2, 3 targets × 3 seeds, 200-question slice | 2.0 h | 7.8 h |
| One self-generated pass over 2,000 train prompts | 1.6 h | 1.6 h |
| Stage 3, 2 arms × 3 seeds, 200-question slice, plus two full 1,000-question evals | 3.0 h | 6.9 h |

Adding the recommended Stage 0–1 budget, a full sequence that reaches Stage 3 at the 256-update dose is about 30 A100-hours. The go/no-go rules are what keep Stage 2 and Stage 3 from running by default. DeepSeek thinking tokens are API spend, not GPU-hours, and are not authorized here.

## Commands

From `granite_query`, with `numpy` available (`python3 -m pip install 'numpy>=1.26'` when it is missing).

CPU smoke, which actually trains the tiny model and is what CI runs:

```bash
python3 cloud/diagnostic_experiment.py stage0 \
  --config cloud/diagnostics/smoke.json \
  --output runs/diagnostic_smoke/stage0.json

python3 cloud/diagnostic_experiment.py stage1 \
  --config cloud/diagnostics/smoke.json \
  --output runs/diagnostic_smoke/stage1.json

python3 cloud/diagnostic_experiment.py summarize \
  --results runs/diagnostic_smoke/stage1.json \
  --output runs/diagnostic_smoke/summary.json

python3 -m unittest cloud.test_diagnostic_experiment
```

Unpaid Granite plan. These commands write the matrix, the power calculation, and the GPU-hour estimate. `backend` is `plan`. They do not load Granite, call Modal, or call DeepSeek:

```bash
python3 cloud/diagnostic_experiment.py plan \
  --config cloud/diagnostics/granite_stage01.json \
  --output runs/diagnostic_granite/plan.json

python3 cloud/diagnostic_experiment.py stage0 \
  --config cloud/diagnostics/granite_stage01.json \
  --output runs/diagnostic_granite/stage0.json

python3 cloud/diagnostic_experiment.py stage1 \
  --config cloud/diagnostics/granite_stage01.json \
  --output runs/diagnostic_granite/stage1.json
```

`runs/` is gitignored. A config with `launch_modal`, `launch_deepseek`, `allow_official_test`, or any other backend raises before it writes.

## What the code does

| Piece | Role |
|---|---|
| `cloud/diagnostic_experiment.py:stage0` | Power report, and greedy-repeat agreement on the tiny model |
| `cloud/diagnostic_experiment.py:stage1` | Dose × learning-rate sweep on one exposure prefix |
| `cloud/diagnostic_experiment.py:summarize` | Paired bootstrap intervals via `evaluation_stats.py:paired` |
| `cloud/diagnostic_experiment.py:plan` | Granite matrix and GPU-hours, no training |
| `cloud/diagnostic_experiment.py:carve_train_split` | Disjoint dev, pilot, and remainder from train rows |
| `cloud/diagnostic_tiny.py:TinyBigramLoRA` | CPU stand-in: frozen base, q/v LoRA, one AdamW step per update |
| `cloud/diagnostics/smoke.json` | Tiny grid, 2 and 4 updates |
| `cloud/diagnostics/granite_stage01.json` | The 12-cell Granite matrix, `backend: plan` |
| `.github/workflows/diagnostic-smoke.yml` | CPU unittest on push and pull request |

The tiny model is a bigram with a zero-initialized LoRA B. Its accuracy is not a Granite result. The smoke checks the contract: token totals, update counts, prefix doses, disjoint taught and fresh ids, finite losses that fall, held-out teacher NLL, retention, truncation stats, sealed-test refusal, and paid-launch refusal.
