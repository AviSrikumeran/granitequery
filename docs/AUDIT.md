# Audit of the Granite training, selection, and evaluation pipeline

This audit checks the repository against two questions. Does fine-tuning the pinned `ibm-granite/granite-4.2-3b` checkpoint (`4e9c275e232f10211ec88644d106a1f6a1b60e06`) beat the untouched model on GSM8K development questions? Does a selector beat a random arm at the same supervised-token budget? The official GSM8K test split is sealed. Nothing below opens it.

Numbers below are the published logs in this repository. Private `runs/` artifacts are gitignored, so per-question generations and training-loss vectors for the GPU jobs are not in the tree. Where a public file records a count, that file is the source.

## Short reading

On the 200-question human-reference development set, non-thinking greedy decoding with a 512-token cap, the published scores are DeepSeek 194, untouched Granite 179, random-trained 174, skill-trained 171. Both adapters record 65,536 supervised answer tokens and 64 optimizer updates. The exposure code and `valuation_contract.validate_arm` require those two quantities to match, and the published tables match them. Example counts and forwarded input tokens do not match: random used 561 examples and 136,319 forwarded tokens; skill used 766 examples and 150,600 forwarded tokens.

The same tables do not support either hypothesis as a positive result. Fine-tuning did not beat the untouched model. Skill did not beat random: the paired question-bootstrap interval for skill minus random is -4.5 to +1.5 percentage points. A separate 64-question multi-source grid, with the same token and update matching, also never has skill or gradient above random.

The results do support a narrower diagnosis. The 200-question set can reliably detect only about a 4.2 percentage-point paired gap at the discordant rate this study actually produced, so a 2-point selector effect would be invisible. The completed recipe is 64 constant-LR AdamW steps at 2e-5 over q/v LoRA (r=8, alpha=16), one pass over the tokens. A later local correction study fit 8/8 taught mistakes only after capacity, update count, and learning rate changed together, while development accuracy fell from 30/32 to 28/32. The concise-versus-thinking student comparison was prepared and never trained.

## End-to-end pipeline

1. Teacher answers are collected for GSM8K **train** prompts. Gold answers are checked locally and are not sent as the target. `ranking/verification.py:teacher_eligible` and `student_valuation.py:_verified` accept a response only when it finished on `stop`, contains `####`, and the final number matches the gold answer. Truncated teacher traces are rejected.
2. A weakness file aggregates the student's errors by lexical skill tags. `student_valuation.py:freeze_weakness` builds it from training-probe or development predictions. The human-200 study points this file at the earlier 64-question development run (`human_study_control.py` constant `WEAKNESS`) and refuses overlap with the new 200 ids.
3. `student_valuation.py:prepare` tokenizes each eligible trace with the pinned tokenizer: chat template, `enable_thinking=False`, prefix and `response + EOS` encoded separately, no truncation. Rows that exceed the context limit, or whose answer is outside 16–768 tokens, are dropped.
4. Selection writes two arms from that pool. `student_valuation.py:_choose_valued` greedily scores weakness, lexical novelty, and skill coverage, with a mild length penalty. `student_valuation.py:select` draws the random arm by shuffling ids with `random.Random(seed)` and taking examples until the token budget is covered. `data_efficiency_control.py:prepare` adds a third arm: gradient scores from `gradient_valuation.py:score_projected`, then `take_until`.
5. `student_valuation.py:_exposure` turns a selected set into labels and optimizer steps. Prompt tokens are labeled `-100`. Answer tokens are labeled with their ids until the budget is exhausted. The crossing example can keep a masked tail. Steps are groups of exactly `tokens_per_update` supervised tokens (1,024 in the real studies). `valuation_contract.py:validate_arm` rejects a plan whose labels, windows, or step count disagree with that budget.
6. `modal_valuation.py:execute` trains one fresh base per call. LoRA is r=8, alpha=16, dropout 0, targets `q_proj` and `v_proj`. AdamW uses the library default (betas 0.9/0.999, eps 1e-8, weight decay 0.01) at a constant learning rate. Each window's mean token loss is multiplied by `supervised_tokens / 1024` and backpropagated; `optimizer.step` runs once per update. The gradient-feature warmup adapter is not loaded (`fresh_base` on the human and data-efficiency plans).
7. The same worker then generates on the frozen development questions: greedy, `enable_thinking=False`, `max_new_tokens` 512, left padding, batch size 8. `valuation_contract.py:trim_generation` marks a generation truncated when no EOS appears before the cap. `comparison_metrics.py:score` counts a truncated generation incorrect even if the answer text is present. `evaluation_stats.py:paired` compares two arms on the same ids.
8. Official-test reads are a separate, unused path (`final_test_loader.py:select` and the final-scale modules). `student_valuation.py:read_jsonl` raises if the filename is `test.jsonl`, `test.json`, or `official_test.jsonl`. The project brief says that read has not been executed.

Older workers are different recipes and should not be pooled with the token-matched studies:

| Worker | Function | How it differs |
|---|---|---|
| `modal_train.py` | `train` | 15-example smoke, learning rate 1e-4, one step per example, prompt text without the word "concisely" |
| `modal_improve.py` | `execute` | `train_full120` steps once per example, so token totals follow example length |
| `native_reasoning_contract.py` | `select_whole_targets`, `exposure_windows` | Whole-target subset sum, 16,384 supervised tokens, 16 updates, open-thinking eval. Adapters were prepared and not trained |

## Where each piece lives

| Piece | Location |
|---|---|
| Skill selector | `student_valuation.py:_choose_valued` |
| Random draw | `student_valuation.py:select` and `data_efficiency_control.py:take_until` |
| Gradient scores | `gradient_valuation.py:score_projected`, `score_file` |
| Exact label mask and update grouping | `student_valuation.py:_exposure` |
| Mask checker used by the worker | `valuation_contract.py:window_labels`, `validate_arm` |
| GPU train and greedy eval | `modal_valuation.py:execute` |
| Human-200 plans (seed 42, 65,536 tokens, LR 2e-5) | `human_study_control.py:prepare`, `plan_for` |
| 32,768 / 65,536 / 131,072 grid | `data_efficiency_control.py:prepare`, `prepare_grid`, `expose` |
| Numeric score and truncation rule | `comparison_metrics.py:score`, `summarize` |
| Paired bootstrap and McNemar | `evaluation_stats.py:paired` |
| Human-200 report | `human_study_report.py:build` |
| Matched DeepSeek-200 collection | `development_teacher200.py:collect`, `summary` (same `score`) |
| Teacher acceptance | `ranking/verification.py:teacher_eligible` |
| Native concise vs thinking targets | `native_reasoning_contract.py:encode_target`, `select_whole_targets` |
| Native control (baseline ran; adapters did not) | `native_reasoning_control.py` and the v2/v3/v4 transport modules |

## Do ranked and random match on supervised tokens and updates?

**Yes for non-masked answer tokens and for optimizer steps. No for examples, prompt compute, and which boundary example is cut.**

`_exposure` walks the selected rows in hash order of `SHA256(f"{seed}:{id}")` and assigns `min(remaining, response_tokens)` supervised labels. It raises if the budget is not a multiple of `tokens_per_update` or if the pool runs out. Each resulting update has `supervised_tokens == tokens_per_update`, and the window `loss_weight` values in an update sum to 1. `validate_arm` checks that the label mask, the windows, and `optimizer_updates * tokens_per_update` all equal the budget. `modal_valuation.py:execute` steps once per update and raises if the running token count disagrees with the budget at the end.

Published completed arms agree with that contract:

| Study | Arm | Supervised tokens | Updates | Examples | Forwarded input tokens | Correct |
|---|---|---:|---:|---:|---:|---:|
| Human dev200, seed 42 | untouched | 0 | 0 | 0 | 0 | 179/200 |
| Human dev200, seed 42 | random | 65,536 | 64 | 561 | 136,319 | 174/200 |
| Human dev200, seed 42 | skill | 65,536 | 64 | 766 | 150,600 | 171/200 |
| Multi-source, 32,768, seeds 42 and 43 | random / skill / gradient | 32,768 | 32 | 340–467 | 67,502–81,893 | 54–57/64 |
| Multi-source, 65,536, seeds 42 and 43 | random / skill / gradient | 65,536 | 64 | 556–825 | 136,410–155,567 | 56–58/64 |
| Untouched multi-source control | baseline | 0 | 0 | 0 | 0 | 58/64 |

Sources: `progress/HUMAN_STUDY_RESULTS.md`, `progress/DATA_EFFICIENCY_RESULTS.md`, `progress/DEVELOPMENT_TEACHER200.md`. The 131,072-token arms were prepared and not run.

The worker's loss scale matches a mean over the update's supervised tokens. Hugging Face returns the mean over non-ignored labels; multiplying by `count / 1024` and summing across windows is the mean over the full 1,024. `test_student_valuation.py` locks the weight sum at 1. This is the right match for "same supervised-token budget." It is not a match for sequence-level loss, and a short window contributes the same per-token weight as a long one.

Three leftovers sit inside that match:

- The selector may overshoot the budget by one example. `_exposure` then keeps only a hash-ordered prefix. `unexposed_preselected_ids` records the dropped ids. The dropped example is whichever id sorts last, not the lowest-scoring one. At 65,536 tokens the skill arm has 824 examples at seed 42 and 825 at seed 43, with the same 65,536 labels. Skill selection itself does not take a seed (`_choose_valued` is called once, outside the seed loop in `data_efficiency_control.py:prepare`). The one-example gap is the boundary moving.
- The crossing example's tail is labeled `-100` while the full token ids stay in the forward. The tail is where `####` and EOS sit. One example per arm is affected. Causal loss on the earlier tokens does not depend on that tail.
- Skill's length penalty prefers shorter answers, so the same 65,536 labels cover more prompts and more forwarded tokens (150,600 versus 136,319 on the human study). Gradient goes the other way (fewer, longer examples). Equal answer-token counts were the declared estimand. They leave prompt compute unmatched.

The early 16-question bridge is a different design. The journal records ranked at 3,260 supervised tokens and random at 2,534. Later token-matched studies replaced that comparison. Those two numbers should not be read as the 65,536-token result.

The local correction study did **not** match tokens. `docs/overnight-feedback-results.json` records 5,888 versus 5,136 loss tokens at 8 updates. `docs/overnight-capacity-results.json` records 47,104 versus 41,088 at 64 updates. `progress/OVERNIGHT_CAPACITY_RESULTS.md` states the cause: complete targets with equal example counts and equal updates, and unequal lengths.

## Evaluation protocol

The human-200, data-efficiency, and DeepSeek-200 measurements share this decoding setup, implemented in `modal_valuation.py:execute` and scored by `comparison_metrics.py:score`:

- Chat template with the same concise prefix used for training (`student_valuation.PREFIX`). `enable_thinking=False`.
- Greedy decoding (`do_sample=False`). No temperature sampling.
- Cap of 512 new tokens. Batch size 8, left padding, `add_special_tokens=False` on the already rendered prompt.
- EOS handling: stop ids are `generation_config.eos_token_id` plus `tokenizer.eos_token_id`. The native contract pins `<|im_end|>` id 100257 and checks it. The valuation worker does not assert that id. A loaded config with an extra stop id would end generations early relative to the native path.
- Truncation: `trim_generation` returns the generated ids through the first EOS, or the whole suffix if no EOS appears. `truncated` is true only when the suffix reaches `max_new_tokens` with no EOS. The returned id list includes the EOS when one was produced; `generated_tokens` therefore counts that EOS. Decode uses `skip_special_tokens=True`.
- Score: last number after the last `####`, otherwise the last number on the last non-empty line. Decimal comparison. A truncated row is incorrect even when that number matches. Strict format (marker plus a clean number) is stored separately. On the 33-case human audit, strict and heuristic totals were identical (`progress/HUMAN_DEVELOPMENT_ERROR_AUDIT.json`).
- Development questions are train-split holds. Teaching ids and normalized prompts are checked disjoint from the 200 (`human_study_control.py:prepare`). The 200 were chosen after earlier negative results. The report calls that adaptive. It is not the official test.

The DeepSeek-200 row uses the same score function (`development_teacher200.py:summary`). The published 194/200 is 189 strict matches plus 5 last-line fallbacks, 0 truncations, 0 API failures (`progress/DEVELOPMENT_TEACHER200.md`).

Two other untouched-Granite numbers on the **same 200 questions** use different decoding and are not the 179 baseline:

| Condition | Correct | Truncated | Training |
|---|---:|---:|---|
| Closed, non-thinking, 512 | 179/200 | 8 | none |
| Open thinking, 1024 | 138/200 | 59 | none |
| Open thinking, batch 32, 2048 | 175/200 | 17 | none |

Sources: `progress/DEVELOPMENT_TEACHER200.md`, `docs/journal.md` entries for the V3 and batch32/2048 terminals, `progress/NATIVE_REASONING_BATCH32_V4.md`. The journal says the 59 caps were checked against native EOS ids and were not an EOS-id bug. Longer generation changed the score and still did not beat the closed 512 baseline.

## Seeds and determinism

`modal_valuation.py:execute` calls `transformers.set_seed(plan['seed'])` before load and again before LoRA initialization. Dropout is 0. The code does not set `torch.use_deterministic_algorithms`, cuDNN determinism, or a deterministic SDPA kernel. Greedy decoding removes sampling noise. It does not remove GPU matmul noise. Two repeats of the untouched model are not guaranteed identical by the code. That measurement was not published. Stage 0 exists to take it.

Seed means different things for the two arms:

- Random: the seed shuffles which examples are drawn, then hash-orders the exposure.
- Skill and gradient: the selected set is computed once. The seed only hash-orders that set, which can change the boundary example and the split of sequences across windows. Forwarded-token totals for skill at 32,768 tokens are 81,230 (seed 42) and 81,893 (seed 43) with the same 467 examples. The two-seed standard deviations in `progress/DATA_EFFICIENCY_RESULTS.md` mix selection noise for random with order noise for skill and gradient. They are not three comparable selection replicates.
- The human-200 pair is one seed, 42.

`evaluation_stats.py:paired` resamples questions with `random.Random(20261103)` for 20,000 draws. The percentile index is `int(0.025 * (resamples - 1))`. The McNemar p-value doubles the one-sided binomial tail. The function's own text says the interval covers question sampling, one training seed, and no multiplicity correction. The published skill-minus-random interval [-4.5, +1.5] is that procedure on 3 corrections and 6 regressions out of 200. A half-width near 3 points matches `sqrt(9/200)` times 1.96.

## Bugs and confounds

These are the issues that change what the published comparison can mean.

1. **Selector seeds are order seeds.** `_choose_valued` and the gradient sort ignore the seed. Calling the 42-versus-43 skill gap a selection replicate overstates how much the selector was re-drawn. Evidence: example count 824 versus 825 at a fixed 65,536 labels (`progress/DATA_EFFICIENCY_RESULTS.md`).
2. **Hash-order exposure can drop a selected example and always masks one answer tail.** The trained subset is the hash prefix of the selected set, not the selector's own order. Magnitude on the human study is about one example in 561 or 766. The masked tail is the end of the answer, including the final number, on that one example.
3. **Prompt compute is unmatched by construction.** Skill's length penalty buys more, shorter traces. The human pair forwards 150,600 versus 136,319 input tokens for the same 65,536 labels and 64 steps.
4. **Truncation is scored as failure.** The rule is applied to every arm, so the 179/174/171 comparison is internally consistent. It still mixes "finished the trace" with "solved the problem." The 33-case audit found 4 of 7 random corrections were capped baseline traces replaced by complete answers, and 0 of 12 random regressions were capped (`progress/HUMAN_DEVELOPMENT_ERROR_AUDIT.json`). The net drop is among completed answers. Adapters also truncated less often (2 versus 8), so the scoring rule is not what made the adapters look worse.
5. **No published fit curve for the 65,536-token arms.** The worker appends a weighted loss per update and writes `weighted_losses` into the private result. The human-200 and data-efficiency reports publish accuracy, truncation, tokens, and updates. They do not publish train loss, held-out teacher loss, or development accuracy at intermediate checkpoints. A flat loss and a memorized loss are both compatible with the public tables.
6. **The correction study changed several knobs at once and did not match tokens.** See the hypothesis section. The public JSON does not record the two learning rates or the two target-module lists, so the confound cannot be decomposed from the tree.
7. **GPU determinism is unmeasured.** `set_seed` plus dropout 0 is the whole reproducibility story in `modal_valuation.py:execute`. SDPA in bf16 can disagree with itself. The data-efficiency scores already move by 2 questions out of 64 across seeds on a fixed skill set, which is the same size as the selector gaps.
8. **Eval EOS is not pinned on the valuation path.** The native contract checks eos id 100257. The valuation worker uses the loaded generation config. This is a protocol gap between the two eval stacks, not a demonstrated mis-stop on the 179/200 run. The journal's EOS check applies to the open-thinking 59-cap run.
9. **The 1400-second training guard can save a partial adapter.** Completed human and data-efficiency jobs reported the full token count, so this did not rewrite those tables. A future 256-update run does not fit the historical 1,800-second worker timeout if it also repeats the 200-question eval (see `docs/EXPERIMENT_PLAN.md`).

The early unequal-token pilot, the full-120 one-step-per-example run, and the open-thinking baselines are confounds only if they are plotted on the same axis as the 65,536-token pair. The reports keep them separate. This audit does too.

## Findings versus hypotheses

### Fine-tuning beats untouched Granite

**Finding: not supported on the published development sets.**

Human dev200: random 174 and skill 171 versus untouched 179, with fewer truncations after training (2 versus 8). Random minus base is -2.5 points, paired interval [-7.0, +1.5], 7 corrections and 12 regressions. Skill minus base is -4.0 points, interval [-7.5, -0.5], 3 corrections and 11 regressions. The skill interval excludes 0. Both are one seed, question bootstrap only (`progress/HUMAN_STUDY_RESULTS.md`).

Multi-source 64-question grid: untouched batch-8 control 58/64. Every completed adapter is 54–58/64. None is above the control (`progress/DATA_EFFICIENCY_RESULTS.md`).

### A selector beats random at the same supervised-token budget

**Finding: not supported. The match on tokens and updates is real, and the gap is smaller than the noise.**

Human dev200, skill minus random: -1.5 points, 3 corrections and 6 regressions, interval [-4.5, +1.5], McNemar p 0.51. One seed.

Multi-source grid: at 32,768 tokens both seeds have skill and gradient one question below random. At 65,536 tokens skill ties random on both seeds (56/64 and 58/64) and gradient ties or falls short. Intervals on 64 questions are several points wide. The 131,072 plans were not run; the preparer labels that budget a near-all-data reference with weak selection contrast (`data_efficiency_control.py:prepare`).

### Too little training

**Hypothesis, consistent with the dose, not isolated.**

The completed student recipe shows each supervised token once (`data_repetition: false`), then takes 32 or 64 AdamW steps at constant 2e-5. There is no public loss curve, so "the model did not fit" and "the model fit and then failed to transfer" are both open for the 65,536-token arms.

The local correction study is the only place a fit measurement was published (`docs/overnight-results.md`). At 8 updates the adapter still scored 0/8 on the taught mistakes, and development stayed 30/32. That supports "this small dose does not fit the taught answers." It does not say whether 64 updates at 2e-5 on 65,536 diverse tokens would fit them.

### Learning rate too low for LoRA

**Hypothesis, confounded.**

The correction write-up says capacity, update count, and learning rate changed together. The larger setting fit the 8 taught mistakes (0/8 to 8/8) and drove training NLL from 1.08005 to 0.001008. The tree does not publish the two learning-rate values or the two module lists, so 2e-5 cannot be singled out. LoRA alpha/r = 2 and bf16 are additional reasons a 2e-5 step can be small. They are reasons to run the sweep, not a result of it.

### Too little headroom

**Finding: headroom on this protocol is real and modest. It does not explain a zero selector effect by itself.**

DeepSeek minus untouched Granite on the same 200 questions, same prefix, non-thinking, 512-token cap: +7.5 points, interval [4.0, 11.5]. DeepSeek solves 16 questions the base misses; the base solves 1 that DeepSeek misses. Seven of those 16 base misses were capped and nine were not (`progress/DEVELOPMENT_TEACHER200.md`). The base is already at 89.5 percent, so the movable set is small. That is a power problem, which is measured below, rather than an absence of teacher-only questions.

At the skill-versus-random discordant rate of 9/200 = 0.045, the paired minimum detectable effect on 200 questions is 4.2 percentage points (80 percent power, two-sided 5 percent). The observed interval half-width is about 2.9 points. A 2-point selector gain is inside that noise. The formula and the 1,000-question recommendation are in `diagnostic_experiment.py:minimum_detectable_effect` and `power_report`.

### Overfitting or memorization

**Finding on the tiny correction set. Not measured on the 65,536-token arms.**

The larger correction adapter fit 8/8 taught mistakes and kept 8/8 replay answers, while unused-teacher-target NLL moved from 1.02025 to 1.03372 and development fell from 30/32 to 28/32 (`docs/overnight-results.md`, `progress/OVERNIGHT_CAPACITY_RESULTS.md`). The same document says the targets are 8 distinct corrections plus 8 replay examples, repeated. That is memorization of a tiny repeated set.

The 65,536-token arms use hundreds of distinct traces once, and the development questions are excluded from teaching. Nothing published separates their train NLL from a held-out teacher NLL. Stage 1 logs those two losses on purpose.

### Teacher-style imitation

**Hypothesis, compatible with the length change, not isolated.**

Skill's human-200 eval generated 23,934 tokens and 2 truncations, against the untouched model's 8 truncations and a longer generation (`PROJECT_BRIEF.md`, `progress/DEVELOPMENT_TEACHER200.md`). Training targets are concise DeepSeek answers under the same prefix. Shorter student outputs plus a net accuracy drop fits "the adapter learned a shorter style and lost some base-correct solutions." It also fits ordinary forgetting. The public logs do not tag which.

### Forgetting

**Finding on the point estimates; the interval excludes 0 for skill versus base only.**

Random versus base: 7 questions gained, 12 lost. Skill versus base: 3 gained, 11 lost, interval [-7.5, -0.5]. The 12 random regressions were not cap failures. Retention on the questions the base already had right was not a logged training metric. The multi-source adapters sit at or below 58/64 as well. This is the pattern Stage 1's retention column is there to track at more than one checkpoint.

### Concise versus high-thinking targets

**Finding: the student comparison never ran.**

Teacher collection for 256 train questions, concise cap 1,024 versus thinking cap 4,096, finished with 246 eligible pairs (`PROJECT_BRIEF.md`). The student plan was two fresh adapters, 16,384 complete supervised tokens, 16 updates, open-thinking eval with a 1,024 cap, then a later batch-32 / 2,048 revision. The open-1024 baseline scored 138/200 and missed the runtime gate (forecasts about 1,754 and 1,747 seconds against a 1,250-second cutoff). The batch-32 / 2,048 baseline scored 175/200 with 17 truncations and also missed the gate. Both concise and thinking adapters stayed in `prepared` (`progress/NATIVE_REASONING_STUDY.md`, `progress/NATIVE_REASONING_BATCH32_V4.md`). Those baselines are not a target-format result.

## Outside starter

`uploads/granitequery-modal.zip` is a separate draft. It switches several factors at once relative to the recipe above: LoRA on all linear layers at r=16 (about 31M trainable parameters), learning rate 1e-4, 3 epochs, DeepSeek Flash with thinking off, and selection from a training-loss trajectory (confidence, variability, learnability). Its README says the Modal GPU path was not run. `prepare_gsm8k.py` in that draft downloads GSM8K and writes the official test split to `test_SEALED.jsonl`.

This repository's diagnostic does not adopt that stack. It keeps four ideas from it: a CPU smoke with a tiny model, a hard refusal to read the test split, a random control matched on answer tokens, and a loss curve logged during training. The Granite matrix stays q/v LoRA r=8, and 1e-4 appears only as one cell next to 2e-5.
