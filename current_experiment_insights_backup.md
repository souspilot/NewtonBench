# NewtonBench under Experimental Budgets

## Results and diagnostics from the completed pilot

**Status:** 15 September 2026  
**Models:** `muse-glimmer-30b`, `qwen38-27b`  
**Scoring:** independent `gemma4-31b` rejudge followed by a deterministic-first scoring cascade  
**Coverage:** 3,455 of 3,456 planned trajectories; Muse at \$1,000 has 287 rather than 288 trials  
**Pilot validity:** scoring is reliable for the stored submissions, but a conservative audit confirms that the experimental trajectories were materially affected by an action-channel handling bug; paper-facing results require a runner repair and full rerun<br>
**Primary run record:** [`full_diagnostics.log`](/Users/harshitbisht/full_diagnostics.log)

This report describes a pilot. It reports observed differences and measurement limitations; it does not claim that the present cost schedule is optimal, realistic, or sufficient to isolate causal mechanisms.

> **Measurement warning.** Under a conservative definition—no syntactically valid action in main content and exactly one valid action in reasoning—20.1–52.1% of trials per model-condition contain at least one recoverable reasoning-only action. Of these candidate turns, 55.0–69.4% for Muse and 98.2–100% for Qwen38 are immediately followed by rejection feedback. Lost experiment and Python actions change the evidence received by the agent. Rejudging can correct labels for stored laws, but it cannot reconstruct these trajectories.

The five traces below quote specific saved trials. Ellipses, compacted JSON, and omitted setup lines shorten the excerpts; the reasoning, actions, and harness responses are otherwise preserved.

## 1. Study at a glance

NewtonBench asks an agent to discover a hidden, counterfactual physical law by choosing experiments and then submitting an executable Python function. The present study adds a hard experimental grant. Each request has a setup cost, each observation has a cost, and better precision or unusual parameter ranges cost more.

| Component | Pilot setting |
|---|---|
| Scientific tasks | 12 NewtonBench modules, one representative difficulty/system cell per module |
| Difficulty mix | 3 easy, 6 medium, 3 hard modules |
| Repetition | 3 hidden law versions × 4 trials |
| Agent interfaces | vanilla and code-assisted |
| Resource conditions | unbudgeted baseline; hard grants of \$1,000, \$800, \$600, \$400, and \$200 |
| Trial count | 288 per model-condition, except Muse \$1,000 (287) |
| Primary outcome | deterministic-first law-recovery success (SA) |
| Secondary outcome | independent Gemma judge score |

The budgeted condition prices five aspects of an experiment:

| Cost component | Rule |
|---|---|
| Setup | \$30 for each input parameter varied within a request |
| Data points | \$15 base price per observation |
| Batch size | later points in a large batch are progressively more expensive |
| Precision | levels 1–5 cost 1×–6.5× and return 1–6 significant figures |
| Parameter range | values outside \(10^{-3}\)–\(10^3\) incur a per-decade surcharge |

Unaffordable requests are rejected without charge. Two consecutive rejections end the experimental campaign and require a final-law submission. The unbudgeted baseline has no monetary pricing or precision-dependent rounding; it retains the benchmark's safety limits of 20 points per request and 200 points per trial.

### Main pilot observations

| Observation | Evidence | Scope of interpretation |
|---|---:|---|
| Recorded accuracy falls as the assigned grant tightens | Muse: 67.7% unbudgeted → 22.2% at \$200; Qwen38: 88.2% → 36.8% | Provisional: action-channel failure rates differ across conditions |
| The largest measured step is \$400 → \$200 | Muse: −20.5 pp; Qwen38: −33.0 pp | A property of this price schedule and task subset |
| Difficulty strongly moderates the budget effect | At \$200, easy/medium/hard SA is 56.9/16.0/0.0% for Muse and 66.7/38.9/2.8% for Qwen38 | Difficulty bins pool different modules |
| Agent interface matters | Muse vanilla exceeds code-assisted at every condition; Qwen38 differences are small and inconsistent | An interaction, not evidence that tools are generally harmful |
| Reasoning-channel action loss is material | Conservatively recoverable actions occur in 31.2–39.6% of Muse trials and 20.1–52.1% of Qwen38 trials | The condition-dependent rates invalidate paper-facing comparisons until rerun |
| Scoring still needs an audit | 125 of 3,455 trials have a judge/checker conflict | The score is usable for the pilot but not yet frozen for publication |

## 2. What a trial measures

A trial contains two linked decisions:

1. **Experimental design:** which inputs to vary, how many observations to buy, and at what precision.
2. **Law recovery:** how to turn the observations into a symbolic mechanism and submit it as `def discovered_law(...)`.

The representative subset contains:

| Difficulty | Modules | Trials per model-condition |
|---|---|---:|
| Easy | Fourier law, Snell law, radioactive decay | 72 |
| Medium | gravity, Coulomb force, underdamped harmonic motion, Malus law, sound speed, BE distribution | 144 |
| Hard | magnetic force, Hooke law, heat transfer | 72 |

For each model-condition, the intended count is

\[
12\ \text{modules} \times 2\ \text{agents} \times 3\ \text{law versions} \times 4\ \text{trials}=288.
\]

Qwen38's unbudgeted source directory contained 480 trajectories because it also held out-of-subset cells. The manifest filter retained the intended 288. Muse's \$1,000 condition is missing one hard trial.

### Trace 1 — turning measurements into a scaling hypothesis

*Muse, code-assisted, gravity, medium/v0, unbudgeted, trial 2.*

The model compared accelerations after varying one input at a time:

> “At distance 4, the acceleration ratio is 8.0; \(4^{1.5}=8\). With `mass1=4`, the ratio is 16.0; \(4^2=16\). With `mass2=4`, the ratio is 4.0.”

It then proposed a targeted invariance test:

```text
<run_experiment>
[
  {"mass1":1,"mass2":1,"distance":1,"initial_velocity":1,
   "duration":0.5,"time_step":0.1},
  {"mass1":1,"mass2":1,"distance":1,"initial_velocity":-1,
   "duration":0.5,"time_step":0.1}
]
</run_experiment>
```

The scientific pattern is recognizable: controlled variation produced an exponent hypothesis, followed by an experiment intended to test whether velocity was irrelevant. However, the action was emitted in separated reasoning rather than main content, so the harness returned:

```text
Action Reminder: Please use exactly 1 action per turn with correct format.
```

This single trace illustrates why the project must measure both scientific reasoning and whether the experimental interface actually executes the intended action.

## 3. How laws are scored

### 3.1 Deterministic checker

The deterministic checker asks a narrow question:

> Does the submitted Python function have exactly the same dependence on the physical inputs as the hidden law, allowing only an overall multiplicative constant to differ?

It does not fit the law to sampled points and does not use RMSLE to decide the primary label.

| Stage | Operation | Why it is needed |
|---:|---|---|
| 1 | Parse `discovered_law` with Python's abstract syntax tree (AST) | Rejects or defers code that cannot be interpreted as a formula |
| 2 | Create positive-real SymPy symbols for the function's positional parameters | Encodes the benchmark domain and preserves positional calling semantics |
| 3 | Resolve local assignments in program order | Distinguishes a genuine constant from an intermediate expression that depends on an input |
| 4 | Convert arithmetic and approved math calls into an exact symbolic expression | Numeric literals are retained as exact rationals; supported calls include powers, roots, exponentials, logarithms, and trigonometric functions |
| 5 | Parse the hidden law using the same physical-variable symbols | Names such as `HIDDEN_CONSTANT` are treated as unknown nuisance constants |
| 6 | Compare physical-variable dependencies | A missing or extra input dependence is a structural mismatch |
| 7 | Simplify `submitted / hidden` | If no physical input remains in the ratio, the laws differ only by an overall constant and pass |

Examples of the equivalence rule:

| Submitted form | Hidden form | Deterministic result | Reason |
|---|---|---|---|
| `6.7e-20 * m1 * m2 / r**2` | `HIDDEN_CONSTANT * m1 * m2 / r**2` | pass | only the multiplicative constant differs |
| `C * m1**2 * m2**2 / r**2` | `HIDDEN_CONSTANT * m1 * m2 / r**2` | fail | physical-variable exponents differ |
| `C * sqrt(T/M)` | `sqrt(HIDDEN_CONSTANT*T/M)` | pass | the hidden constant changes only total scale |
| `1/(exp(C*x/T)+1)` | same form with unknown `C` | not checkable | the unknown constant is inside a nonlinear function |

The checker supports `+`, `-`, `*`, `/`, `**`, unary signs, and a whitelist of common `math`/NumPy-style functions. It also ignores a simple domain guard such as `if x == 0: return ...` and can unwrap a simple `try` body whose `except` branch is only a fallback. It returns `not_checkable` for unsupported function calls, complex control flow, missing returns, parse failures, timeouts, or hidden constants that cannot be separated from the functional shape. Each unique submitted/hidden-law pair has a 20-second symbolic timeout.

Important boundaries:

| The checker does | The checker does not |
|---|---|
| Prove exact structural equivalence up to total scale | Decide whether a close decimal approximation should count as recovering a named constant or exponent |
| Detect wrong variables, exponents, operators, and terms when symbolically expressible | Infer equivalence by evaluating a finite numerical grid |
| Fail closed to `not_checkable` on unsupported constructs | Treat every local variable as a freely adjustable constant |
| Treat malformed completed submissions as part of the scientific denominator | Mix runner/infrastructure failure stubs into scientific accuracy |

### 3.2 Scoring cascade

All trial trees were first rejudged by an independently hosted `gemma4-31b`. The final pilot label is then assigned as follows:

| Deterministic result | Independent judge | Pilot label |
|---|---|---|
| `constant_equivalent` | either | pass |
| `structurally_different` | either | fail |
| `not_checkable` | equivalent | pass via independent-judge fallback |
| `not_checkable` | not equivalent | fail via independent-judge fallback |
| any result | explicit adjudication exists | use the adjudicated label |

RMSLE remains a diagnostic measure, not the primary decision rule. Completed trajectories with missing, malformed, non-executable, or NaN-producing laws remain in the denominator and normally fail. Separate `trialN_fail.json` files represent runner failures and are reported operationally rather than as scientific attempts.

### 3.3 Checker coverage and conflicts

| Checker outcome | Muse (n=1,727) | Qwen38 (n=1,728) |
|---|---:|---:|
| `constant_equivalent` | 776 (44.9%) | 1,002 (58.0%) |
| `structurally_different` | 383 (22.2%) | 358 (20.7%) |
| `not_checkable` | 568 (32.9%) | 368 (21.3%) |
| Judge/checker conflicts requiring audit | 33 (1.9%) | 92 (5.3%) |

The 125 conflicts comprise 124 cases where Gemma passes a SymPy mismatch and one Qwen38 \$400 case where SymPy passes but Gemma fails. This audit matters because some submissions are close numerical approximations rather than exact identities. Until a tolerance-aware policy is fixed, the independent-judge and deterministic-primary columns below are best read as two documented views of the same pilot.

## 4. Results

All tables in this section accurately summarize the completed run, but they are **debugging results from compromised trajectories**. They should not be used as final estimates of model ability or budget effects.

### 4.1 Accuracy by assigned grant

| Assigned grant | Muse: Gemma judge | Muse: deterministic-primary | Qwen38: Gemma judge | Qwen38: deterministic-primary |
|---:|---:|---:|---:|---:|
| Unbudgeted | 70.5% | **67.7%** | 93.1% | **88.2%** |
| \$1,000 | 55.4% | **52.6%** (n=287) | 85.4% | **80.9%** |
| \$800 | 55.6% | **53.5%** | 84.0% | **78.1%** |
| \$600 | 52.4% | **50.3%** | 82.3% | **76.4%** |
| \$400 | 43.4% | **42.7%** | 75.7% | **69.8%** |
| \$200 | 23.3% | **22.2%** | 41.0% | **36.8%** |

Observed pattern: both models decline under tighter grants, and the largest adjacent decline occurs between \$400 and \$200. The small Muse increase from \$1,000 to \$800 (+0.9 pp) should be treated as sampling variation unless it replicates.

The unbudgeted comparison changes more than the presence of a dollar cap: budgeted observations also use precision-dependent rounding. It is therefore a useful full-system baseline, not a clean estimate of the effect of the cap alone.

### 4.2 Accuracy by difficulty

#### Muse

| Assigned grant | Easy SA (n) | Medium SA (n) | Hard SA (n) |
|---:|---:|---:|---:|
| Unbudgeted | 93.1% (72) | 70.8% (144) | 36.1% (72) |
| \$1,000 | 87.5% (72) | 60.4% (144) | 1.4% (71) |
| \$800 | 93.1% (72) | 56.9% (144) | 6.9% (72) |
| \$600 | 94.4% (72) | 51.4% (144) | 4.2% (72) |
| \$400 | 87.5% (72) | 41.0% (144) | 1.4% (72) |
| \$200 | 56.9% (72) | 16.0% (144) | 0.0% (72) |

#### Qwen38

| Assigned grant | Easy SA (n) | Medium SA (n) | Hard SA (n) |
|---:|---:|---:|---:|
| Unbudgeted | 100.0% (72) | 89.6% (144) | 73.6% (72) |
| \$1,000 | 100.0% (72) | 88.9% (144) | 45.8% (72) |
| \$800 | 94.4% (72) | 86.1% (144) | 45.8% (72) |
| \$600 | 93.1% (72) | 86.1% (144) | 40.3% (72) |
| \$400 | 98.6% (72) | 76.4% (144) | 27.8% (72) |
| \$200 | 66.7% (72) | 38.9% (144) | 2.8% (72) |

The clearest stratified observation is that hard tasks degrade before easy tasks. Because each bin contains different modules, the table does not identify which mathematical feature causes the difference. Four repeats per hidden law are also too few for strong claims about individual laws.

### 4.3 Accuracy by agent interface

| Grant | Muse code | Muse vanilla | Difference | Qwen38 code | Qwen38 vanilla | Difference |
|---:|---:|---:|---:|---:|---:|---:|
| Unbudgeted | 57.6% | 77.8% | −20.2 pp | 85.4% | 91.0% | −5.6 pp |
| \$1,000 | 41.7% | 63.6% | −21.9 pp | 82.6% | 79.2% | +3.4 pp |
| \$800 | 43.8% | 63.2% | −19.4 pp | 77.8% | 78.5% | −0.7 pp |
| \$600 | 41.0% | 59.7% | −18.7 pp | 75.0% | 77.8% | −2.8 pp |
| \$400 | 36.8% | 48.6% | −11.8 pp | 68.1% | 71.5% | −3.4 pp |
| \$200 | 18.1% | 26.4% | −8.3 pp | 36.8% | 36.8% | 0.0 pp |

Muse vanilla is higher in every measured condition. Qwen38 shows no stable interface advantage. This supports an interface-by-model interaction in the present implementation; it does not show that code access is intrinsically harmful.

### 4.4 Experiments, requests, and realized spend

| Model-agent | \$1,000 experiments / requests | \$800 | \$600 | \$400 | \$200 |
|---|---:|---:|---:|---:|---:|
| Muse code | 18 / 4.8 | 17 / 4.4 | 14 / 3.8 | 12 / 3.0 | 6 / 1.7 |
| Muse vanilla | 21 / 5.9 | 20 / 5.2 | 16 / 4.3 | 13 / 3.3 | 6 / 1.7 |
| Qwen38 code | 13 / 2.9 | 11 / 2.8 | 10 / 2.6 | 7 / 2.0 | 4 / 1.3 |
| Qwen38 vanilla | 15 / 3.6 | 13 / 3.3 | 11 / 3.0 | 7 / 2.3 | 4 / 1.3 |

| Grant | Muse mean spend | Muse utilization | Qwen38 mean spend | Qwen38 utilization |
|---:|---:|---:|---:|---:|
| \$1,000 | \$796 | 79.6% | \$882 | 88.2% |
| \$800 | \$685 | 85.6% | \$723 | 90.4% |
| \$600 | \$532 | 88.7% | \$546 | 91.0% |
| \$400 | \$366 | 91.5% | \$367 | 91.8% |
| \$200 | \$180 | 90.0% | \$176 | 88.0% |

No budgeted trial overspent. Both models reduce experiments and requests as the assigned grant tightens. Qwen38 buys fewer observations than Muse at matched grants while attaining higher accuracy, but this aggregate comparison mixes task difficulty, interface, precision, and experiment placement; it is not yet a controlled efficiency metric.

### 4.5 Precision choices and rejected requests

The next table reports requested data points by precision level. It counts parseable `<run_experiment>` blocks in the saved assistant history, including requests that were subsequently rejected for cost.

| Model | Grant | Precision 1 | Precision 2 | Precision 3 | Precision 4 | Precision 5 | Valid requested points | Unparseable experiment blocks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Muse | \$1,000 | 68.9% | 4.3% | 18.0% | 8.2% | 0.6% | 6,449 | 152 |
| Muse | \$800 | 69.1% | 7.7% | 17.5% | 5.4% | 0.3% | 6,008 | 137 |
| Muse | \$600 | 71.5% | 13.7% | 10.0% | 4.7% | 0.0% | 5,314 | 134 |
| Muse | \$400 | 82.7% | 9.7% | 5.3% | 2.3% | 0.0% | 4,260 | 103 |
| Muse | \$200 | 92.2% | 5.4% | 0.4% | 2.0% | 0.0% | 2,562 | 126 |
| Qwen38 | \$1,000 | 9.2% | 10.6% | 47.1% | 28.3% | 4.8% | 4,348 | 231 |
| Qwen38 | \$800 | 11.9% | 15.2% | 47.6% | 20.2% | 5.1% | 3,770 | 240 |
| Qwen38 | \$600 | 13.3% | 29.7% | 42.7% | 12.3% | 2.0% | 3,074 | 235 |
| Qwen38 | \$400 | 21.1% | 30.0% | 37.6% | 10.2% | 1.2% | 2,223 | 214 |
| Qwen38 | \$200 | 46.0% | 34.7% | 13.4% | 5.2% | 0.7% | 1,499 | 226 |

Both models request lower precision as grants tighten, but their policies differ markedly: Muse is dominated by precision 1 throughout, whereas Qwen38 favors precision 3 at larger grants. These are requested, not necessarily purchased, points; the table does not show whether either allocation is efficient.

The following cells report the percentage of trials with at least one unaffordable request. Difficulty order is easy / medium / hard.

| Grant | Muse: trials with rejection | Qwen38: trials with rejection |
|---:|---:|---:|
| \$1,000 | 31.9% / 25.0% / 23.9% | 6.9% / 9.0% / 9.7% |
| \$800 | 31.9% / 33.3% / 20.8% | 13.9% / 13.2% / 9.7% |
| \$600 | 48.6% / 41.0% / 22.2% | 27.8% / 25.0% / 15.3% |
| \$400 | 34.7% / 47.9% / 18.1% | 34.7% / 34.0% / 27.8% |
| \$200 | 44.4% / 46.5% / 37.5% | 44.4% / 47.2% / 31.9% |

Rejections become more common as Qwen38's grant tightens; Muse is already rejection-prone at larger grants and is not monotonic at every step. Hard tasks often have the lowest rejection rate despite the lowest accuracy. Request rejection should therefore be treated as a policy diagnostic rather than a proxy for task difficulty or failure.

### 4.6 Inference tokens

| Grant | Muse code | Muse vanilla | Qwen38 code | Qwen38 vanilla |
|---:|---:|---:|---:|---:|
| Unbudgeted | 7.9k | 19.0k | 17.3k | 22.0k |
| \$1,000 | 10.3k | 20.4k | 21.8k | 34.2k |
| \$800 | 10.0k | 19.8k | 22.7k | 35.1k |
| \$600 | 10.7k | 18.0k | 26.9k | 36.6k |
| \$400 | 11.9k | 18.4k | 25.7k | 38.2k |
| \$200 | 12.4k | 17.3k | 32.3k | 39.1k |

Qwen38 uses more tokens as experimental resources tighten, particularly with the code-assisted interface. This is an observed substitution pattern, not evidence that more reasoning causes failure: difficult trajectories can both last longer and fail more often.

## 5. Trajectory diagnostics

### 5.1 Rejected actions and reasoning-channel separation

The diagnostic marks a trial when a user/tool feedback message contains `Invalid response`, `Action Reminder`, or `exactly 1 action per turn`. These are feedback emitted by the harness after it fails to recognize an action. They should not be labeled uniformly as model formatting errors.

| Condition | Muse: affected trials | Muse feedback events: invalid / reminder | Muse SA with / without | Qwen38: affected trials | Qwen38 feedback events: invalid / reminder | Qwen38 SA with / without |
|---:|---:|---:|---:|---:|---:|---:|
| Unbudgeted | 105/288 (36.5%) | 222: 80 / 142 | 65.7% / 68.9% | 79/288 (27.4%) | 136: 54 / 82 | 75.9% / 92.8% |
| \$1,000 | 125/287 (43.6%) | 259: 110 / 149 | 47.2% / 56.8% | 161/288 (55.9%) | 283: 89 / 194 | 73.9% / 89.8% |
| \$800 | 110/288 (38.2%) | 187: 95 / 92 | 51.8% / 54.5% | 160/288 (55.6%) | 298: 87 / 211 | 72.5% / 85.2% |
| \$600 | 131/288 (45.5%) | 257: 105 / 152 | 42.7% / 56.7% | 133/288 (46.2%) | 248: 73 / 175 | 68.4% / 83.2% |
| \$400 | 127/288 (44.1%) | 261: 82 / 179 | 38.6% / 46.0% | 132/288 (45.8%) | 224: 46 / 178 | 60.6% / 77.6% |
| \$200 | 143/288 (49.7%) | 295: 90 / 205 | 16.8% / 27.6% | 95/288 (33.0%) | 147: 27 / 120 | 32.6% / 38.9% |

The extracted histories expose a concrete interface failure. The vLLM response separates `reasoning` from `content`; the harness records them together for diagnostics but executes only the main `content` response. In the examples below, the model placed a syntactically valid action in `reasoning`, while `content` was empty or only repeated `**Main Response:**`. The action was therefore visible in the saved history but invisible to the action handler.

| Model | Condition | Trials with ≥1 recoverable reasoning action | Recoverable turns | Followed by rejection feedback | Experiment | Python | Final law |
|---|---:|---:|---:|---:|---:|---:|---:|
| Muse | Unbudgeted | 94/288 (32.6%) | 112 | 69 (61.6%) | 63 | 37 | 12 |
| Muse | \$1,000 | 105/287 (36.6%) | 122 | 68 (55.7%) | 62 | 45 | 15 |
| Muse | \$800 | 114/288 (39.6%) | 131 | 72 (55.0%) | 54 | 62 | 15 |
| Muse | \$600 | 102/288 (35.4%) | 118 | 77 (65.3%) | 56 | 38 | 24 |
| Muse | \$400 | 90/288 (31.2%) | 109 | 67 (61.5%) | 49 | 44 | 16 |
| Muse | \$200 | 96/288 (33.3%) | 124 | 86 (69.4%) | 51 | 53 | 20 |
| Qwen38 | Unbudgeted | 70/288 (24.3%) | 114 | 112 (98.2%) | 80 | 31 | 3 |
| Qwen38 | \$1,000 | 150/288 (52.1%) | 238 | 237 (99.6%) | 149 | 82 | 7 |
| Qwen38 | \$800 | 139/288 (48.3%) | 246 | 245 (99.6%) | 149 | 91 | 6 |
| Qwen38 | \$600 | 118/288 (41.0%) | 199 | 196 (98.5%) | 115 | 72 | 12 |
| Qwen38 | \$400 | 110/288 (38.2%) | 177 | 176 (99.4%) | 96 | 72 | 9 |
| Qwen38 | \$200 | 58/288 (20.1%) | 86 | 86 (100.0%) | 36 | 45 | 5 |

A recoverable turn has no syntactically valid action in main content and exactly one syntactically valid allowed action in reasoning. The action-type columns partition recoverable turns and therefore sum to the `Recoverable turns` column. `Followed by rejection feedback` is the strongest direct evidence that the harness failed to execute the recoverable action.

Across the full pilot, 1,246/3,455 trials (36.1%) contain at least one such turn. The audit finds 1,776 recoverable turns: 960 experiment actions, 672 Python actions, and 144 final laws. Rejection feedback immediately follows 1,491/1,776 (84.0%).

The failure is both large and condition-dependent. Muse's affected-trial rate is comparatively stable at 31.2–39.6%. Qwen38 ranges from 20.1% at \$200 and 24.3% unbudgeted to 48.3% at \$800 and 52.1% at \$1,000. Nearly every Qwen38 recoverable turn receives rejection feedback. Consequently, the current Qwen38 resource curve is especially confounded; fixing the bug could change both its level and its shape.

#### Trace 2 — a well-formed experiment rejected as invalid

*Muse, vanilla, gravity, medium/v0, unbudgeted, trial 0.* The model explained that varying distance over 500, 1,000, 2,000, and 4,000 would test a proposed \(d^{-1.5}\) dependence. It then emitted five well-formed JSON experiments; this display keeps the first two:

```text
<run_experiment>
[
  {"mass1":1e12,"mass2":1e6,"distance":500,
   "initial_velocity":0,"duration":0.2,"time_step":0.2},
  {"mass1":1e12,"mass2":1e6,"distance":1000,
   "initial_velocity":0,"duration":0.2,"time_step":0.2},
  ... three further valid objects ...
]
</run_experiment>
```

The saved main-content field was empty, so the harness did not see the block in reasoning:

```text
Invalid response. Please use <run_experiment> tag with the correct JSON format
or <final_law> tag to submit the law.
```

This is not a malformed-JSON example. It is a valid experimental design lost at the reasoning/content boundary.

#### Trace 3 — local analysis written correctly but never executed

*Qwen38, code-assisted, gravity, medium/v0, unbudgeted, trial 2.* After collecting observations over many orders of magnitude, the model attempted a log–log fit:

```text
<python>
logs = [(math.log(d), math.log(abs(v/t)))
        for d, v, t in zip(distances, v1, dt)]
slope = (n*sxy - sx*sy) / (n*sxx - sx*sx)
print("fitted exponent p =", -slope)
print("fitted C =", math.exp(intercept))
</python>
```

The main response again contained no executable action, and the harness replied:

```text
Action Reminder: Please use exactly 1 action per turn with correct format.
Turn 8: 0/1 Python calls used this turn.
```

The model had selected a sensible analysis and written executable Python, but it never received the fit output. This directly changes the scientific trajectory rather than merely changing how the final answer is formatted.

Separately, the budgeted histories contain 103–152 unparseable Muse experiment blocks and 214–240 unparseable Qwen38 blocks per grant condition. Thus both mechanisms exist: some action blocks are malformed, while complete actions are also lost at the reasoning/content boundary. The two counts can overlap and should not be added.

The runner should preserve raw `reasoning` and `content` separately and implement the conservative recovery rule for experiment, Python, and final-law actions. Multiple, conflicting, or syntactically invalid actions must remain rejected. After unit tests, the full model × grant matrix—including the unbudgeted baseline—must be rerun into new result trees. Because evaluation generation uses temperature 0.4, rerunning only the previously affected trials and pooling them with a post-hoc-selected unaffected subset would not provide a clean replacement sample. Rejudge, scoreboard, or diagnostics alone cannot repair affected trajectories.

### 5.2 Final-law visibility and placeholder submissions

| Condition | Muse: no parseable law, SA | Qwen38: no parseable law, SA |
|---:|---:|---:|
| Unbudgeted | 64, 7.8% | 0, — |
| \$1,000 | 77, 3.9% | 3, 0.0% |
| \$800 | 67, 0.0% | 1, 0.0% |
| \$600 | 65, 0.0% | 3, 0.0% |
| \$400 | 54, 1.9% | 1, 0.0% |
| \$200 | 51, 0.0% | 0, — |

The table is a trace-parser result: it reports whether a `def discovered_law(...)` body was recoverable from the main-response portion of the saved history. It is not a count of invalid stored submissions.

The following traces show two distinct cases.

#### Trace 4 — evidence acquired, but no final law submitted

*Muse, code-assisted, gravity, medium/v1, unbudgeted, trial 2.* The final prompt required an immediate law submission. The model recognized the limit and summarized a plausible result:

> “We have used all experiment turns ... so we must submit the final law now. We have strong evidence: force magnitude = `C * mass1 / distance**2.6`.”

It nevertheless used the final response for another calculation:

```text
<python>
a, b = np.linalg.lstsq(A, v, rcond=None)[0]
print(a)
pred = C * 3**(-2.6)
print(pred)
print(a/pred)
</python>
```

Because no `<final_law>` was submitted after the forced-final prompt, the runner stored its fallback:

```python
def discovered_law(mass1, mass2, distance):
    return float('nan')
```

Here the failure is not absence of scientific progress. It is failure to switch from further verification to the required terminal action.

#### Trace 5 — a complete law in reasoning, but `None` in main content

*Qwen38, code-assisted, BE distribution, medium/v2, \$1,000, trial 0.* The model remained uncertain about one observation but committed to a temperature-cubed exponential law:

> “Given forced final, I will submit the simple \(T^3\) Bose law.”

```text
<final_law>
def discovered_law(omega, T):
    import math
    Constant = 3.3247976403434714e-8
    x = Constant * omega / (T ** 3)
    if x <= 0: return 0.0
    if x < 1e-5: return 1.0 / x - 0.5 + x / 12.0
    if x > 700: return 0.0
    return 1.0 / math.expm1(x)
</final_law>
```

The raw main response was `None`. The runner's full-history extractor nevertheless recovered and scored the function, whereas the trace diagnostic—restricted to main-response text—marked the turn as having no parseable law. This explains why `no_parseable_law` cannot be equated with an invalid stored submission.

For publication, these categories must remain separate: absent final tag, final law stranded in reasoning but recovered, forced NaN fallback, syntactically invalid Python, non-finite executable law, and valid-but-incorrect law. The current `no_parseable_law` aggregate should be used only as a trace-visibility diagnostic.

### 5.3 Realized-spend quartiles

Each row below partitions trials *within one assigned-grant condition* by realized spending. A cell reports deterministic-primary SA followed by the bin size.

#### Muse

| Assigned grant | Q1 cheapest SA (n) | Q2 SA (n) | Q3 SA (n) | Q4 most expensive SA (n) |
|---:|---:|---:|---:|---:|
| \$1,000 | 26.4% (72) | 61.1% (72) | 64.8% (71) | 58.3% (72) |
| \$800 | 36.1% (72) | 54.7% (86) | 62.3% (61) | 62.3% (69) |
| \$600 | 30.6% (72) | 67.5% (77) | 50.6% (77) | 51.6% (62) |
| \$400 | 25.6% (86) | 50.8% (59) | 62.8% (94) | 24.5% (49) |
| \$200 | 19.8% (81) | 17.9% (78) | 27.1% (118) | 18.2% (11) |

#### Qwen38

| Assigned grant | Q1 cheapest SA (n) | Q2 SA (n) | Q3 SA (n) | Q4 most expensive SA (n) |
|---:|---:|---:|---:|---:|
| \$1,000 | 82.2% (73) | 83.9% (87) | 80.7% (57) | 76.1% (71) |
| \$800 | 90.3% (72) | 68.0% (75) | 78.6% (84) | 75.4% (57) |
| \$600 | 81.1% (74) | 84.3% (70) | 69.3% (75) | 71.0% (69) |
| \$400 | 78.4% (74) | 64.7% (85) | 70.5% (61) | 66.2% (68) |
| \$200 | 27.4% (73) | 33.8% (71) | 42.6% (101) | 44.2% (43) |

The bins are unequal because spending takes repeated discrete values. Quantile boundaries are computed from the observed spend values, and `pandas.cut` keeps tied values together instead of arbitrarily splitting equal-spend trials. This can produce a bin such as Muse \$200 Q4 with only 11 trials.

These tables are descriptive. Realized spending is chosen by the agent after seeing the task and observations: easy trials may stop cheaply, difficult trials may spend more and still fail, and two equal-cost campaigns may buy differently informative experiments. The assigned-grant table in Section 4.1 is the appropriate starting point for studying the budget intervention; the realized-spend table cannot estimate the causal benefit of spending another dollar.

## 6. What the pilot currently establishes

| Observation in the recorded runs | Important limitation |
|---|---|
| Both models score lower under tighter hard grants | Differential action-channel loss prevents a clean estimate of the intended budget effect |
| The \$200 condition is much worse than \$400 | The numerical threshold is specific to this synthetic price schedule |
| Hard tasks degrade more sharply than easy tasks | Difficulty bins contain different laws and system types |
| Muse vanilla exceeds Muse code-assisted | The result may depend on these prompts and implementations |
| Qwen38 increases token use as experimental access falls | Longer reasoning may be a consequence of difficult cases, not a cause of failure |
| Rejected-action feedback correlates with lower SA in most cells | Some apparently valid actions are stranded by reasoning/content separation; the association is not adjusted for task difficulty or trajectory length |
| Qwen38 obtains higher SA with fewer mean observations | Observation count alone does not measure information quality or cost efficiency |

The pilot validates the result-processing and scoring workflow and identifies useful behavioral measurements. It does **not yet** provide publication-ready accuracy curves. After the action-channel repair and rerun, further controls are still needed to show adaptive feedback use, policy quality, cost-schedule robustness, or generalization beyond the two evaluated models.

## 7. Research context

| Research area | What is usually evaluated | What this pilot adds |
|---|---|---|
| Interactive scientific agents | hypothesis–experiment–conclusion loops | heterogeneous costs and hard resource limits |
| Symbolic regression | equation recovery from a fixed dataset | the upstream decision of which observations to acquire |
| Optimal experimental design | information gained per action or cost | an LLM agent must also express an executable symbolic law |
| Test-time scaling | allocation of tokens, rollouts, or tool calls | simultaneous measurement of inference tokens and purchased evidence |

The closest novelty risk is work that already studies active experiment selection on NewtonBench. A paper contribution will therefore require controls showing whether the agent uses observation feedback and baselines showing how far its acquisitions are from simple fixed or classical designs. The current pilot establishes the measurement setting and the first resource-response curves; it does not yet establish adaptive or optimal experimental design.

## Sources

1. [NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](https://arxiv.org/abs/2510.07172) — parent benchmark and counterfactual-law motivation.
2. [Modern Bayesian Experimental Design](https://doi.org/10.1214/23-STS915) — background on selecting informative experiments.
3. [LLM-AutoSciLab: Closed-Loop Scientific Discovery via Active Experimentation with LLMs](https://arxiv.org/abs/2605.24043) — close active-discovery comparison.
4. [LLMs for Experiment Design in Scientific Domains: Are We There Yet?](https://openreview.net/forum?id=dIEeOwrmOe) — motivation for feedback controls and classical baselines.
5. [DiscoveryWorld](https://proceedings.neurips.cc/paper_files/paper/2024/file/13836f251823945316ae067350a5c366-Paper-Datasets_and_Benchmarks_Track.pdf) — neighboring benchmark for simulated experimental discovery.
