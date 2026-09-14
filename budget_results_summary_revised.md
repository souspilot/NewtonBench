# Resource-Constrained Scientific Discovery: Preliminary Results Across Budget Regimes

> **Status (14 September 2026).** This report separates two interventions that
> were previously conflated:
>
> 1. **Hard-capped convex budgets** (`budget_{200,400,600,800,1000}.json`), in
>    which unaffordable requests are rejected. Complete budget curves are
>    available for qwen38-27b (288/288 trials at every budget) and nearly complete
>    for muse-glimmer-30b (286/288 at $1,000; 283/288 at $800; 287/288 at $600;
>    288/288 at $400 and $200).
> 2. **Legacy soft/uncapped $1,000 runs**, in which agents could overdraw the
>    grant. These include qwen38-27b, qwq-32b, and an earlier partial Muse run.
>    They are retained as a separate historical ablation and are not pooled with
>    the hard-cap curves.
>
> All reported SA values are **raw symbolic accuracy from the benchmark's LLM
> judge**, unless explicitly marked verified. The representative subset contains
> one difficulty-system cell per domain, evaluated at all three law versions and
> four trials per version: 144 trials per agent and 288 per model when complete.

## Executive summary

This project extends NewtonBench from unconstrained interactive law discovery to
**cost-sensitive mechanism discovery**. An agent receives a finite research
grant and must decide not only *which* interventions to perform, but also how to
allocate funds among setup complexity, sample volume, parameter range, and
measurement precision.

The new $200-$1,000 hard-cap curves produce a stronger result than the earlier
single-budget comparison:

- Scientific-law recovery exhibits a pronounced **budget-response curve**, with
  the largest loss concentrated below roughly $400 in the present cost units.
- The two strong models occupy different **accuracy-resource frontiers**.
  Qwen38 is substantially more robust across the curve; Muse remains competitive
  at high budgets but degrades earlier and more sharply.
- Agent architecture interacts with budget. Code assistance is usually helpful
  for Muse and usually harmful for Qwen38, but Qwen38 reverses at $1,000. Because
  several neighboring points are non-monotonic, this interaction needs repeated
  seeds before it should be treated as a stable crossover.
- Tighter experimental budgets do not necessarily reduce total inference cost.
  Qwen38 performs fewer experiments but generally uses more tokens as the grant
  shrinks, suggesting that scarcity transfers effort from data acquisition to
  internal reasoning.
- Within-budget spend quartiles are **descriptive, not causal**. They mix task
  difficulty, domain, agent type, and policy behavior; unequal success rates do
  not show that spending itself caused success.

The immediate research opportunity is to turn these curves into a benchmark of
**resource-rational scientific agency**: how well an agent converts heterogeneous
experimental resources into correct mechanistic knowledge.

---

## 1. Scientific intervention

NewtonBench asks agents to recover counterfactual physical laws by actively
probing simulated experimental systems. Its central advance over passive
symbolic regression is the closed loop between hypothesis formation, experiment
selection, observation, and revision. The original benchmark reports that law
recovery becomes fragile under system complexity and observational noise, and
that code tools can alter the exploration-exploitation balance
([Zheng et al., 2026](https://arxiv.org/html/2510.07172v3)).

We add a second decision problem to every experimental action:

> Given several possible experiments with different costs and fidelities, which
> evidence is worth purchasing before the grant is exhausted?

This shifts the object of evaluation from unconstrained discovery accuracy to a
joint assessment of:

1. hypothesis quality;
2. intervention choice;
3. measurement-fidelity choice;
4. batch and setup efficiency;
5. stopping behavior; and
6. final mechanism recovery.

This framing is closely connected to optimal experimental design (OED), which
chooses observations to maximize scientific utility, often information gain.
Modern Bayesian experimental design explicitly treats experiment selection as
a sequential decision problem
([Rainforth et al., 2024](https://doi.org/10.1214/23-STS915)). It also recovers a
classic distinction: model-discrimination experiments should be selected where
rival mechanisms make divergent predictions, rather than merely where a single
model predicts uncertain outputs
([Box and Hill, 1967](https://doi.org/10.1080/00401706.1967.10490441)).

The resource model adds a feature that ordinary fixed-query benchmarks omit:
actions are not exchangeable. Experimental-design work has long considered
measurement costs and transition/setup costs, including optimizing information
per unit cost
([Tack and Vandebroek, 2001](https://doi.org/10.1016/S0378-3758(00)00299-8))
and balancing scientific value against the resources required by complete or
reduced designs
([Collins, Dziak and Li, 2009](https://doi.org/10.1037/a0015826)). More recent
cost-aware Bayesian optimization likewise studies settings where evaluation cost
varies with the chosen action
([Xie et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/d14c355d5e88cff437a6303d2d716252-Paper-Conference.pdf)),
while multi-fidelity optimization formalizes trading cheap low-fidelity evidence
against expensive high-fidelity evidence
([Foumani et al., 2023](https://doi.org/10.1016/j.cma.2023.115937)).

### What is new relative to concurrent active-discovery work

LLM-AutoSciLab independently evaluates NewtonBench under a fixed query budget
and reports that explicit hypothesis-conditioned acquisition can be 2-5x more
sample-efficient than comparison methods. Its NewtonBench reference condition
uses 20 queries; unless otherwise noted, its benchmark instances are
deterministic and zero-noise
([Kabra et al., 2026](https://arxiv.org/html/2605.24043v1)).

The present intervention is complementary:

| Fixed-query active discovery | This project |
|---|---|
| Each query consumes one budget unit | Actions have heterogeneous monetary cost |
| Main choice: where to sample | Choices: where, how many, what precision, and how extreme a range |
| Measures sample efficiency | Measures cost efficiency and allocation strategy |
| Budget controls quantity | Budget jointly controls quantity and measurement fidelity |
| Usually spends the full query allowance | Allows early stopping, underspending, and rejected requests |

The strongest novelty claim is therefore not simply “NewtonBench with a
budget.” It is a configurable environment for studying **heterogeneous-cost,
multi-fidelity experimental planning by LLM agents**.

### Connection to autonomous laboratories

Self-driving laboratories close the loop between algorithmic experiment
selection and automated execution, particularly in chemistry and materials
science
([Abolhasani and Kumacheva, 2023](https://www.nature.com/articles/s44160-022-00231-0)).
Deployed systems already face expensive actions, noisy feedback, feasibility
constraints, instrument limits, and the need for reproducible decision logs.
Recent work consequently argues that agent evaluations for self-driving labs
should include cost-aware performance, safety/constraint behavior, robustness,
and provenance—not correctness alone
([Chen et al., 2026](https://arxiv.org/abs/2601.17920)).

NewtonBench remains a simulator with synthetic costs, not a laboratory model.
Its value is as a controlled precursor: it lets us isolate whether an agent can
reason about scientific value per unit resource before adding hardware failure,
safety, delay, and domain-specific logistics.

---

## 2. Cost model and experimental conditions

| Parameter | Value |
|---|---:|
| Hard-cap grants | $200, $400, $600, $800, $1,000 per trial |
| Per-datapoint base cost | $15 |
| Batch pricing | Marginal convex tiers: 1-5 points at 1x, 6-10 at 2x, 11-15 at 3.5x, 16-20 at 6x |
| Setup fee | $30 per input parameter varied within a request |
| Precision levels | 1-5 |
| Significant figures | 1, 2, 3, 4, 6 |
| Precision multipliers | 1x, 1.6x, 2.5x, 4x, 6.5x |
| Default precision | p1: 1 significant figure |
| Parameter-range surcharge | 15% per decade outside $10^{-3}$ to $10^3$ |
| Hard-cap rule | Requests exceeding remaining funds are rejected; two consecutive rejections force submission |
| Noise in current runs | Zero, with precision-dependent rounding still active |
| Evaluated tasks | 12-domain representative subset |
| Trials per law version | 4 |

The coefficients are a synthetic benchmark parameterization. They should not be
interpreted as calibrated laboratory dollars. The scientific claim concerns the
agent's response to a transparent cost structure and the robustness of rankings
across alternative structures.

Two design choices deserve explicit attention:

- The convex batch schedule encourages small requests. It measures whether an
  agent adapts to that incentive, but it also makes “small batches” partly a
  prompt-following result rather than an emergent optimum.
- Because the default sensor returns only one significant figure even at zero
  noise, comparison with the original unconstrained condition confounds scarcity
  with observation fidelity. Comparisons *among* the five capped budgets hold
  default precision fixed and are cleaner.

---

## 3. Primary result: hard-cap accuracy curves

### 3.1 Symbolic accuracy by budget

| Grant | Muse code | Muse vanilla | Qwen38 code | Qwen38 vanilla |
|---:|---:|---:|---:|---:|
| Unconstrained | 96.2% (144) | 88.5% (144) | 94.8% (144) | 99.2% (144) |
| $1,000 cap | 86.4% (144) | 73.0% (142) | 93.9% (144) | 87.9% (144) |
| $800 cap | 84.1% (139) | 77.9% (144) | 84.3% (144) | 92.9% (144) |
| $600 cap | 73.1% (143) | 68.4% (144) | 82.6% (144) | 87.3% (144) |
| $400 cap | 68.7% (144) | 54.1% (144) | 77.8% (144) | 86.5% (144) |
| $200 cap | 28.4% (144) | 29.0% (144) | 41.5% (144) | 47.0% (144) |

Numbers in parentheses are observed trials. Unconstrained results use the
original high-precision observation regime and are context rather than a pure
infinite-budget endpoint.

### 3.2 What the curves show

**A pronounced low-budget cliff.** From $400 to $200, Muse loses 40.3 percentage
points with code and 25.1 points without it; Qwen38 loses 36.3 and 39.5 points.
At $200, agents average only 4-6 measured datapoints across roughly 1-2 billed
requests. For many shifted laws, that is insufficient to distinguish structural
alternatives.

**Different efficiency frontiers.** Qwen38 dominates Muse at most matched
budgets. At $400, Qwen38 reaches 77.8%/86.5% versus Muse's 68.7%/54.1%; at $200,
the corresponding values are 41.5%/47.0% versus 28.4%/29.0%. This is stronger
evidence of resource robustness than a single $1,000 comparison.

**A broad middle regime rather than smooth linear scaling.** Between $400 and
$800, Qwen38 changes relatively modestly, especially in vanilla mode. Muse gains
more across this range. The response is not perfectly monotone—Muse vanilla is
higher at $800 than $1,000, and Qwen38 vanilla is higher at $800 than $1,000.
These reversals are compatible with trial stochasticity, raw-judge variation,
and small per-cell samples; they should not be interpreted as evidence that less
budget improves performance without replicated runs.

**Hard laws fail first.** At low budgets, simple/easy cells often remain near
100%, while difficult vanilla equations and difficult complex systems collapse.
For example, at $200 both models retain high scores on several simple systems,
but hard direct-equation and hard complex-system cells are often near zero. This
is consistent with a structural-identifiability account: scarce low-precision
measurements preserve obvious dependencies but fail to resolve fractional
exponents, multi-term forms, and embedded dynamics.

### 3.3 Agent-type interactions

| Grant | Muse code-minus-vanilla | Qwen38 code-minus-vanilla |
|---:|---:|---:|
| Unconstrained | +7.7 pp | -4.4 pp |
| $1,000 | +13.4 pp | +6.0 pp |
| $800 | +6.2 pp | -8.6 pp |
| $600 | +4.7 pp | -4.7 pp |
| $400 | +14.6 pp | -8.7 pp |
| $200 | -0.6 pp | -5.5 pp |

Muse usually benefits from code assistance, although the advantage disappears
when the $200 grant permits too little evidence for either agent. Qwen38 usually
performs better without code, matching NewtonBench's observation that tools can
induce premature exploitation in capable models. Its +6-point code advantage at
$1,000 is an isolated crossover and needs repeated seeds before being elevated
to a substantive finding.

The important supported conclusion is therefore an **architecture-by-resource
interaction**, not a universal claim that code helps or hurts.

---

## 4. How agent behavior changes with the grant

### 4.1 Experimental volume and request structure

| Model-agent | $1,000: points / requests | $800 | $600 | $400 | $200 |
|---|---:|---:|---:|---:|---:|
| Muse code | 18 / 4.8 | 17 / 4.4 | 14 / 3.8 | 12 / 3.0 | 6 / 1.7 |
| Muse vanilla | 21 / 5.9 | 20 / 5.2 | 16 / 4.3 | 13 / 3.3 | 6 / 1.7 |
| Qwen38 code | 13 / 2.9 | 11 / 2.8 | 10 / 2.6 | 7 / 2.0 | 4 / 1.3 |
| Qwen38 vanilla | 15 / 3.6 | 13 / 3.3 | 11 / 3.0 | 7 / 2.3 | 4 / 1.3 |

Both models respond rationally in the narrow sense that smaller grants reduce
datapoints and requests. Typical request sizes remain about 3-5 points, close to
the cheapest batch tier. Muse purchases more observations than Qwen38 at matched
budgets, yet usually obtains lower accuracy. This suggests that the benchmark is
capturing more than sample count: experiment placement, hypothesis quality, and
use of measurements matter.

### 4.2 Budget utilization

| Grant | Muse code spend | Muse vanilla spend | Qwen38 code spend | Qwen38 vanilla spend |
|---:|---:|---:|---:|---:|
| $1,000 | $710 (71%) | $882 (88%) | $869 (87%) | $895 (90%) |
| $800 | $640 (80%) | $727 (91%) | $709 (89%) | $737 (92%) |
| $600 | $505 (84%) | $562 (94%) | $537 (90%) | $555 (93%) |
| $400 | $358 (90%) | $374 (94%) | $368 (92%) | $366 (92%) |
| $200 | $178 (89%) | $182 (91%) | $179 (90%) | $173 (87%) |

All hard-cap runs correctly report 0% overspending. Utilization generally rises
as grants become tighter. Muse code leaves considerably more money unused at
$1,000 than the other agents, which may indicate early confidence, inability to
construct a useful affordable follow-up, or a stopping-policy difference.
Transcript analysis is required to distinguish these explanations.

### 4.3 Inference tokens move differently from experimental resources

| Grant | Muse code | Muse vanilla | Qwen38 code | Qwen38 vanilla |
|---:|---:|---:|---:|---:|
| Unconstrained | 7,899 | 19,006 | 17,288 | 21,960 |
| $1,000 | 10,312 | 20,418 | 21,829 | 34,188 |
| $800 | 10,035 | 19,835 | 22,725 | 35,053 |
| $600 | 10,742 | 17,993 | 26,902 | 36,571 |
| $400 | 11,924 | 18,357 | 25,703 | 38,225 |
| $200 | 12,441 | 17,327 | 32,341 | 39,114 |

Qwen38's experimental budget falls while its inference usage rises. From $1,000
to $200, measured datapoints fall from 13 to 4 for code-assisted Qwen38, while
tokens rise from 21.8k to 32.3k; vanilla datapoints fall from 15 to 4 while
tokens rise from 34.2k to 39.1k. This is evidence of a resource substitution:
when external evidence is scarce, the agent spends more computation reasoning
over less data.

It is not yet evidence that additional reasoning *causes* failure. Difficult
tasks can simultaneously demand more reasoning and yield lower accuracy. A
within-task or mixed-effects analysis is needed.

---

## 5. Why spend-quartile accuracy differs

“SA% vs. budget-spent quartile” sorts trials by **realized spend** within one
run and reports mean success in each bin. The quartiles are not randomized
treatment groups, so there is no reason their SA values should be equal.

Several processes create differences:

1. **Task difficulty confounding.** Hard tasks may trigger additional experiments
   and still fail, producing high-spend/low-accuracy trials.
2. **Adaptive spending.** A capable agent may stop cheaply when evidence is
   decisive but spend more when hypotheses remain ambiguous.
3. **Strategy quality.** Two trials can spend the same amount on experiments of
   very different diagnostic value.
4. **Agent and domain composition.** The displayed table pools code-assisted and
   vanilla agents and all modules. If those groups have different spend
   distributions and base accuracies, aggregate bins can show Simpson's-paradox
   behavior.
5. **Discrete prices and ties.** Spending takes a limited set of values. The code
   computes 25th/50th/75th-percentile boundaries and then uses `pandas.cut`.
   Every trial tied at a boundary stays in the same bin, so bin sizes need not be
   25% each. This explains outputs such as 81/78/118/11 rather than 72/72/72/72.
6. **Raw judge noise.** These are raw LLM-judge labels, adding evaluation
   variability.

Consequently, a rising quartile curve does **not** establish that spending more
would cause higher accuracy, and a falling curve does not establish diminishing
returns. The causal budget evidence comes from the externally assigned grant
curves in Section 3, because grant size changes the available resource before
the agent acts.

For descriptive within-budget analysis, report adjusted estimates from a model
such as

\[
\Pr(\text{success}) = \operatorname{logit}^{-1}(\beta_0 + \beta_1
\text{spend fraction} + \text{difficulty} + \text{system} + \text{module} +
\text{agent}),
\]

with task/configuration-level clustered uncertainty. Even then, realized spend
is endogenous, so the coefficient should be called an association rather than a
causal effect.

---

## 6. Legacy soft-cap results: a separate ablation

The earlier $1,000 results allowed agents to overdraw their grants. They answer a
different question: how agents behave when budget feedback is visible but not
enforced.

| Model-agent | Unconstrained SA | Legacy soft-cap SA | Change | Mean spend | Overspent |
|---|---:|---:|---:|---:|---:|
| Muse code | 96.2% | 92.3% | -3.9 pp | $780 | 30.2% |
| Muse vanilla | 88.5% | 75.2% | -13.3 pp | $974 | 44.0% |
| Qwen38 code | 94.8% | 90.8% | -4.0 pp | $887 | 9.0% |
| Qwen38 vanilla | 99.2% | 95.8% | -3.4 pp | $918 | 13.2% |
| Qwq code | 50.9% | 28.4% | -22.5 pp | $327 | 1.4% |
| Qwq vanilla | 36.2% | 32.2% | -4.0 pp | $499 | 9.7% |

These values should not be compared point-for-point with the new `$1,000_cap`
condition. A useful future experiment would intentionally retain both policies
under identical seeds:

- **hard budget:** unaffordable requests are rejected;
- **soft budget:** overdrafts are permitted but recorded;
- **advisory budget:** costs are displayed but never constrain execution.

That design would isolate instruction compliance and constraint handling from
the information effect of having fewer measurements.

The old Qwq result remains suggestive: it underspent severely and code assistance
lost its unconstrained advantage. But until Qwq has a complete hard-cap curve,
it should not be placed on the same accuracy-resource frontier as Muse and
Qwen38.

---

## 7. What this project can become

### 7.1 A benchmark of resource-rational scientific agency

Most scientific-agent benchmarks ask whether the final answer or workflow is
correct. For example, ScienceAgentBench evaluates executable programs, results,
and costs on tasks drawn from scientific papers, while finding that the best
tested agent solved only about one third independently
([Chen et al., 2024](https://arxiv.org/abs/2410.05080)). DiscoveryBench evaluates
data-driven discovery workflows and similarly reports substantial remaining
headroom
([Majumder et al., 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0d70af566e69f1dfb687791ecf955e28-Abstract-Conference.html)).

This project can contribute a complementary evaluation layer: not only “was the
law recovered?” but “how efficiently and robustly did the agent acquire the
evidence needed to recover it?”

Recommended headline outputs are:

- the full accuracy-versus-grant curve;
- area under that curve, preferably over log budget;
- minimum grant needed to reach a target SA;
- a Pareto frontier of SA versus realized experimental cost;
- robustness across cost schedules and precision regimes;
- constraint violations/rejections and unused budget;
- experimental and inference resources reported separately.

### 7.2 A controlled testbed for discovery-policy research

The environment can compare:

- unconstrained LLM agents;
- explicit hypothesis-set agents;
- falsification-first prompting;
- random and Latin-hypercube designs;
- one-factor-at-a-time controlled sweeps;
- symbolic-regression backends;
- Bayesian model discrimination;
- information-gain-per-cost policies;
- non-myopic policies that reserve funds for follow-up tests;
- learned policies trained by reinforcement learning.

This matters because recent constrained Bayesian experimental design work treats
experiment selection as an online planning problem with future constraints,
rather than a sequence of independent greedy choices
([Guo et al., 2026](https://arxiv.org/abs/2605.26990)). NewtonBench's
finite, interpretable laws make it possible to compare LLM heuristics against
normative baselines and to inspect *why* the policies differ.

### 7.3 A curriculum for training scientific agents

The independent axes of law complexity, system complexity, noise, grant size,
precision, and cost structure provide a natural curriculum. Agents could be
trained first on direct noiseless laws, then progressively exposed to indirect
systems, tighter resources, and misspecified costs. Reward can combine correct
mechanism recovery with resource use, while keeping final accuracy separately
visible to avoid training agents merely to be cheap.

### 7.4 A bridge to realistic self-driving-lab evaluation

The next realism layers could include:

- elapsed-time costs and delayed results;
- reagent/instrument-specific budgets rather than one fungible currency;
- switching and calibration costs that depend on experiment order;
- sensor drift and failed measurements;
- safety and feasibility constraints;
- multiple fidelity sources with systematic bias;
- hidden or distractor variables;
- competing mechanisms that are indistinguishable in parts of the design space;
- explicit abstention when the available budget cannot identify a law.

These extensions would move the project from a benchmark of cost-sensitive
symbolic identification toward a simulator for laboratory decision policy.
Real autonomous experiments require robustness to epistemic and stochastic
error, reproducibility, and interoperability as well as efficient optimization
([Ren et al., 2023](https://www.nature.com/articles/s41578-023-00588-4)).

---

## 8. Analyses required before strong claims

### Priority 1: verify the outcome labels

Headline results currently rely on the LLM symbolic-equivalence judge. Run the
deterministic structural diagnostics for every model-budget combination and
report raw and verified bands. Because the current diagnostic filename is reused
across budget configs, preserve each output before running the next condition.

```bash
for model in muse-glimmer-30b qwen38-27b; do
  for b in 1000 800 600 400 200; do
    python analysis/diagnostics.py verdicts \
      --model "$model" \
      --budget-config "configs/budget/budget_${b}.json" \
      --subset_file configs/representative_subset.json
    cp "analysis/verdicts_${model}_budget.csv" \
       "analysis/verdicts_${model}_budget_${b}.csv"
  done
done
```

### Priority 2: mine behavior, not only outcomes

Run trace diagnostics to distinguish useful controlled sweeps from format
failures, rejected requests, premature submission, hypothesis churn, Python
errors, and unverified final laws:

```bash
for model in muse-glimmer-30b qwen38-27b; do
  for b in 1000 800 600 400 200; do
    python analysis/diagnostics.py trace \
      --model "$model" \
      --budget-config "configs/budget/budget_${b}.json" \
      --subset_file configs/representative_subset.json
    cp "analysis/trajectory_trace_${model}_budget.csv" \
       "analysis/trajectory_trace_${model}_budget_${b}.csv"
  done
done
```

Add explicit parsing for total rejected requests if it is not already surfaced
in the trace CSV. Under hard caps, rejection rate is a core policy metric.

### Priority 3: quantify uncertainty at the right level

Do not treat all 144 trial rows as independent Bernoulli samples. Trials share
domain, system, difficulty, and law version. Use either:

- a cluster bootstrap over task configurations; or
- a mixed-effects logistic regression with fixed effects for grant, agent,
  difficulty, and system, plus random intercepts for module/law configuration.

Report confidence intervals for every curve and contrasts such as:

- $200 vs. $400;
- $400 vs. $600;
- code vs. vanilla at each budget;
- model-by-budget and agent-by-budget interactions.

### Priority 4: complete and replicate

- Fill Muse's 2/5/1 missing trials at $1,000/$800/$600.
- Run the full hard-cap curve for Qwq if it remains scientifically interesting.
- Repeat the same task manifests with at least two additional stochastic seeds.
  The non-monotone neighboring points and isolated agent crossover make this
  essential.

### Priority 5: separate precision from scarcity

Run a factorial ablation:

| Budget | Default precision | Purpose |
|---:|---:|---|
| Same five grants | p1 | Current curve |
| Same five grants | p3 | Tests whether extra fidelity changes the frontier |
| Unconstrained count cap | p1 | Isolates coarse observations without monetary scarcity |
| Unconstrained count cap | high precision | Original reference |

An even cleaner alternative is to give every trial a small free calibration set
at fixed precision, then charge only for adaptive follow-ups.

### Priority 6: add normative baselines

At minimum, compare against:

1. random feasible designs;
2. controlled one-variable sweeps;
3. space-filling designs;
4. a greedy disagreement-per-cost policy over a candidate-law library; and
5. an oracle planner that knows the candidate family but not the selected law.

The oracle planner supplies an upper bound on achievable information efficiency;
random designs supply a floor. Without these, the curves show model differences
but cannot say how close any model is to efficient experimental design.

---

## 9. Claims currently supported—and claims to avoid

### Supported by the present evidence

- Hard experimental budgets substantially change law-discovery accuracy.
- Accuracy-resource curves differ strongly between model-agent combinations.
- A severe performance cliff appears between $400 and $200 under the current
  synthetic cost schedule.
- Experimental scarcity can increase rather than decrease inference-token use.
- Code assistance interacts with both the underlying model and resource regime.
- Realized spend alone is not an adequate measure of experimental efficiency.

### Not yet supported

- That a particular dollar grant corresponds to a real laboratory budget.
- That $400 is a universal discovery threshold; it is specific to this cost
  parameterization and task subset.
- That spending more within a fixed grant causally increases success.
- That the isolated $1,000 Qwen38 agent crossover is stable.
- That any LLM policy is near-optimal without algorithmic OED baselines.
- That the unconstrained-to-budget gap is caused solely by planning rather than
  lower measurement precision.
- That results generalize to noisy, delayed, unsafe, or hardware-based science.

---

## Preliminary conclusion

> Scientific-discovery accuracy under abundant, uniform-cost observations is
> not equivalent to the ability to acquire mechanistic knowledge efficiently
> under heterogeneous resource constraints.

The five-budget curves show that this is not merely a harsher version of the
same leaderboard. Models adapt differently: Qwen38 usually preserves high
accuracy while reducing experiment count, Muse purchases more evidence but
degrades sooner, and tool effects depend on both model and grant. Below $400,
both systems encounter a sharp identifiability bottleneck.

With verified scoring, replicated curves, precision ablations, and normative
experimental-design baselines, this project can become a rigorous benchmark for
the capability that autonomous science actually needs: choosing evidence whose
expected mechanistic value justifies its cost.

## References

- Abolhasani, M. & Kumacheva, E. (2023). [The rise of self-driving labs in chemical and materials sciences](https://www.nature.com/articles/s44160-022-00231-0). *Nature Synthesis*.
- Box, G. E. P. & Hill, W. J. (1967). [Discrimination Among Mechanistic Models](https://doi.org/10.1080/00401706.1967.10490441). *Technometrics*.
- Chen, Z. et al. (2024). [ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery](https://arxiv.org/abs/2410.05080).
- Chen, X. et al. (2026). [Agentic AI for Self-Driving Laboratories in Soft Matter: Taxonomy, Benchmarks, and Open Challenges](https://arxiv.org/abs/2601.17920).
- Collins, L. M., Dziak, J. J. & Li, R. (2009). [Design of Experiments with Multiple Independent Variables: A Resource Management Perspective](https://doi.org/10.1037/a0015826). *Psychological Methods*.
- Foumani, Z. Z. et al. (2023). [Multi-fidelity cost-aware Bayesian optimization](https://doi.org/10.1016/j.cma.2023.115937). *Computer Methods in Applied Mechanics and Engineering*.
- Kabra, S. et al. (2026). [LLM-AutoSciLab: Closed-Loop Scientific Discovery via Active Experimentation with LLMs](https://arxiv.org/html/2605.24043v1).
- Majumder, B. P. et al. (2025). [DiscoveryBench: Towards Data-Driven Discovery with Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0d70af566e69f1dfb687791ecf955e28-Abstract-Conference.html). ICLR.
- Rainforth, T. et al. (2024). [Modern Bayesian Experimental Design](https://doi.org/10.1214/23-STS915). *Statistical Science*.
- Ren, Z. et al. (2023). [Autonomous experiments using active learning and AI](https://www.nature.com/articles/s41578-023-00588-4). *Nature Reviews Materials*.
- Tack, L. & Vandebroek, M. (2001). [(Dt,C)-optimal run orders](https://doi.org/10.1016/S0378-3758(00)00299-8). *Journal of Statistical Planning and Inference*.
- Xie, Q. et al. (2024). [Cost-aware Bayesian Optimization via the Pandora's Box Gittins Index](https://proceedings.neurips.cc/paper_files/paper/2024/file/d14c355d5e88cff437a6303d2d716252-Paper-Conference.pdf). NeurIPS.
- Zheng, T. et al. (2026). [NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](https://arxiv.org/html/2510.07172v3). ICLR.
