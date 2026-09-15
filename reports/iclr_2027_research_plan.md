# Science on a Budget

## Research plan for an ICLR-level NewtonBench investigation

**Planning date:** 15 September 2026  
**Target:** ICLR 2027 main conference  
**Official deadlines:** abstract 18 September 2026, paper 25 September 2026, both 11:59 p.m. Anywhere on Earth.[^1]

## Executive recommendation

A credible ICLR submission is still possible, but the current budget curves alone are unlikely to be sufficient. NewtonBench itself is already an ICLR 2026 paper, and recent work already studies active, budget-constrained scientific discovery. A paper whose contribution is only “NewtonBench with dollar costs” will look incremental.

The strongest feasible paper is instead:

> **A controlled evaluation of resource-rational scientific agency, showing how LLMs allocate experimental evidence and inference compute under heterogeneous costs, testing whether they actually use feedback, and introducing a simple resource-aware discovery policy that improves the accuracy-cost frontier.**

The completed 3,455-trajectory pilot is a strong foundation. It already supplies the main phenomenon: nonlinear accuracy-resource curves, a low-budget cliff, model-specific substitution from experiments to tokens, an agent-interface interaction, and important protocol failures. The work becomes ICLR-level if the next experiments answer three reviewer questions:

1. **Is the effect actually caused by constrained evidence, or by a changed prompt and lower observation precision?**
2. **Are the agents adapting to experimental feedback, or merely spending according to priors and instructions?**
3. **How far are the LLM policies from principled or even simple experimental-design baselines?**

The minimum viable submission should therefore prioritize causal controls, a feedback-sensitivity test, paired uncertainty, and at least one normative baseline. Adding many more models is less valuable than answering those questions.

## 1. Proposed paper identity

### Working title

**Science on a Budget: Measuring Resource-Rational Experimental Discovery in Language Agents**

Alternatives:

- **Beyond Test-Time Compute: Budgeting Experimental Evidence for Scientific Agents**
- **What Is an Experiment Worth? Resource-Rational Evaluation of LLM Scientific Discovery**
- **GrantBench: Heterogeneous-Cost Experimental Discovery with Language Agents**

The first is the safest. `GrantBench` should be used only if the environment and evaluation protocol become a reusable artifact beyond a small NewtonBench extension.

### One-sentence thesis

Current scientific agents are not uniformly “good” or “bad” at discovery: their apparent capability depends strongly on how they allocate heterogeneous experimental resources, and stronger models often replace missing evidence with more internal computation without recovering the lost mechanistic accuracy.

### Contribution package

The target paper should make four contributions:

1. **Environment:** a reproducible heterogeneous-cost extension of interactive law discovery, with hard grants, setup costs, nonlinear sample prices, selectable fidelity, range costs, and explicit constraint handling.
2. **Evaluation:** paired accuracy-resource curves and resource-rational metrics that keep experimental cost, inference cost, scientific correctness, and operational reliability separate.
3. **Diagnosis:** causal controls for precision, budget framing, adaptivity, and feedback use; trajectory analyses of stopping, invalid actions, and inference/experiment substitution.
4. **Method or baseline:** a lightweight hypothesis-and-value-of-information scaffold that produces a better Pareto frontier than unstructured LLM agents and basic fixed designs.

Without item 3, the result is correlational. Without item 4, the work can still be a useful benchmark paper, but the bar for breadth, robustness, and evaluation completeness becomes much higher.

## 2. Novelty boundary

### What NewtonBench already contributes

NewtonBench already provides interactive, counterfactual-law discovery across 12 physics domains and shows fragility under complexity, noise, and tool use.[^2] The present paper must not resell interactivity, memorization resistance, or the basic code-versus-vanilla comparison as new.

### Closest neighboring work

| Work | What it already covers | What this project must add |
|---|---|---|
| NewtonBench[^2] | Interactive recovery of shifted physical laws | Heterogeneous resource allocation and its causal evaluation |
| LLM-AutoSciLab[^3] | Hypothesis-conditioned active acquisition; budget-constrained ActiveSciBench; sample-efficiency gains | Monetary/fidelity/setup trade-offs, joint token-evidence allocation, and resource-rational diagnostics |
| DiscoveryWorld[^4] | Complete simulated discovery cycles | Controlled quantitative law identifiability and cost frontiers |
| DiscoverPhysics[^5] | Interactive altered-physics discovery in N-body worlds | Resource intervention, allocation policy, and causal budget controls |
| ScienceAgentBench[^6] | Realistic scientific code tasks, execution, results, and costs | Adaptive acquisition rather than analysis of a provided task/dataset |
| Budget-aware agent scaling[^7] | Joint token/tool budgets and budget-aware planning, mainly in search agents | Scientific information quality, measurement fidelity, and mechanistic recovery |
| Bayesian experimental design[^8] | Normative information-gain acquisition under costs | Evaluation of general LLM policies without assuming a correct probabilistic model |

The novelty is not “budget awareness.” Budget-aware web and tool agents already exist. The novelty is the **scientific semantics of the resource**: experiments differ in their ability to discriminate mechanisms, precision is purchasable, extreme regimes cost more, and success is an executable recovered law.

## 3. Research questions and preregistered hypotheses

### RQ1: How does mechanistic recovery scale with experimental resources?

**H1.1:** Symbolic recovery is nonlinear in log grant size and contains module-dependent minimum-evidence thresholds.  
**H1.2:** Stronger models dominate at moderate budgets, but model gaps compress under extreme scarcity.  
**Current evidence:** strongly supportive, but confidence intervals and replicated seeds are missing.

### RQ2: Do agents allocate heterogeneous resources rationally?

**H2.1:** Better agents achieve higher accuracy at matched assigned grants, not merely by spending more.  
**H2.2:** Their experiment choices eliminate more candidate laws per unit cost than random, space-filling, or one-factor-at-a-time policies.  
**H2.3:** They reserve enough grant for a discriminating confirmation experiment rather than exhausting funds on broad initial sweeps.  
**Current evidence:** H2.1 is suggestive; H2.2 and H2.3 are untested.

### RQ3: Do agents use observations adaptively?

**H3.1:** Permuting or corrupting experimental feedback reduces accuracy and changes subsequent actions.  
**H3.2:** Closed-loop policies outperform replayed open-loop action schedules at matched cost.  
**Current evidence:** untested. This is a critical gap because recent scientific-agent work found little or no sensitivity to experimental feedback and stronger classical baselines.[^9]

### RQ4: How do internal and external resources interact?

**H4.1:** Lower experimental grants increase inference-token use for reasoning models.  
**H4.2:** Extra tokens have diminishing value when the observed data do not identify the law.  
**H4.3:** A policy that explicitly separates hypothesis generation, acquisition, fitting, and verification converts both resources more efficiently.  
**Current evidence:** H4.1 is supported descriptively for Qwen38; H4.2 and H4.3 need matched controls.

### RQ5: Which failures are scientific and which are operational?

**H5.1:** Invalid actions and missing final laws account for a meaningful, model-dependent fraction of failures.  
**H5.2:** After controlling for task identity and difficulty, genuine model protocol failures still predict lower recovery.
**Current evidence:** an initial upper-bound audit finds reasoning-only action blocks in 29.2–54.9% of trials per model-condition, and exact examples confirm genuine rejected actions. A conservative count requiring no valid main action and exactly one valid reasoning action is pending. This is primarily a harness-validity issue; genuine malformed actions must be measured separately after repair.

## 4. Mandatory workstream A: freeze reliable trajectories and outcomes

This must finish before any new headline analysis.

### A0. Repair action-channel handling and rerun the pilot

The current trajectories are not yet publication-ready. The vLLM endpoint separates `reasoning` and `content`, while the agents generally execute only actions found in `content`. The initial 29.2–54.9% range is an upper bound; exact examples nevertheless confirm that valid experiment and Python actions can be lost, changing later evidence in a way rejudging cannot repair. Complete the conservative channel audit first, then determine the replacement-run scope.

Implement and test one explicit policy: if main content contains no action and reasoning contains exactly one complete, valid allowed action, execute that action; otherwise preserve the existing rejection behavior. Store raw reasoning and content separately and record which field supplied each executed action. Add adversarial tests for all three action types, malformed blocks, and conflicting/multiple actions. Then rerun the complete model × resource matrix, including the unbudgeted baseline. Preserve the current trees as a named flawed-pilot artifact rather than overwriting them.

### A1. Establish two named recovery metrics

Report both:

- **Exact Structural Recovery (ESR):** exact dependence on the scientific variables up to nuisance multiplicative constants and documented algebraic/domain equivalences.
- **Tolerance-Aware Scientific Recovery (TSR):** ESR plus recovered special constants or exponents within a preregistered numerical tolerance, provided the functional structure is otherwise correct.

Predictive RMSLE remains a continuous secondary metric. It must not silently convert interpolation into symbolic recovery.

### A2. Audit the 125 judge/checker disagreements

There are 33 Muse and 92 Qwen38 disagreement candidates across the 12 model-resource conditions. Audit every row, blind to model and budget. Record:

- exact equivalent;
- tolerance-equivalent;
- predictively close but structurally wrong;
- clearly wrong; or
- invalid/uninterpretable.

A second annotator should independently review at least all tolerance-equivalent and ambiguous rows. Report agreement and resolve conflicts before unblinding condition labels.

### A3. Validate the independent-judge fallback

The deterministic checker returned `not_checkable` for 936 of 3,455 trajectories, so Gemma supplies many final labels. Manually audit a stratified sample across model, budget, module, judge label, and output complexity. A practical minimum is 120 rows plus all surprising positives from invalid-looking code. Use the sample to estimate false-positive and false-negative rates with intervals.

### A4. Freeze judge provenance

Archive:

- exact Hugging Face repository and commit/revision;
- quantization and dtype;
- vLLM version and complete serve command;
- judge prompt hash;
- temperature (`0` in the current code);
- maximum generation setting or confirmation that none was imposed;
- parser and retry policy;
- source and output tree hashes; and
- NewtonBench git commit.

The two-A100 limit is compatible with this protocol. No publication label should use the evaluated model as its own judge.

### A5. Improve deterministic coverage only if low-risk

High-value extensions are AST normalization, safe evaluation on randomized domain-valid points, analytic fitting of one nuisance scale, and explicit exponent tolerance. Do not attempt an ambitious theorem prover during the deadline window. Every checker rule needs adversarial unit tests that include near-fit structural errors.

**Exit criterion:** the action-channel tests pass; replacement trajectories contain no unhandled single valid actions; ESR and TSR labels are frozen; a scoring rerun produces identical scoreboards; no unresolved publication labels remain; the audit sheet and configuration hashes are archived.

## 5. Mandatory workstream B: isolate the intervention

The five hard-cap conditions are internally comparable because they share the same cost schedule and precision options. The unbudgeted endpoint is not a clean infinite-budget control: budget mode also introduces one-significant-figure default observations, cost instructions, balance messages, and rejection behavior.

Run the following factorial controls on a diagnostic subset first. Select six modules spanning easy and hard, simple and complex, and distinct mathematical structures—for example Snell, Fourier, gravity, BE distribution, magnetic force, and Hooke.

| Control | Constraint | Observation fidelity | Budget text | Purpose |
|---|---|---|---|---|
| C0 original | No money | Original | No | Existing unbudgeted reference |
| C1 rounded-only | No money | p1 rounding | No | Isolate fidelity loss |
| C2 sham budget | Requests never rejected | Same as budget mode | Yes | Isolate framing/accounting effects |
| C3 hard grant | $200/$400/$1,000 | Same as budget mode | Yes | Existing treatment |
| C4 fixed-count | Match treatment's observation count | Fixed p1 or p3 | Neutral | Separate quantity from adaptive allocation |
| C5 high-fidelity grant | Same grant | Fixed p3/p5 | Yes | Test quantity-fidelity substitution |

The most important contrasts are C0–C1, C1–C2, C2–C3, and C3–C4. If time allows only two new controls, run **rounded-only** and **sham budget**. They directly answer the most damaging confound.

Do not describe `budget_1000_uncap.json` as a causal uncap control merely because of its filename. Under the current code, hard rejection is a runtime policy, not a property disabled by that JSON. Verify the actual execution semantics before using any historical soft-cap tree.

### Cost-schedule robustness

The present convex pricing strongly rewards batches of at most five. A limited robustness grid should alter one feature at a time:

- flat versus convex batch price;
- setup cost on versus off;
- precision cost shallow versus steep; and
- range surcharge on versus off.

This can be run on the six-module subset at $400. The goal is not an exhaustive economics study; it is to show that the core model ranking and policy conclusions are not artifacts of one hand-picked tariff.

## 6. Mandatory workstream C: test whether feedback matters

This is likely the highest-value new analysis.

### C1. Feedback permutation

Within each module and legal parameter schema, permute experiment outputs across law versions or trials while preserving output scale as much as possible. The agent sees plausible but wrong feedback. Compare final recovery and subsequent action choices with the correct-feedback condition.

Interpretation:

- large performance drop and action divergence: evidence of feedback-conditioned discovery;
- unchanged actions but lower score: feedback may influence fitting but not acquisition;
- unchanged actions and score: evidence that performance is driven mainly by prior/template behavior.

### C2. Open-loop replay

Take an action sequence generated in one seed and replay it without allowing later actions to depend on observations. Give the resulting dataset to the same final-law inference stage. Match total cost and inference allowance.

The **adaptivity gain** is:

\[
\Delta_{\text{adaptive}} = \mathrm{SA}_{\text{closed loop}} - \mathrm{SA}_{\text{replayed open loop}}.
\]

This is cleaner than merely counting hypothesis revisions.

### C3. Observation ablation

Run final inference with:

- all acquired observations;
- observations shuffled among requests;
- outputs removed but action history retained; and
- only the first request retained.

Use a small stratified subset if compute is tight. The purpose is to locate where information enters the trajectory.

### C4. Retrospective action informativeness

Because the simulator and candidate law versions are known to the evaluator, score each purchased action by how strongly the candidate laws disagree at that design point. Useful metrics include:

- number of candidate laws eliminated after the observation;
- reduction in candidate-set entropy;
- minimum pairwise standardized prediction separation;
- information proxy divided by monetary cost; and
- fraction of spending devoted to redundant versus discriminating points.

This converts qualitative trajectory claims into a measurable acquisition-policy comparison.

## 7. Mandatory workstream D: add baselines and one intervention

### D1. Fixed-design baselines

All baselines must receive the same grant, price schedule, parameter domain, and output fidelity.

1. **Random feasible:** sample legal designs until the grant cannot fund another request.
2. **Space-filling:** Latin hypercube or log-uniform coverage over legal ranges.
3. **One-factor-at-a-time:** controlled logarithmic sweeps around a reference point.
4. **Cheap-first:** maximize number of p1 observations in the lowest batch tier.
5. **High-fidelity-first:** purchase fewer p3/p5 observations.

Feed each acquired dataset to the same downstream law-inference backend so the comparison isolates acquisition.

### D2. Symbolic-regression baseline

Run a standard symbolic-regression system on the observations from each fixed design. AI Feynman established that physics-informed structure can solve many fixed-table equation-recovery tasks,[^10] while newer neural and LLM-guided symbolic regression methods expand that space. The key comparison is not whether symbolic regression beats an LLM on familiar equations; it is how acquisition and inference interact under identical evidence.

If integrating PySR or AI Feynman is too risky before the deadline, implement a transparent grammar-based search over the operators used by the selected modules and clearly label it as a restricted symbolic baseline.

### D3. Candidate-disagreement oracle

Give a planner the finite candidate set of NewtonBench law variants but not the active version. At each step, choose the affordable experiment that maximizes candidate prediction disagreement per dollar, update the candidate set, and stop when one remains or the grant is exhausted.

This is not a fair general-purpose competitor because it knows the hypothesis family. It is an **information-efficiency upper bound**. That makes it extremely useful: it separates “the grant is intrinsically insufficient” from “the agent spent it poorly.”

### D4. Proposed method: Budgeted Hypothesis–Experiment Loop

Add a lightweight scaffold around the same base LLM:

1. maintain 2–5 explicit candidate mechanisms;
2. state what observation would discriminate the top pair;
3. propose several legal experiments with expected outcomes under each candidate;
4. estimate discriminatory value and cost;
5. purchase the best value-per-cost action;
6. update candidates after the observation;
7. reserve a configurable fraction, e.g. 15%, for final verification; and
8. pass the final expression through a local execution and format validator before submission.

Call it **BHEL** only if an acronym is useful; descriptive naming is safer in a rushed draft. The method should not require training and should run with both current evaluated models.

The main method claim should be Pareto-oriented: at matched grants, the scaffold improves ESR/TSR or matches accuracy with less realized cost and fewer invalid actions. Do not optimize a weighted reward whose coefficients can manufacture a win.

## 8. Experimental design and statistics

### Unit of analysis

The 288 rows in a condition are not 288 fully independent tasks. Trials share module, law version, agent backend, difficulty, and system. The primary resampling unit should be the **task configuration**, with stochastic trials nested inside it.

### Pairing

Use common random seeds and the same manifest across grants and methods. Pairing dramatically improves contrasts and prevents rerun-directory selection from changing the task set.

### Recommended estimates

- cluster bootstrap confidence intervals over module/law/agent configurations;
- a mixed-effects logistic model with fixed effects for log grant, model, agent, and key interactions, plus random intercepts for module and law version;
- paired grant contrasts, especially $400 versus $200 and $1,000 versus $600;
- sensitivity analyses under ESR and TSR;
- bootstrap intervals for area under the accuracy-versus-log-budget curve; and
- no uncorrected fishing across every module-budget-agent cell.

### Primary metrics

1. ESR and TSR at each assigned grant.
2. Area under the accuracy versus log-grant curve, normalized to the measured interval.
3. Minimum grant achieving 50% and 80% of the model's high-resource performance.
4. Pareto frontier over accuracy, assigned grant, realized experimental spend, and inference tokens.
5. Adaptivity gain from closed-loop versus replayed open-loop acquisition.
6. Feedback-sensitivity gap under correct versus permuted observations.
7. Candidate elimination or disagreement reduction per dollar.

### Secondary metrics

- valid final-law rate;
- format failures and wasted turns;
- unaffordable/rejected requests;
- unused grant;
- number, size, precision, and parameter range of requests;
- stopping point and verification reserve;
- RMSLE conditional on structural failure; and
- operational runner failures, reported separately from scientific failures.

Avoid a single “efficiency score” unless every component is also reported. A scalar can hide an agent that achieves cheapness by abstaining or failing.

## 9. Model matrix under the hardware constraint

The current two-model comparison is enough for a focused pilot but weak for a broad benchmark claim. The ideal matrix is:

- current strong reasoning model: Qwen38-27B;
- current contrasting model: Muse-Glimmer-30B;
- one smaller open model from the same or adjacent family to test scaling; and
- one different-family model that still fits on at most two A100-80GB GPUs.

Do not test `gemma4-31b` as an evaluated model unless a different independent judge is served for its scoring. Given the deadline, finish controls and baselines for the existing two before launching a full five-budget curve for another model. A third model on the six-module diagnostic subset is more useful than an incomplete full matrix.

## 10. Minimum viable run matrix

To control scope, separate **full benchmark evidence** from **diagnostic subset evidence**.

### Full representative subset

- existing two models × two agents × six resource conditions;
- BHEL at $200, $400, and $1,000 for both models;
- one strong fixed-design baseline and the candidate-disagreement oracle at the same three grants; and
- at least one additional paired stochastic repeat for the pivotal $200 and $400 points.

### Six-module diagnostic subset

- rounded-only and sham-budget controls;
- feedback permutation;
- open-loop replay;
- p1 versus p3/p5 fidelity;
- flat versus convex batch costs; and
- optional third model.

This design is much more defensible than spreading compute across many models without causal controls.

## 11. Figures and tables the paper should contain

### Main figures

1. **Accuracy-resource curves:** ESR and TSR versus log grant, with clustered 95% intervals, faceted by model and agent.
2. **Joint-resource frontier:** accuracy versus experimental spend, with point size or color representing inference tokens.
3. **Causal decomposition:** original, rounded-only, sham-budget, hard-budget, and fixed-count controls.
4. **Feedback and adaptivity:** correct versus permuted feedback and closed-loop versus replayed open-loop.
5. **Policy quality:** candidate disagreement eliminated per dollar for LLM, BHEL, fixed baselines, and oracle.

### Main tables

1. task and cost-model specification;
2. aggregate ESR/TSR with confidence intervals;
3. baseline and method comparisons at $200/$400/$1,000;
4. operational reliability and valid-submission rates; and
5. scoring audit agreement.

### Appendix

- all module × model × agent × grant results;
- prompts and configuration JSON;
- exact judge/server metadata;
- cost-schedule sensitivity;
- trajectory examples selected by a predeclared rule;
- scorer unit tests and adjudication guidelines; and
- result-tree checksums and command manifests.

## 12. Ten-day execution plan

The official ICLR calendar is tighter than “two weeks.” A genuine abstract must be registered by 18 September and the full paper is due 25 September.[^1]

### 15 September

- Repair and test reasoning/content action handling.
- Launch replacement unbudgeted and hard-grant pilot runs; do not overwrite the flawed pilot.
- Freeze the thesis, research questions, and primary metrics.
- Generate paired manifests and cluster-bootstrap code.
- Register paper title, authors, and a genuine abstract draft internally.

### 16 September

- Complete the replacement pilot and rerun rejudge, diagnostics, and scoreboards.
- Regenerate the disagreement set; do not assume the current 125 cases persist.
- Complete ESR/TSR scoring and judge-validation sample.
- Implement rounded-only and sham-budget controls.
- Implement random/space-filling and candidate-disagreement baselines.
- Unit-test the final-law validator and cost accounting.

### 17 September

- Launch the six-module causal controls and feedback-permutation runs.
- Launch baseline runs at $200/$400/$1,000.
- Draft methods and related work while jobs run.

### 18 September — abstract deadline

- Submit a genuine abstract before the official deadline.
- Lock the main model and ablation matrix; stop adding speculative experiments.
- Inspect early results for broken controls or ceiling/floor effects.

### 19–20 September

- Finish causal controls, open-loop replay, and the first BHEL comparison.
- Run one paired replication of $200 and $400 for the existing models.
- Make a go/no-go decision on the main-conference framing.

### 21 September

- Freeze data.
- Produce all main figures and clustered intervals.
- Write results and limitations from the frozen tables only.

### 22 September

- Complete the first full draft.
- Run an internal adversarial review focused on novelty, confounds, scoring, and causal language.

### 23 September

- Revise the argument and remove unsupported claims.
- Complete appendix, prompt disclosure, and reproducibility checklist.

### 24 September

- Final independent score audit and figure/table cross-check.
- Anonymity, citation, and author-policy checks.
- Upload a near-final draft early enough to catch formatting problems.

### 25 September — paper deadline

- Incorporate only critical corrections.
- Submit before 11:59 p.m. AoE; do not use the final hours for new experiments.

## 13. Go/no-go criteria

Proceed with the ICLR main-paper claim if, by 20 September:

- the ESR/TSR audit is stable and does not qualitatively reverse the curves;
- at least one clean control isolates a true resource constraint effect beyond p1 rounding and prompt framing;
- feedback corruption or open-loop replay shows a measurable, interpretable effect—or the absence of such an effect becomes a well-powered negative result;
- at least one non-LLM baseline and the oracle upper bound are complete; and
- either BHEL improves the frontier or the diagnostic evaluation itself is unusually comprehensive and robust.

Narrow the paper to a diagnostic benchmark if BHEL does not help but controls and baselines are strong. Defer to ICML or a later venue if scoring remains unstable, the entire effect is explained by observation rounding, or no evidence shows that trajectories use experimental feedback.

The ICLR call explicitly asks authors to submit work they are confident is correct and sufficiently complete.[^1] A rushed weak submission is not automatically useful feedback: public reviews of an incremental or confounded extension may say little beyond issues already identifiable now.

## 14. Likely reviewer objections and required answers

### “The dollar scale is arbitrary.”

Agree that it is synthetic. Emphasize dimensionless grant-to-cost ratios, show robustness to rescaled or altered tariffs, and avoid real-laboratory interpretations.

### “This is just fewer samples.”

Use fixed-count controls, precision ablations, and action-informativeness metrics. Show whether equal counts with different design policies produce different recovery.

### “The model does not use feedback.”

Answer with permutation, observation ablation, and replayed open-loop experiments. This objection cannot be answered from ordinary trajectories alone.

### “Two models are not enough.”

Frame the work as a mechanistic case study with paired causal interventions, or add a third model on the diagnostic subset. Do not claim universal scaling laws.

### “The judge determines the result.”

Report ESR, TSR, predictive RMSLE, deterministic coverage, fallback-judge audit, and sensitivity to adjudication. Release per-trial labels.

### “The baseline knows the answer family.”

Label the candidate-disagreement planner as an oracle upper bound, not a fair competitor. Include family-agnostic random, space-filling, and symbolic-regression baselines.

### “Longer reasoning causes failure only because hard tasks are longer.”

Use paired within-task contrasts or multilevel models. Describe raw token correlations as descriptive, never causal.

### “The method is just prompt engineering.”

The scaffold is valuable only if its components are explicit and ablated: hypothesis set, costed acquisition, reserve policy, and final validator. Compare against an equally long generic planning prompt.

## 15. Paper outline

1. **Introduction:** unconstrained scientific accuracy misses the evidence-allocation problem; summarize the cliff and joint-resource substitution.
2. **Related work:** NewtonBench and altered-physics discovery; scientific agents; symbolic regression; experimental design; budget-aware agent scaling.
3. **Resource-constrained NewtonBench:** action costs, fidelity, hard constraints, task subset, and two-resource formulation.
4. **Evaluation protocol:** ESR/TSR, independent judge, deterministic cascade, audit, pairing, and statistics.
5. **How current agents spend a grant:** primary curves, module thresholds, agent interactions, tokens versus experiments, reliability.
6. **Do agents perform adaptive experimental design?:** causal controls, feedback permutation, replay, and action informativeness.
7. **Baselines and BHEL:** fixed designs, symbolic inference, oracle, method, and ablations.
8. **Discussion:** resource rationality, implications for autonomous laboratories, limitations of synthetic economics.
9. **Reproducibility and ethics:** local models, compute, judge dependence, provenance, and absence of real-world safety claims.

## 16. Success standard

The ideal paper should let a skeptical reader answer four questions from one set of figures:

1. How much mechanistic accuracy is lost as evidence becomes expensive?
2. Is the loss caused by evidence constraints rather than incidental prompt or fidelity changes?
3. Do the agents actually adapt their experiments to observations?
4. Can a simple principled policy use both experimental and computational resources better?

The completed pilot answers the first question well and exposes compelling behavior around the fourth. The next ten days should be spent answering the middle two and building one credible policy comparison—not accumulating more unstructured trajectories.

## Sources

The five most important sources for the research plan are:

1. [ICLR 2027 Call for Papers](https://www.iclr.cc/Conferences/2027/CallForPapers) — official deadlines and submission expectations.
2. [NewtonBench](https://arxiv.org/abs/2510.07172) — the parent benchmark and the baseline novelty boundary.
3. [LLM-AutoSciLab](https://arxiv.org/abs/2605.24043) — the closest active, budget-constrained scientific-discovery work.
4. [LLMs for Experiment Design in Scientific Domains: Are We There Yet?](https://openreview.net/forum?id=dIEeOwrmOe) — motivation for feedback corruption and classical baselines.
5. [A Unified Stochastic Gradient Approach to Designing Bayesian-Optimal Experiments](https://proceedings.mlr.press/v108/foster20a.html) — a representative normative reference for information-gain-based experiment design.

[^1]: [ICLR 2027 Call for Papers](https://www.iclr.cc/Conferences/2027/CallForPapers) and [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).
[^2]: T. Zheng et al., [“NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents,”](https://arxiv.org/abs/2510.07172) ICLR 2026.
[^3]: S. Kabra et al., [“LLM-AutoSciLab: Closed-Loop Scientific Discovery via Active Experimentation with LLMs,”](https://arxiv.org/abs/2605.24043) 2026.
[^4]: P. Jansen et al., [“DiscoveryWorld,”](https://arxiv.org/abs/2406.06769) NeurIPS 2024 Datasets and Benchmarks Track.
[^5]: M. L. Wiemann et al., [“DiscoverPhysics: Benchmarking LLMs for Out-of-the-Box Scientific Thinking,”](https://arxiv.org/abs/2605.26087) 2026.
[^6]: Z. Chen et al., [“ScienceAgentBench,”](https://arxiv.org/abs/2410.05080) 2024.
[^7]: [“Cost-Effective Agent Test-Time Scaling via Budget-Aware Thinking,”](https://openreview.net/forum?id=AaMB3SFmBy) ICLR 2026 submission; K. Zhu et al., [“Scaling Test-time Compute for LLM Agents,”](https://arxiv.org/abs/2506.12928) 2025.
[^8]: A. Foster et al., [“A Unified Stochastic Gradient Approach to Designing Bayesian-Optimal Experiments,”](https://proceedings.mlr.press/v108/foster20a.html) AISTATS 2020; E. H. Lee et al., [“A Nonmyopic Approach to Cost-Constrained Bayesian Optimization,”](https://proceedings.mlr.press/v161/lee21a.html) UAI 2021.
[^9]: R. Gupta, J. Hartford, and B. Liu, [“LLMs for Experiment Design in Scientific Domains: Are We There Yet?”](https://openreview.net/forum?id=dIEeOwrmOe) ICML 2025 GenBio Workshop.
[^10]: S.-M. Udrescu and M. Tegmark, [“AI Feynman: A Physics-Inspired Method for Symbolic Regression,”](https://arxiv.org/abs/1905.11481) *Science Advances*, 2020.
