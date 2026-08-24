"""
Planned Agent
=============
Phase 1 (no LLM):  Parse prompt → generate experiment plan.
Phase 2 (no LLM):  Execute planned experiments via the module.
Phase 3 (LLM):     Hand curated data to LLM for equation discovery.

Exports ``conduct_planned_exploration`` with the same signature and
return shape as ``utils.vanilla_agent.conduct_exploration``.
"""

import json
import re
from typing import List, Dict, Any

from utils.call_llm_api import call_llm_api
from .planner import ExperimentPlanner, sanitize_experiment


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert physicist tasked with discovering a scientific law "
    "from experimental data.\n\n"
    "You will receive a structured dataset collected via controlled-variable "
    "sweeps. Your job is to:\n"
    "1. Preprocess the raw data using the provided strategy (if any) to "
    "   recover the target quantity.\n"
    "2. For each input variable, examine the sweep where only that variable "
    "   changes. Determine the functional relationship (power law, "
    "   exponential, logarithmic, …).\n"
    "3. Combine dependencies into a single equation and estimate constants.\n\n"
    "Important: This universe's laws may differ from ours. Let data guide you.\n\n"
    "You may run additional experiments with <run_experiment> if needed.\n"
    "One action per round: <run_experiment> OR <final_law>.\n"
    "After <run_experiment>, wait for <experiment_output>."
)

DISCOVERY_TEMPLATE = """\
Here is the full mission description and apparatus:

{task_prompt}

---

The following experiments were already run for you using a systematic
controlled-variable strategy:

{data}

---

**Your task:**
1. Apply the preprocessing strategy above to convert raw outputs into
   the target quantity.
2. For each variable, look at its sweep (others held fixed) and determine
   the functional dependence.
3. Combine into one equation; estimate constants.
4. You may run up to {remaining} more verification experiments.
5. Submit with <final_law> when ready.

Begin your analysis now."""

DISCOVERY_TEMPLATE_COMPACT = """\
{condensed_prompt}

---

The following preprocessed data was collected via controlled-variable
sweeps.  The assisting equations have already been applied — each row
shows the physics inputs and the computed target quantity.

{data}

---

**Your task:**
1. For each input variable, examine the rows where only that variable
   changes and determine the functional dependence.
2. Combine into one equation and estimate constants.
3. You may run up to {remaining} more verification experiments.
4. Submit with <final_law> when ready.

Begin your analysis now."""


# ---------------------------------------------------------------------------
# Helpers (mirrors vanilla_agent logic)
# ---------------------------------------------------------------------------

def _parse_experiment_request(text: str) -> List[Dict[str, float]]:
    si = text.rfind("<run_experiment>")
    if si == -1:
        return []
    ei = text.find("</run_experiment>", si)
    if ei == -1:
        return []
    content = text[si + len("<run_experiment>"):ei].strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        return []
    except Exception:
        return []


def _extract_final_law(text: str, func_sig: str):
    si = text.rfind("<final_law>")
    if si == -1:
        return False, f"{func_sig} return float('nan')"
    ei = text.find("</final_law>", si)
    if ei == -1:
        return False, f"{func_sig} return float('nan')"
    body = text[si + len("<final_law>"):ei].strip()
    matches = re.findall(r'(def discovered_law.*?(?=\ndef|\Z))', body, re.DOTALL)
    if matches:
        return True, matches[-1].strip()
    return False, f"{func_sig} return float('nan')"


def _condense_prompt(task_prompt: str, module) -> str:
    """
    Extract just the essential parts of the task prompt:
    function signature, parameter description, and submission format.
    Drops the long apparatus/system description to save tokens.
    """
    lines = []
    lines.append("You are discovering a scientific law in a universe where physics may differ from ours.")
    lines.append("")

    # Include the function signature
    if hasattr(module, 'FUNCTION_SIGNATURE'):
        lines.append(f"Your discovered law must use this signature: {module.FUNCTION_SIGNATURE}")
    if hasattr(module, 'PARAM_DESCRIPTION'):
        lines.append(f"Parameter description: {module.PARAM_DESCRIPTION}")
    lines.append("")

    # Extract the <final_law> submission format from the prompt
    import re
    final_law_section = re.search(
        r'(<final_law>.*?</final_law>)',
        task_prompt,
        re.DOTALL,
    )
    if final_law_section:
        lines.append("Submit your answer using this format:")
        lines.append(final_law_section.group(1))
    else:
        lines.append("Submit using <final_law>def discovered_law(...): return ...</final_law>")

    # Extract run_experiment format if present
    run_exp_section = re.search(
        r'(<run_experiment>.*?</run_experiment>)',
        task_prompt,
        re.DOTALL,
    )
    if run_exp_section:
        lines.append("")
        lines.append("To run additional experiments use:")
        lines.append(run_exp_section.group(1))

    return "\n".join(lines)


def _append_assistant(messages, response_text, reasoning):
    combined = response_text or ""
    if reasoning and reasoning.strip():
        combined = f"**Reasoning Process:**\n{reasoning}\n\n**Main Response:**\n{combined}"
    messages.append({"role": "assistant", "content": combined})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def conduct_planned_exploration(
    module,
    model_name: str,
    noise_level: float,
    difficulty: str = "easy",
    system: str = "vanilla_equation",
    law_version: str = None,
    max_turns: int = 10,
    trial_info: Dict[str, Any] = None,
    planner_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Run the planned-agent loop.

    Returns the same dict shape as ``vanilla_agent.conduct_exploration``.
    """
    cfg = planner_config or {}
    planner = ExperimentPlanner(
        points_per_sweep=cfg.get("points_per_sweep", 10),
        noise_check_reps=cfg.get("noise_check_reps", 3),
    )

    # ── Phase 1: plan ────────────────────────────────────────────────
    task_prompt = module.get_task_prompt(system, noise_level=noise_level)
    plan = planner.plan(task_prompt, system_type_override=system)

    # ── Phase 2: execute ─────────────────────────────────────────────
    all_exps = [sanitize_experiment(e) for e in plan.noise_check + plan.sweeps]
    all_results: List[Any] = []
    n_experiments = 0

    for exp in all_exps:
        result = module.run_experiment_for_module(
            **exp,
            noise_level=noise_level,
            difficulty=difficulty,
            system=system,
            law_version=law_version,
        )
        all_results.append(result)
        n_experiments += 1

    # Noise detection
    noise_note = ""
    if plan.noise_check and len(plan.noise_check) >= 2:
        nr = all_results[: len(plan.noise_check)]
        if isinstance(nr[0], (int, float)):
            vals = [float(r) for r in nr]
            mx = max(abs(v) for v in vals) if vals else 0
            if mx > 0 and (max(vals) - min(vals)) / mx > 1e-10:
                noise_note = (
                    "\n\nNOTE: Noise detected in repeated measurements. "
                    "Consider averaging or accounting for uncertainty."
                )

    data_str = planner.format_results_for_llm(plan, all_exps, all_results, prompt=task_prompt)
    if noise_note:
        data_str += noise_note

    # ── Phase 3: LLM discovery ───────────────────────────────────────
    llm_turns = max(max_turns - 1, 3)

    # Use compact template for systems with preprocessed data
    lower_prompt = task_prompt.lower()
    is_preprocessed = (
        ("velocity" in lower_prompt)
        or ("spectral_radiance" in lower_prompt)
        or ("calorimeter" in lower_prompt)
    )

    if is_preprocessed:
        # Build a condensed prompt: just the function signature, param
        # description, and submission format — skip the long apparatus text
        condensed = _condense_prompt(task_prompt, module)
        user_content = DISCOVERY_TEMPLATE_COMPACT.format(
            condensed_prompt=condensed,
            data=data_str,
            remaining=llm_turns,
        )
    else:
        user_content = DISCOVERY_TEMPLATE.format(
            task_prompt=task_prompt,
            data=data_str,
            remaining=llm_turns,
        )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    total_tokens = 0
    total_rounds = 1  # planning = round 1

    for _ in range(llm_turns):
        response_text, reasoning, tokens = call_llm_api(
            messages, model_name=model_name, trial_info=trial_info,
        )
        total_tokens += tokens
        response_text = response_text or ""
        _append_assistant(messages, response_text, reasoning)
        total_rounds += 1

        ok, law = _extract_final_law(response_text, module.FUNCTION_SIGNATURE)
        if ok:
            return dict(
                status="completed", submitted_law=law,
                rounds=total_rounds, total_tokens=total_tokens,
                num_experiments=n_experiments, chat_history=messages,
            )

        extras = _parse_experiment_request(response_text)
        if extras:
            n_experiments += len(extras)
            results = []
            for exp in extras:
                r = module.run_experiment_for_module(
                    **exp, noise_level=noise_level, difficulty=difficulty,
                    system=system, law_version=law_version,
                )
                if system == "vanilla_equation":
                    r = "{:.15e}".format(r)
                results.append(r)
            messages.append({
                "role": "user",
                "content": f"<experiment_output>\n{json.dumps(results)}\n</experiment_output>",
            })
        else:
            messages.append({
                "role": "user",
                "content": (
                    "Please either run verification experiments with "
                    "<run_experiment> or submit your final law with <final_law>."
                ),
            })

    # Force submission
    force = "You have used all turns. Submit your final law now with <final_law>."
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"] += "\n\n" + force
    else:
        messages.append({"role": "user", "content": force})

    response_text, reasoning, tokens = call_llm_api(
        messages, model_name=model_name, trial_info=trial_info,
    )
    total_tokens += tokens
    _append_assistant(messages, response_text or "", reasoning)
    total_rounds += 1

    _, law = _extract_final_law(response_text or "", module.FUNCTION_SIGNATURE)
    return dict(
        status="max_turns_reached", submitted_law=law,
        rounds=total_rounds, total_tokens=total_tokens,
        num_experiments=n_experiments, chat_history=messages,
    )