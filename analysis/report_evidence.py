#!/usr/bin/env python3
"""Extract report-ready evidence from completed NewtonBench trace CSVs.

This utility is intentionally read-only.  It follows the exact trial paths in
``analysis/trajectory_trace_*.csv`` so that subset filtering and de-duplication
match the completed diagnostics run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_GLOB = "analysis/trajectory_trace_*judged_by_gemma4-31b*.csv"
FORMAT_MARKERS = (
    "Invalid response",
    "Action Reminder",
    "exactly 1 action per turn",
)
RUN_EXPERIMENT_RE = re.compile(
    r"<run_experiment>\s*(.*?)\s*</run_experiment>", re.DOTALL
)
ACTION_BLOCKS = {
    "experiment": re.compile(r"<run_experiment>.*?</run_experiment>", re.DOTALL),
    "python": re.compile(r"<python>.*?</python>", re.DOTALL),
    "final_law": re.compile(r"<final_law>.*?</final_law>", re.DOTALL),
}
MAIN_RESPONSE_BOUNDARY = "\n\n**Main Response:**\n"
BUDGET_DIR_RE = re.compile(r"budget_evaluation_results_(\d+)_cap")


def parse_bool(value):
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def clip(text, limit=1800, *, keep_tail=False):
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    if keep_tail:
        return "...[truncated]...\n" + text[-limit:]
    return text[:limit] + "\n...[truncated]..."


def condition_from_path(path):
    match = BUDGET_DIR_RE.search(str(path))
    return f"${match.group(1)}" if match else "unbudgeted"


def condition_key(condition):
    if condition == "unbudgeted":
        return (0, 0)
    return (1, -int(condition.removeprefix("$")))


def load_records(pattern):
    trace_files = sorted(Path().glob(pattern))
    if not trace_files:
        raise SystemExit(f"No trace CSVs matched {pattern!r}; run diagnostics.py trace first.")

    records = []
    seen = set()
    for trace_path in trace_files:
        with trace_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw_path = row.get("path", "")
                if not raw_path or raw_path in seen:
                    continue
                seen.add(raw_path)
                trial_path = Path(raw_path)
                if not trial_path.exists():
                    print(f"WARNING: missing trial JSON: {trial_path}")
                    continue
                try:
                    data = json.loads(trial_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    print(f"WARNING: could not read {trial_path}: {exc}")
                    continue
                model = (
                    data.get("evaluated_model")
                    or data.get("model_name")
                    or trial_path.parts[-7]
                )
                records.append(
                    {
                        "path": str(trial_path),
                        "condition": condition_from_path(trial_path),
                        "model": str(model),
                        "difficulty": row.get("equation_difficulty")
                        or data.get("equation_difficulty")
                        or "unknown",
                        "success": parse_bool(row.get("verified_success")),
                        "no_parseable_law": parse_bool(row.get("no_parseable_law")),
                        "data": data,
                    }
                )
    if not records:
        raise SystemExit("Trace CSVs were found, but no referenced trial JSON could be read.")
    return records


def preceding_assistant(history, user_index):
    for index in range(user_index - 1, -1, -1):
        if history[index].get("role") == "assistant":
            return history[index].get("content", "") or ""
    return ""


def split_saved_channels(content):
    """Recover the harness-combined reasoning/content fields when present.

    The harness prefixes separated reasoning with ``**Reasoning Process:**``
    and inserts the final ``**Main Response:**`` boundary itself.  ``rsplit``
    is deliberate because some models repeat the same heading in reasoning.
    """
    content = content or ""
    if content.startswith("**Reasoning Process:**") and MAIN_RESPONSE_BOUNDARY in content:
        reasoning, main = content.rsplit(MAIN_RESPONSE_BOUNDARY, 1)
        return reasoning.removeprefix("**Reasoning Process:**\n"), main
    return "", content


def command_format_examples(records):
    counts = defaultdict(lambda: {"trials": set(), "events": Counter()})
    channel_counts = defaultdict(Counter)
    examples = {}
    no_law_examples = {}

    for record in records:
        key = (record["model"], record["condition"])
        counts[key]["trials"].add(record["path"])
        history = record["data"].get("chat_history", []) or []
        for message in history:
            if message.get("role") != "assistant":
                continue
            reasoning, main = split_saved_channels(message.get("content", "") or "")
            if not reasoning:
                continue
            stranded_any = False
            for action, pattern in ACTION_BLOCKS.items():
                in_reasoning = len(pattern.findall(reasoning))
                in_main = len(pattern.findall(main))
                if in_reasoning and not in_main:
                    channel_counts[key][action] += in_reasoning
                    stranded_any = True
            if stranded_any:
                channel_counts[key]["turns"] += 1
                channel_counts[key]["trial::" + record["path"]] = 1

        for index, message in enumerate(history):
            if message.get("role") != "user":
                continue
            feedback = message.get("content", "") or ""
            found = [marker for marker in FORMAT_MARKERS if marker in feedback]
            if not found:
                continue
            counts[key]["events"]["any"] += 1
            counts[key]["events"]["affected::" + record["path"]] = 1
            for marker in found:
                counts[key]["events"][marker] += 1
                example_key = (record["model"], marker)
                examples.setdefault(
                    example_key,
                    (
                        record["condition"],
                        record["path"],
                        preceding_assistant(history, index),
                        feedback,
                    ),
                )

        if record["no_parseable_law"] and record["model"] not in no_law_examples:
            assistants = [
                message.get("content", "") or ""
                for message in history
                if message.get("role") == "assistant"
            ]
            no_law_examples[record["model"]] = (
                record["condition"],
                record["path"],
                record["data"].get("submitted_law"),
                assistants[-1] if assistants else "",
            )

    print("# Format-feedback counts")
    print("model,condition,trials,affected_trials,feedback_events,invalid_response,action_reminder,exactly_one_action")
    for key in sorted(counts, key=lambda item: (item[0], condition_key(item[1]))):
        model, condition = key
        entry = counts[key]
        events = entry["events"]
        affected = sum(name.startswith("affected::") for name in events)
        print(
            f"{model},{condition},{len(entry['trials'])},{affected},{events['any']},"
            f"{events['Invalid response']},{events['Action Reminder']},"
            f"{events['exactly 1 action per turn']}"
        )

    print("\n# Complete action blocks stranded in separated reasoning")
    print("model,condition,affected_trials,assistant_turns,experiment_blocks,python_blocks,final_law_blocks")
    for key in sorted(counts, key=lambda item: (item[0], condition_key(item[1]))):
        model, condition = key
        entry = channel_counts[key]
        affected = sum(name.startswith("trial::") for name in entry)
        print(
            f"{model},{condition},{affected},{entry['turns']},{entry['experiment']},"
            f"{entry['python']},{entry['final_law']}"
        )

    print("\n# Exact format-failure examples")
    print("One example per model and detector marker; text is truncated, not paraphrased.")
    for (model, marker), (condition, path, assistant, feedback) in sorted(examples.items()):
        print(f"\n## {model} | {condition} | marker={marker!r}\nPATH: {path}")
        print("\nASSISTANT TURN:\n" + clip(assistant, keep_tail=True))
        print("\nREJECTION FEEDBACK:\n" + clip(feedback))

    print("\n# No-parseable-law examples")
    for model, (condition, path, submitted, last_assistant) in sorted(no_law_examples.items()):
        print(f"\n## {model} | {condition}\nPATH: {path}")
        print("\nSTORED submitted_law:\n" + clip(submitted))
        print("\nLAST ASSISTANT TURN:\n" + clip(last_assistant, keep_tail=True))


def requested_precisions(history):
    levels = Counter()
    invalid_actions = 0
    for message in history:
        if message.get("role") != "assistant":
            continue
        content = message.get("content", "") or ""
        for match in RUN_EXPERIMENT_RE.finditer(content):
            try:
                experiments = json.loads(match.group(1))
                if not isinstance(experiments, list):
                    raise ValueError("experiment action is not a list")
                for experiment in experiments:
                    if isinstance(experiment, dict):
                        raw = experiment.get("precision", 1)
                        try:
                            level = int(round(float(raw)))
                        except (TypeError, ValueError):
                            level = "invalid"
                        levels[level] += 1
            except (json.JSONDecodeError, ValueError):
                invalid_actions += 1
    return levels, invalid_actions


def command_budget_details(records):
    groups = defaultdict(
        lambda: {
            "n": 0,
            "successes": 0,
            "success_n": 0,
            "spent": 0.0,
            "rejections": 0,
            "trials_rejected": 0,
        }
    )
    precision_groups = defaultdict(Counter)

    for record in records:
        budget = record["data"].get("budget") or {}
        if not budget or not budget.get("enabled"):
            continue
        key = (record["model"], record["condition"], record["difficulty"])
        group = groups[key]
        group["n"] += 1
        if record["success"] is not None:
            group["success_n"] += 1
            group["successes"] += int(record["success"])
        group["spent"] += float(budget.get("total_spent") or 0.0)
        rejections = int(budget.get("total_rejections") or 0)
        group["rejections"] += rejections
        group["trials_rejected"] += int(rejections > 0)

        history = record["data"].get("chat_history", []) or []
        precision, invalid = requested_precisions(history)
        condition_key_ = (record["model"], record["condition"])
        precision_groups[condition_key_].update(precision)
        precision_groups[condition_key_]["invalid_actions"] += invalid

    print("# Budget rejection table")
    print("model,condition,difficulty,n,success_pct,mean_spend,trials_with_rejection_pct,mean_rejections")
    for key in sorted(groups, key=lambda item: (item[0], condition_key(item[1]), item[2])):
        model, condition, difficulty = key
        group = groups[key]
        success_pct = 100 * group["successes"] / group["success_n"]
        print(
            f"{model},{condition},{difficulty},{group['n']},{success_pct:.1f},"
            f"{group['spent'] / group['n']:.1f},"
            f"{100 * group['trials_rejected'] / group['n']:.1f},"
            f"{group['rejections'] / group['n']:.2f}"
        )

    print("\n# Requested precision levels")
    print("These count points in parseable assistant <run_experiment> actions, including requests later rejected for cost.")
    print("model,condition,precision_1,precision_2,precision_3,precision_4,precision_5,other,invalid_actions")
    for key in sorted(precision_groups, key=lambda item: (item[0], condition_key(item[1]))):
        model, condition = key
        levels = precision_groups[key]
        other = sum(
            count
            for level, count in levels.items()
            if level not in {1, 2, 3, 4, 5, "invalid_actions"}
        )
        print(
            f"{model},{condition},{levels[1]},{levels[2]},{levels[3]},"
            f"{levels[4]},{levels[5]},{other},{levels['invalid_actions']}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("format-examples", "budget-details"))
    parser.add_argument(
        "--trace-glob",
        default=DEFAULT_GLOB,
        help=f"trace CSV glob (default: {DEFAULT_GLOB})",
    )
    args = parser.parse_args()
    records = load_records(args.trace_glob)
    if args.command == "format-examples":
        command_format_examples(records)
    else:
        command_budget_details(records)


if __name__ == "__main__":
    main()
