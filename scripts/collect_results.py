#!/usr/bin/env python3
"""Zbiera wyniki pruning-only z wandb (projekt magisterka-pruning) w tabelę zadanie x model.

Runy nazwane <model>-<task> (np. depth20-hellaswag), tworzone przez run_pruneonly.sh.
Metryka raportowa: acc_norm dla hellaswag/arc_challenge, acc dla winogrande/truthfulqa_mc2.

Uruchom na hoście (venv z wandb + internet):
    export PATH="$HOME/pruning/venv/bin:$PATH"
    python scripts/collect_results.py [entity/project]
"""
import sys
import wandb

REPORT = {
    "hellaswag": "acc_norm",
    "arc_challenge": "acc_norm",
    "winogrande": "acc",
    "truthfulqa_mc2": "acc",
}


def metric_value(summary, want):
    """Wyłuskaj wartość metryki z run.summary, odporne na format klucza (np. 'acc_norm,none')."""
    for k, v in summary.items():
        if not isinstance(v, (int, float)):
            continue
        kl = k.lower()
        if want == "acc_norm" and "acc_norm" in kl:
            return v
        if want == "acc" and "acc" in kl and "acc_norm" not in kl:
            return v
    return None


def main():
    api = wandb.Api()
    project = sys.argv[1] if len(sys.argv) > 1 else f"{api.default_entity}/magisterka-pruning"
    print("projekt:", project)

    table = {}
    for run in api.runs(project):
        model, _, task = run.name.partition("-")
        if task not in REPORT:
            continue
        table[(model, task)] = metric_value(run.summary, REPORT[task])

    models = sorted({m for (m, _) in table})
    print(f"\n{'zadanie':18}" + "".join(f"{m:>14}" for m in models))
    for t in REPORT:
        line = f"{t:18}"
        for m in models:
            v = table.get((m, t))
            line += f"{(f'{v:.4f}' if isinstance(v, float) else str(v)):>14}"
        print(line)


if __name__ == "__main__":
    main()
