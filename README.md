# magisterka — benchmark pruningu LLM (depth / width / combined)

Spójny benchmark do porównywania wydajności pruningu wzdłuż osi depth, width i mieszanej.
Metoda bazowa: Minitron (NVIDIA / NeMo / modelopt). Model poligonowy: Qwen3-1.7B-Base.

## Układ
- `scripts/` — własne skrypty (import, deploy, eval)
- `configs/` — configi eksperymentów (reproducibility) [w budowie]

Modele, checkpointy, dane i wyniki NIE są w repo (patrz `.gitignore`) — transfer przez rsync/wandb.
Dokumentacja metodyczna (runbook, stan, troubleshooting) trzymana lokalnie, poza repo.

## Ewaluacja pruning-only (natywna, bez HF) — `scripts/run_pruneonly.sh`

Zwalidowana ścieżka (2026-06-10) do ewaluacji **surowego sprunowanego** checkpointu NeMo
**bez distillacji i bez eksportu do HF**:

1. `deploy(..., legacy_ckpt=True)` stawia serwer OpenAI (PyTriton + FastAPI :8080).
   `legacy_ckpt=True` jest kluczowe — degraduje brak kluczy `_extra_state` z fatal erroru
   do ostrzeżenia, więc surowy pruned checkpoint ładuje się bez re-save.
2. `lm_eval --model local-completions` odpytuje endpoint; zadania loglikelihood (`acc_norm`)
   działają, bo endpoint zwraca `logprobs`+`echo`. Wyniki → Weights & Biases.

Uruchamianie **w kontenerze NeMo** (`nvcr.io/nvidia/nemo:25.09`):
```bash
export WANDB_API_KEY=<klucz>      # ~/.netrc z hosta nie jest widoczny w kontenerze
./scripts/run_pruneonly.sh /workspace/nemo_ckpts/qwen3-1.7b-base-depth20 depth20 0
```
Argumenty: `<checkpoint_dir> <run_name> [gpu_id]`. Zadania: hellaswag, arc_challenge,
winogrande, truthfulqa_mc2. Projekt wandb: `magisterka-pruning`.

Gotchas (szczegóły w lokalnym `troubleshooting.md` #1b):
- ubij stare serwery przed deployem (kolizja portów → `403 triton-access-token`) — skrypt to robi;
- `tokenized_requests=False` (endpoint przyjmuje prompt-string, nie list[int]);
- osobny `HF_DATASETS_CACHE` dla kontenera (wspólny cache z hostem psuje wersja `datasets`).
