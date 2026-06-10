# magisterka — benchmark pruningu LLM (depth / width / combined)

Spójny benchmark do porównywania wydajności pruningu wzdłuż osi depth, width i mieszanej.
Metoda bazowa: Minitron (NVIDIA / NeMo / modelopt). Model poligonowy: Qwen3-1.7B-Base.

## Układ
- `scripts/` — własne skrypty (import, eval, perplexity, narzędzia benchmarku)
- `configs/` — configi eksperymentów (reproducibility) [w budowie]

Modele, checkpointy, dane i wyniki NIE są w repo (patrz `.gitignore`) — transfer przez rsync/wandb.
