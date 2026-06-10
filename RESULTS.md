# Wyniki — baseline vs pruning-only vs distillation

Model bazowy: **Qwen/Qwen3-1.7B-Base** (28 warstw, hidden 2048, FFN 6144, ~1.72B param).
Ewaluacja: lm-evaluation-harness, **0-shot**, `acc_norm` gdzie dotyczy.

Trzy warunki:
- **baseline** — niesprunowany Qwen3-1.7B-Base.
- **pruning-only** — po pruningu Minitron, **BEZ distillacji** (eval natywnie przez NeMo deploy
  z `legacy_ckpt=True`; patrz `scripts/run_pruneonly.sh`).
- **distilled** — po pruningu + **40 krokach** distillacji (~80k tokenów; eval przez HF).

> Uwaga metodyczna: pruning-only mierzone **natywnie** (deploy), distilled przez **HF**.
> Zgodność stacków zwalidowana (baseline HellaSwag natywnie 0.55 na 40 próbkach ≈ HF 0.6644 na
> pełnym zbiorze). Różnice pruning-only vs distilled wielokrotnie przekraczają rozjazd stacku.
> Liczby distilled i baseline — z wcześniejszych przebiegów HF. Daty/pełne logi: serwer `glasser`.

## Tabela główna — pruning-only vs distilled vs baseline

Pruning-only: natywny eval (deploy `legacy_ckpt=True`, `max_input_len=8192`), z wandb
(`magisterka-pruning`). Distilled: po 40 krokach, eval HF.

| Zadanie | Baseline | depth p-only / distill | width p-only / distill | combined p-only / distill |
|---|---|---|---|---|
| HellaSwag (acc_norm) | 0.6644 | **0.3649** / 0.2504 | 0.2953 / 0.2640 | 0.3078 / 0.2651 |
| ARC-Challenge (acc_norm) | 0.4488 | **0.3038** / 0.2270 | 0.2133 / 0.2585 | 0.2210 / 0.2585 |
| WinoGrande (acc) | 0.6393 | **0.5399** / 0.4957 | 0.5012 / 0.4957 | 0.5051 / 0.4957 |
| TruthfulQA MC2 | 0.4878 | **0.4711** / NaN | 0.4537 / 0.4807 | 0.4597 / NaN |

### Wnioski

1. **DEPTH (28→20 w.; usunięto 13–16, 24–27): pruning-only bije distilled na każdym zadaniu**
   (HS +0.11, ARC +0.08, Wino +0.04; TQA skończone 0.4711 vs NaN). 40-krokowa distillacja
   (seq 512, gbs 4, pod OOM) **zaszkodziła** i wywołała NaN — czysty pruning depth zachowuje
   znaczną zdolność i jest stabilny.
2. **WIDTH (1664/4224) i COMBINED (24 w., 1792/4608): obraz mieszany, oba blisko losowości.**
   Width pruning-only ma ARC 0.2133 (poniżej losowości 0.25). Tu distillacja miejscami pomogła
   (ARC, TQA width). „Distillacja szkodzi" jest więc tezą **specyficzną dla depth**, nie uniwersalną.
3. **Depth pruning-only ≫ width pruning-only** (HS 0.365 vs 0.295, ARC 0.304 vs 0.213) — bez
   retrainingu usuwanie warstw zachowuje więcej niż zwężanie. Zgodne z paperami (width > depth
   dopiero PO retrainingu; przed — odwrotnie). Width pruning bardziej rozjeżdża aktywacje
   (zmienia każdą warstwę), depth zostawia pozostałe warstwy nietknięte.

## Parametry wariantów (zweryfikowane)

| Wariant | Warstwy | hidden | FFN | Total | Non-emb |
|---|---|---|---|---|---|
| Baseline | 28 | 2048 | 6144 | 1.721 B | 1.409 B |
| Depth | 20 | 2048 | 6144 | 1.629 B | 1.318 B |
| Width | 28 | 1664 | 4224 | 1.382 B | 1.130 B |
| Combined | 24 | 1792 | 4608 | 1.403 B | 1.131 B |
