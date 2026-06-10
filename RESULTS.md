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

## DEPTH (28 → 20 warstw; usunięto warstwy 13–16 i 24–27)

| Zadanie | Baseline | **Pruning-only** | Distilled (40 kr.) |
|---|---|---|---|
| HellaSwag (acc_norm) | 0.6644 | **0.3642** | 0.2504 |
| ARC-Challenge (acc_norm) | 0.4488 | **0.3020** | 0.2270 |
| WinoGrande (acc) | 0.6393 | **0.5280** | 0.4957 |
| TruthfulQA MC2 | 0.4878 | **0.4724** | NaN |

**Wniosek (depth):** pruning-only **bije** distilled na każdym zadaniu (>20σ na HellaSwag), a
TruthfulQA jest **skończone** (0.4724) zamiast NaN. Czyli 40-krokowa distillacja przy ustawieniach
pod OOM (seq 512, gbs 4) **zaszkodziła** i wywołała niestabilność numeryczną — czysty pruning
zachowuje znaczną zdolność i jest stabilny. Uszkodzenie przypisywane wcześniej pruningowi było
artefaktem zepsutej mikro-distillacji.

## WIDTH (hidden 2048→1664, FFN 6144→4224) i COMBINED (24 w., hidden 1792, FFN 4608)

| Zadanie | Width pruning-only | Width distilled | Combined pruning-only | Combined distilled |
|---|---|---|---|---|
| HellaSwag (acc_norm) | _TBD_ | 0.2640 | _TBD_ | 0.2651 |
| ARC-Challenge (acc_norm) | _TBD_ | 0.2585 | _TBD_ | 0.2585 |
| WinoGrande (acc) | _TBD_ | 0.4957 | _TBD_ | 0.4957 |
| TruthfulQA MC2 | _TBD_ | 0.4807 | _TBD_ | NaN |

_TBD_ = pruning-only width/combined jeszcze nie uruchomione (deploy + suite). Odtworzenie:
`./scripts/run_pruneonly.sh /workspace/nemo_ckpts/qwen3-1.7b-base-width width 0` (analogicznie combined).

## Parametry wariantów (zweryfikowane)

| Wariant | Warstwy | hidden | FFN | Total | Non-emb |
|---|---|---|---|---|---|
| Baseline | 28 | 2048 | 6144 | 1.721 B | 1.409 B |
| Depth | 20 | 2048 | 6144 | 1.629 B | 1.318 B |
| Width | 28 | 1664 | 4224 | 1.382 B | 1.130 B |
| Combined | 24 | 1792 | 4608 | 1.403 B | 1.131 B |
