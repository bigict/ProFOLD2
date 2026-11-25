# Fold-switcher mining & multi-conformation probes

Research tooling for detecting protein fold-switching from the PDB and probing
ProFOLD2 multi-conformation generation. Two parts:

## 1. Fold-switcher mining pipeline (production route: hybrid)

Detect fold-switchers by **dual clustering**: MMseqs2 sequence clusters ×
Foldseek structure clusters, with an intra-fold swSS supplement.

| stage | script | purpose |
|---|---|---|
| sequence grouping | (MMseqs2 easy-cluster, 95% id) | same-protein sequence clusters |
| structure extraction | `extract_structs.py` | raw mmCIF.gz from `pdb_train.zip` for Foldseek |
| structure clustering | (Foldseek easy-cluster, 3Di+AA, `-c 0.8`) | fold-level clusters |
| **hybrid detection** | `hybrid_detect.py` | seq cluster spanning ≥2 fold clusters = fold-level switcher; single-fold cluster split by swSS = topology switcher (Mad2); fold0=majority, full retention |
| backbone extraction | `fast_extract.py` | fast mmCIF `_atom_site` backbone parse (N/CA/C/O) |
| conformation metrics | `stage2_metrics.py` | `classify_pair`: fixed-alignment TM-score + **swSS** (SS-change among switching residues) + max_disp + rigid-body tag |
| pairwise mining (baseline) | `stage2_mine.py`, `dual_clustering.py` | swSS≥0.30 + max_disp≥12 switch-pair detection / greedy conformation clustering |
| APSP grouping (superseded) | `apsp.py`, `mine_switchers.py` | exact suffix-prefix overlap grouping (MMseqs2 dominates) |
| validation | `cross_validate.py`, `compare_grouping.py`, `compare_v1v2.py` | Porter known-switcher recall, mode comparisons |
| pair extraction | `extract_pair.py` | build fold0/fold1 backbone pair npz from two structures |

**Key criterion — swSS** (`stage2_metrics.classify_pair`): a switch pair has
swSS ≥ 0.30 AND max_disp ≥ 12 Å. swSS = secondary-structure-change fraction among
the switching-field residues (core-aligned displacement > 0.5·max). This catches
same-fold topology switches (Mad2 O/C, high swSS) while excluding rigid-body
domain motion (calmodulin open/closed, low swSS) — the two blind spots of
TM-score/Foldseek-only and SS-only methods respectively.

**Result** (default 3Di + swSS hybrid): 14,083 switchers (8,976 fold-level +
5,107 topology), Porter known-switcher recall 60.2% (56/93), calmodulin correctly
excluded, Mad2/RfaH/KaiB recovered. Full-retention manifest keeps every PDB entry
per switcher cluster (fold0/fold1 labels; random draw at training use).

TM-align mode (`--exact-tmscore 1`) compared and rejected: more sensitive (+4
fold-level switches) but less precise (calmodulin false positive; ~5–10× slower).

## 2. Multi-conformation generation probes (`probe_*.py`)

Probe ProFOLD2 trunk/StructureModule representability and conditioning for
alternative folds (XCL1/RfaH/Mad2 pairs in `data/`):

- `probe_repr.py`, `probe_decode.py`, `probe_norm.py` — representation / decoding / LayerNorm washout.
- `probe_fit.py`, `probe_grad.py`, `probe_loss_dissect.py` — fit-alone representability (~2 Å), gradient-flow, FAPE loss dissection.
- `probe_v1..v6.py` — conditioning iterations; **v6** = frozen-trunk field-gated adaLN-Zero (positive on XCL1; trunk finetuning destabilizes).

Findings: trunk CAN represent alternative folds; ~2 Å floor is FAPE local-frame
geometry (not a bug); frozen trunk + per-residue switching-field conditioning
steers modes; mechanism bounded by IPA-reachability (large rigid-body domain
switches need a generative decoder).

## 3. AFDB augmentation (direction-2 data prep)

Transfer mined fold0/fold1 templates onto AFDB to augment per-mode training data
for the FM decoder. Runs on the AFDB machine (192 cores / 1.5T RAM).

| stage | script | purpose |
|---|---|---|
| template extraction | `extract_templates.py` | per switcher per fold, write representative backbone PDB (`sw<seq_rep>_fold<N>.pdb`) + `manifest.json` |
| AFDB search | `afdb_search.py --stage search` | Foldseek `easy-search` templates vs AFDB (3Di+AA, `-c 0.8`, TM-score output) |
| per-mode assignment | `afdb_search.py --stage assign` | assign each AFDB hit to the fold of its best-matching template (TM/coverage-gated) → `afdb_per_fold.json` |

Caveats (§4.2/§4.3): AFDB hits are **pseudo-labels** (predicted structures) —
downweight + identity-gate; per-mode augmentation ≠ new-switcher discovery (a
switcher must match BOTH folds); downstream construction is backbone-only.

## Data

- `data/*.npz` — small fold0/fold1 backbone test pairs (XCL1, RfaH, Mad2).
- Mining outputs (`mining/`, large) are **not committed** — regenerate via the
  pipeline against `~/shared_hosts/pdb_train.zip`.

## Dependencies

`numpy`, `biopython`, `pyahocorasick`, `edlib`; external tools MMseqs2 + Foldseek
(static binaries). Python ≥ 3.10.