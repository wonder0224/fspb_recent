# FSPB: Feature-Saturation-Preserving Binarization

Reproduction package for the paper *Feature-Saturation-Preserving Binarization:
Reading an LLM's Discriminative Structure as a Bounded Binary Feature Array
within a Closed World* (Won Kyoung Kim).

This repository contains the token inventories, the binarization and evaluation
code, and the canonical result files needed to reproduce every table and figure
in the paper.

## Repository layout

```
.
├── data/              token inventories (units) for the three closed worlds
│   ├── news_legal_units.json       508 legal-opinion tokens
│   ├── oncology_units.json         4629 oncology tokens
│   ├── news_diverse_units.json     2327 mixed-news tokens
│   └── world_entities.json         named-entity world (NER corroboration)
├── embeddings/        embedding matrices (.npy) — see "Embeddings" below
├── scripts/           all experiment and analysis code
├── results/           canonical result files (JSON) used by the paper
├── figures/           the four paper figures (PDF, Type-42 fonts)
├── requirements.txt
└── README.md
```

## Embeddings

The embedding matrices (`*.npy`, ~140 MB total) are **not** stored in this Git
repository. They are archived at Zenodo (DOI below) and can also be regenerated
from the token inventories:

```bash
# primary embedding (OpenAI text-embedding-3-large, 3072-d)
python scripts/embed_only.py            # needs OPENAI_API_KEY
# secondary models (e5-large-v2 1024-d; all-mpnet-base-v2 768-d)
python scripts/embed_other_models.py    # downloads public HF models
```

Expected files (place under `embeddings/`):

| file | model | dim | world |
|------|-------|-----|-------|
| `news_legal_embeddings.npy`   | text-embedding-3-large | 3072 | legal |
| `oncology_embeddings.npy`     | text-embedding-3-large | 3072 | oncology |
| `news_diverse_embeddings.npy` | text-embedding-3-large | 3072 | diverse |
| `oncology_emb_e5.npy`         | e5-large-v2            | 1024 | oncology |
| `oncology_emb_mpnet.npy`      | all-mpnet-base-v2      | 768  | oncology |
| `world_entities_emb.npy`      | text-embedding-3-large | 3072 | NER world |

Note: `e5-large-v2` is run **without** the usual `query:`/`passage:` prefix, for
a clean cross-model comparison (see `embed_other_models.py`).

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the paper

Each prediction maps to specific scripts. Five random seeds (42, 7, 123, 2024,
99) are used throughout; values in the paper are the seed mean ± sd.

### P1 — bounded semantic ceiling (Table 1)
```bash
python scripts/maxsuff_508.py        # legal ceiling (63)
python scripts/maxsuff_dom.py        # oncology (105), diverse (101) ceilings
python scripts/maxsuff_robust.py     # robustness to the meaningfulness criterion
python scripts/ceiling_threshold_robustness.py  # graded vs dominant-label test
python scripts/make_all_figs_type42.py          # render Figures 2 (ceiling plateau) and 3 (log fits)
```
The reported ceilings (63 / 101 / 105) are canonical single-run whole numbers;
`ceiling_threshold_robustness.py` re-measures them under a graded criterion
(label-overlap threshold tau) and confirms the ceiling shifts by only a few
features (legal 62–66, oncology 98–105; results/ceiling_threshold_robustness.json).

### P3 — information-theoretic rate (Table 2, Figure 2)
```bash
python scripts/measure_sublinear.py  # efficiency coefficient c = k / log2 N
python scripts/info_lower_bound.py   # log2 N floor
python scripts/power_proof.py        # log vs sqrt vs linear fits
python scripts/robust_z_kpool.py     # robustness to threshold z and pool size
```
Figure 2 is produced from the `measure_sublinear` outputs.

### P4 — post-hoc interpretability (Table 3, Figure 3)
```bash
python scripts/naturalclass_508.py       # branch sets vs random (Table 3)
python scripts/trie_person_508.py        # person-root trie
python scripts/extract_person_branch.py  # per-token branch raw (one seed)
python scripts/person_branch_5seed.py    # 5-seed quantities in the text
python scripts/branch_cohesion_508.py    # cohesion-by-depth
python scripts/make_figB_person_trie.py  # render Figure 3 from person_branch_raw.json
```
Figure 3 (person branch trie) is illustrative (one seed): it draws the actual
binary branch trie from `results/person_branch_raw.json`, where circle size is
the token count, colour the dominant role, and shading the purity. The
quantities in the text (64.2 ± 4.6 meaningful splits, deepest 12.6 ± 2.3) come
from `person_branch_5seed.py`. This analysis covers the person root only.

### H — NER corroboration (unsupervised entity-category recovery)
```bash
python scripts/world_entity_pipeline.py
python scripts/g_absorption_508.py
python scripts/g_entity_boundary.py
python scripts/h_world_absorption.py
python scripts/h_loc_event_diag.py
```

### I — role separation (fixed-basis stabilization)
```bash
python scripts/I45_repro.py          # Hungarian-aligned agreement ratios (1.3–1.6x)
python scripts/basis_robustness.py   # secondary check on the fixed 26-supersense basis
```
The paper uses a single fixed basis of all 26 WordNet noun supersenses, the same
for every domain; in any one world only some categories are activated (seven in
legal, ten in oncology) because the rest carry no discovered feature.
`basis_robustness.py` confirms that restricting the basis to the occurring
categories changes nothing (results/basis_robustness.json).

### Canonical results and gate check
```bash
python scripts/regenerate_canonical.py   # rebuilds results/canonical_results.json
```
The gate check verifies that the reference conditions reproduce the reported
figures; if it passes, the pipeline is reproduced.

## Data sources

The token inventories here are derived from public corpora:
- **legal**: court-opinion text (CourtListener)
- **oncology**: PubMed abstracts
- **diverse**: mixed-topic news

This repository ships the **token inventories** (the closed-world vocabularies)
and the code to regenerate embeddings, rather than redistributing the upstream
corpora.

## Citing

If you use this package, please cite the paper (citation block to be added on
acceptance) and this archive:

> Kim, Won Kyoung. 2026. FSPB reproduction package. Zenodo.
> https://doi.org/10.5281/zenodo.20791083

## License

Code: MIT (see LICENSE). Token inventories and derived results: CC BY 4.0.
