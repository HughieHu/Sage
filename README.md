# Sage Benchmark

Sage is a benchmark for evaluating deep research agents on scientific literature retrieval and comprehension. It contains 1,200 expert-curated queries across four academic domains, split into two question types.

## Structure

```
Sage/
├── Sage_Short_Form_Questions/    # 600 queries (150 per domain)
│   ├── computer_science.json
│   ├── healthcare.json
│   ├── humanities.json
│   └── natural_science.json
└── Sage_Open_Ended_Questions/    # 600 queries (150 per domain)
    ├── computer_science.json
    ├── healthcare.json
    ├── humanities.json
    └── natural_science.json
```

## Question Types

### Short-Form Questions (Exact Match)

Each query provides a detailed description of a target paper, including publication metadata, key findings, and citation relationships. The task is to identify the exact paper.

**Fields:**
- `complete_query`: Detailed natural language query describing the paper
- `ground_truth`: The target paper (ID and title)

### Open-Ended Questions (Weighted Recall)

Research-style open-ended questions that span two related papers. The task requires retrieving relevant papers across multiple relevance tiers.

**Fields:**
- `question`: Natural language research question
- `ground_truth`: Papers at two relevance levels (`most_relevant`, `relevant`)

## Domains

| Domain | Short-Form | Open-Ended |
|--------|-----------|------------|
| Computer Science | 150 | 150 |
| Healthcare | 150 | 150 |
| Humanities | 150 | 150 |
| Natural Science | 150 | 150 |

## Retrieval Corpus

Both question types are answered against **the same corpus** — one shared pool of
183,765 full-text papers across the four domains. A short-form query and an open-ended
query in the same domain search exactly the same documents; nothing is held out or
swapped between the two settings.

The corpus and the prebuilt retrieval indices are on the Hugging Face Hub:

| Dataset | Contents | Size |
|---|---|---|
| [SAGE-Corpus](https://huggingface.co/datasets/HughieHu/SAGE-Corpus) | Full text (Markdown) + per-paper metadata | 9.5 GB |
| [SAGE-Index-BM25](https://huggingface.co/datasets/HughieHu/SAGE-Index-BM25) | BM25 indices + the chunk metadata every retriever needs | 23 GB |
| [SAGE-Index-Qwen3](https://huggingface.co/datasets/HughieHu/SAGE-Index-Qwen3) | Qwen3-Embedding-8B, raw + augmented | 162 GB |
| [SAGE-Index-ReasonIR](https://huggingface.co/datasets/HughieHu/SAGE-Index-ReasonIR) | ReasonIR, raw + augmented | 162 GB |
| [SAGE-Index-Qwen3-LoRA](https://huggingface.co/datasets/HughieHu/SAGE-Index-Qwen3-LoRA) | Qwen3-Embedding-8B + LoRA, augmented | 81 GB |

### Getting the corpus

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import snapshot_download

# everything (9.5 GB)
snapshot_download("HughieHu/SAGE-Corpus", repo_type="dataset",
                  local_dir="sage_corpus")

# or one domain
snapshot_download("HughieHu/SAGE-Corpus", repo_type="dataset",
                  local_dir="sage_corpus",
                  allow_patterns=["corpus/healthcare/*", "metadata/healthcare_*"])
```

```
sage_corpus/
├── corpus/<domain>/<shard>.json     # [{id, url, markdown}], ~124 papers per shard
└── metadata/<domain>_mapping.json   # doc_id, url, paper_title, authors, year, venue, abstract
```

### Getting a retrieval index

Start with BM25 — it needs no GPU and carries `chunk_meta/`, the row-to-document map
that the dense indices are aligned to.

```python
snapshot_download("HughieHu/SAGE-Index-BM25", repo_type="dataset", local_dir="sage_bm25")
```

Dense indices are float32, L2-normalised `.npy` matrices; row *i* is entry *i* of the
matching `chunk_meta/<domain>/chunk256_merged_index.json`.

```python
import numpy as np, json
from huggingface_hub import hf_hub_download

emb = np.load(hf_hub_download("HughieHu/SAGE-Index-Qwen3",
              "augmented/computer_science/chunk256_embeddings.npy",
              repo_type="dataset"), mmap_mode="r")        # (1414444, 4096)

meta = json.load(open(hf_hub_download("HughieHu/SAGE-Index-BM25",
       "chunk_meta/computer_science/chunk256_merged_index.json", repo_type="dataset")))

top = np.argsort(-(emb @ encode(query)))[:10]
for i in top:
    print(meta[i]["doc_id"], meta[i]["paper_title"])
```

Each dense repository has a `raw/` variant and an `augmented/` variant; `augmented/`
is encoded with the corpus augmentation described in the paper.

### Linking a question to the corpus

Ground truth is given as `{paperId, title}`. Evaluation matches on the normalised title
(`title.strip().lower()`), which is also what `metadata/<domain>_mapping.json` holds in
`paper_title`; records resolved against Semantic Scholar additionally carry
`s2_paper_id`, so you can join on either.


## Evaluation Metrics

- **Short-Form**: Exact Match — whether the retrieved paper matches the ground truth
- **Open-Ended**: Weighted Recall — recall across relevance tiers with decreasing weights
