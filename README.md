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

## Evaluation Metrics

- **Short-Form**: Exact Match — whether the retrieved paper matches the ground truth
- **Open-Ended**: Weighted Recall — recall across relevance tiers with decreasing weights
