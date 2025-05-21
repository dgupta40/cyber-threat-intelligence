# Sentence Embedding Model Analysis for CTI Pipeline

## Overview

This report analyzes different sentence transformer models to determine the optimal embeddings for cybersecurity text data.

## Model Comparison

| Model | Embedding Quality | Speed | Efficiency | Cyber Understanding | Total Score |
| --- | --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | 4.84 | 4.55 | 11.05 | 6.82 | 6.19 |
| all-mpnet-base-v2 | 4.96 | 0.45 | 9.66 | 6.95 | 5.03 |
| all-distilroberta-v1 | 4.78 | 1.25 | 10.71 | 8.30 | 5.67 |
| paraphrase-MiniLM-L3-v2 | 4.98 | 10.00 | 11.26 | 5.99 | 7.43 |
| all-MiniLM-L12-v2 | 4.84 | 3.12 | 10.65 | 7.98 | 6.07 |

## Recommendation

Based on our analysis, **paraphrase-MiniLM-L3-v2** is the recommended model with a score of 7.43/10.

Switching from your current model (all-MiniLM-L6-v2) to paraphrase-MiniLM-L3-v2 could provide significant improvements (score difference: +1.24).

## Detailed Justification

### all-MiniLM-L6-v2 (Current Model)

- **Embedding Dimension**: 384
- **Model Size**: 86.65 MB
- **Encoding Speed**: 1447.96 texts/second (batch size 64)
- **Clustering Quality (Silhouette)**: 0.0660
- **Cybersecurity Semantic Understanding**: Contrast ratio of 3.69

This model offers an excellent balance of performance, size, and semantic understanding for cybersecurity text. The 384-dimensional embeddings provide sufficient representational power while keeping computational requirements manageable. Key strengths include compact model size, fast encoding speed, strong cybersecurity concept differentiation. 

## Full Evaluation Data

See the accompanying JSON file for complete evaluation metrics.
