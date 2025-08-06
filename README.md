# Human Value Detection from Text

## Overview
This repository contains the code, data preparation scripts, and evaluation results for a **multi-label classification** task aimed at detecting **human values** in textual arguments.  
The project was developed as part of an **Information Retrieval** course assessment and was inspired by **SemEval Task 4**.

The objective is to identify the presence or absence of **20 predefined value categories** within given arguments using state-of-the-art transformer models.

## Dataset
The dataset consists of arguments annotated with 20 value categories.  
Each argument contains:
- **Conclusion**
- **Stance** (supporting or opposing the conclusion)
- **Premise**

Data splits:
- **Training**, **Validation**, and **Test** sets
- Additional evaluation sets for robustness:
  - **Validation-Zhihu** (Chinese Q&A website)
  - **Test-Nahjalbalagha** (religious text dataset)
  - **Test-NYT** (New York Times COVID-related articles)

Labels are binary (1 = present, 0 = absent) for each value category.

## Preprocessing
- Merged argument and label datasets into a single DataFrame.
- Concatenated **premise**, **conclusion**, and **stance** into a single `text` column.
- Created a `category` column indicating active value categories.
- Prepared data in **SimpleTransformers**-compatible format with `text` and `labels` columns.

## Models
We evaluated three transformer-based architectures using **SimpleTransformers**:
- **BERT**
- **ALBERT** (A Lite BERT)
- **RoBERTa** (Robustly Optimized BERT Approach)

## Experimental Setup
- **Task**: Multi-label classification
- **Metric**: Macro F1-score (primary), along with precision, recall, confusion matrices, MPR, and MBR
- **Framework**: Python + SimpleTransformers
- **Training**: Fine-tuned each model separately on the training set and evaluated on all datasets.

## Results
- **RoBERTa** consistently achieved the highest F1 scores on most datasets (First, Zhihu, Nahjalbalagha).
- **BERT** outperformed others on the **New York Times dataset** due to higher precision.
- ALBERT performed consistently but lagged behind BERT and RoBERTa.

| Dataset          | Best Model | Macro F1 Score | Notes |
|------------------|-----------|---------------|-------|
| First            | RoBERTa   | Highest overall recall and precision | Balanced across labels |
| Zhihu            | RoBERTa   | Best across all metrics | Robust to domain shift |
| Nahjalbalagha    | RoBERTa   | Consistent top performer | High precision and recall |
| NYT              | BERT      | Best precision | Slightly lower recall than RoBERTa |



The dataset can be accessed [here](https://zenodo.org/records/10564870).

