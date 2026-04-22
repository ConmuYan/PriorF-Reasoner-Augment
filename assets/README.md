# PriorF-Reasoner Assets

This directory holds the local assets needed by the Reasoner pipeline.

Expected layout:

```text
assets/
  data/
    Amazon.mat
    YelpChi.mat
  teacher/
    amazon/
      best_model.pt
      model_summary.json
    yelpchi/
      best_model.pt
      model_summary.json
  teacher_exports/
    amazon_train_evidence.parquet
    amazon_test_evidence.parquet
    yelpchi_train_evidence.parquet
    yelpchi_test_evidence.parquet
```