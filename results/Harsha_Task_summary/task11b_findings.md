# Task 11b: extraction features and embeddings for the category classifier

Same split (173 train / 44 holdout) and same CV as D11. CV = 5-fold, 3 seeds averaged, all 217 events.

## Run in the cloud workspace (no embeddings)
| variant | holdout acc | holdout macro-F1 | CV acc | CV macro-F1 |
|---|---|---|---|---|
| tfidf (baseline) | 0.568 | 0.477 | 0.598 | 0.494 |
| extraction features only | 0.341 | 0.244 | 0.335 | 0.256 |
| tfidf + extraction features | 0.455 | 0.400 | 0.525 | 0.409 |

Weight sweep on the feature block (0.1 / 0.3 / 0.6): CV accuracy 0.599 / 0.568 / 0.542, holdout 0.545 / 0.591 / 0.500. Neutral at best.

Conclusion: the extraction features (trigger rule, trigger score, number of units, system/org found, source type) do NOT help category. They describe how an announcement is worded, not which of the 8 categories it belongs to, and they are noisy (trigger score is not an event measure). Do not add them to the final classifier for category. They remain useful for the extraction report and the timeline, not for classification.

## Embeddings (run in Colab, BAAI/bge-small-en-v1.5, 395 labelled docs)
| variant | holdout acc | holdout macro-F1 | CV acc | CV macro-F1 |
|---|---|---|---|---|
| tfidf (baseline) | 0.568 | 0.477 | 0.598 | 0.494 |
| extraction features only | 0.341 | 0.244 | 0.335 | 0.256 |
| tfidf + features | 0.455 | 0.400 | 0.525 | 0.409 |
| **embeddings only** | **0.636** | **0.580** | **0.602** | **0.538** |
| embeddings + features | 0.545 | 0.447 | 0.518 | 0.429 |
| embeddings + tfidf + features | 0.500 | 0.404 | 0.559 | 0.445 |

## Reading the results
1. Embeddings alone are the best variant, but the gain is small. Holdout accuracy 0.636 vs 0.568 is 28 vs 25 of 44 documents. CV accuracy is flat (0.602 vs 0.598); macro-F1 is higher (0.538 vs 0.494). Call it "slightly better, not clearly better".
2. Every variant that adds the extraction features is worse than the same model without them (embeddings 0.636 -> 0.545, tfidf 0.568 -> 0.455 on holdout). Extraction features stay out of the category classifier.
3. Combining all three feature groups is worse than embeddings alone; the extra blocks add noise with only 173 training rows.
4. Both results are one split, no tuning of C or the embedding model. Next step options: a larger embedding model (bge-base), embedding only the trigger unit, or more labelled data (the next annotation batch). Final classifier choice for the report: embeddings only + logistic regression.
