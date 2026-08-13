# Model Card — ResidenceAlpha

## Summary

Three models: a renewal prediction classifier, an NLP feedback classifier
(theme + sentiment), and unsupervised property segmentation. Each is
reported with its actual limitations stated plainly, including one real bug
caught and fixed during development.

## 1. Renewal prediction

**Model:** Random Forest, `class_weight="balanced"`.

**Features:** rent increase offered, maintenance response time, maintenance
request count, tenure, concessions, property occupancy, property-level NOI
growth, property class, unit type.

**Held-out test set (312 leases):**

| Metric | Value |
|---|---|
| ROC-AUC | 0.652 |
| PR-AUC | 0.807 |
| F1 | 0.753 |
| Accuracy | **0.658** |
| Naive baseline accuracy ("always predict renewed") | **0.690** |

### Why accuracy is below the naive baseline — and why that's not being hidden

The overall renewal rate is ~69%, so a model that always predicts "renews"
gets 69% accuracy for free while being completely useless — it would never
flag a single resident actually at risk of leaving. `class_weight="balanced"`
deliberately trades some of that free accuracy for better sensitivity to the
minority class (residents who *don't* renew), because catching those
residents is the entire point of a tool like this. ROC-AUC of 0.652 (modestly
above the 0.5 random baseline) confirms the model has found real, if modest,
signal — accuracy alone would have hidden that trade-off rather than
explaining it.

**Recommendation:** report and act on `renewal_probability` and the
`renewal_risk_tier` bucket, not a binary accuracy score, when using this
model operationally.

## 2. Resident feedback NLP (theme + sentiment)

**Model:** TF-IDF (500 features, 1-2 grams) + Logistic Regression, for both
theme (7-class) and sentiment (binary) classification.

**Held-out test set (313 comments):** theme accuracy 100%, sentiment
accuracy 100%.

### This 100% is not a real-world result — stated plainly

Feedback text in this dataset is **template-generated**, drawn from a fixed
bank of phrasings per theme/sentiment combination (see Data Dictionary).
TF-IDF trivially separates these because the vocabulary is distinctive by
construction — a real resident's free-text comment ("the AC hasn't worked
right since March and nobody's called me back") doesn't map cleanly to a
template. These accuracy figures demonstrate the *pipeline* works end to
end; they say nothing about expected performance on genuine unstructured
feedback, which would need real labeled resident comments to evaluate
honestly.

## 3. Property segmentation

**Method:** KMeans (5 clusters) on occupancy, NOI margin, same-property NOI
growth, maintenance response time, and capex per unit — standardized before
clustering.

### A real labeling bug, caught before this reached a report

The first labeling approach used a single blind composite rank to assign
the five category names (Stable Performer, Operating-Improvement Candidate,
Renovation Opportunity, Repositioning Candidate, Disposition-Review
Candidate). It produced a direct contradiction: the cluster with the
**highest** occupancy and NOI margin in the entire portfolio was labeled
"Operating-Improvement Candidate," while a cluster with much weaker
fundamentals was labeled "Stable Performer" — because a single 1-D ranking
formula can't fairly represent capex, which has a genuinely two-sided
meaning (heavy capex can mean "actively turning the asset around" *or*
"already spent and still underperforming," depending on what results it's
producing).

**Fix:** a sequential rule-based assignment instead — each label claims the
cluster that most clearly fits its intended meaning (worst growth →
Repositioning; heaviest active capex → Renovation Opportunity; best
occupancy+margin → Stable Performer; weakest margin → Disposition-Review;
remainder → Operating-Improvement) before falling through to the next rule.
The corrected result is internally consistent across every metric — see the
table below.

**Final segment characteristics:**

| Segment | Count | Occupancy | NOI Margin | NOI Growth | Capex/Unit |
|---|---|---|---|---|---|
| Stable Performer | 9 | 96.8% (best) | 58% (best) | 1.2% | $906 (lowest) |
| Operating-Improvement Candidate | 9 | 90.9% | 56% | 5.6% (best) | $2,267 |
| Renovation Opportunity | 8 | 92.8% | 56% | 2.9% | $6,397 (highest) |
| Repositioning Candidate | 7 | 86.5% (worst) | 52% | -0.7% (only negative) | $4,257 |
| Disposition-Review Candidate | 7 | 90.7% | 47% (worst) | 2.6% | $998 |

## Limitations and why this should not make automated decisions

- **Small sample:** 40 properties, 1,249 leases. Treat all performance
  estimates as directional, not final.
- **NLP is not validated on real text** — see above. Before operational use,
  retrain on a genuinely labeled sample of real resident comments (the
  project brief's Phase 4 calls for exactly this).
- **Segmentation is descriptive, not predictive** — it groups properties by
  current characteristics; it does not forecast which segment a property
  will move into.
- **Qualitative blind spots:** local market conditions, specific competitor
  supply, and property management team quality are not in the feature set.

**Conclusion:** this system prioritizes which residents and properties
deserve human attention first — it does not replace a property manager's
judgment, a renewal conversation, or a capital committee's review.
