"""
train_models.py -- ResidenceAlpha data science layer

Three separate modeling exercises:

1. RENEWAL PREDICTION (supervised, genuinely uncertain): predicts whether a
   resident renews their lease. The target has real randomness baked into
   its generation (logistic model + noise), so there is honest prediction
   error to report -- same design principle used throughout this portfolio.

2. RESIDENT FEEDBACK NLP: classifies feedback comments by theme (7 classes)
   and sentiment (binary). IMPORTANT LIMITATION, stated here and in the
   model card: the feedback text is template-generated, not real free-text,
   which makes this classification task easier than it would be on real
   resident comments -- reported honestly rather than presented as if it
   were a harder, real-world NLP benchmark.

3. PROPERTY SEGMENTATION (unsupervised): groups properties into five
   asset-management categories per the project brief -- stable performer,
   operating-improvement candidate, renovation opportunity, repositioning
   candidate, disposition-review candidate.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    classification_report, confusion_matrix,
)
import shap
import joblib
import os

np.random.seed(42)
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

leases = pd.read_csv("data/leases.csv")
properties = pd.read_csv("data/properties.csv")

# ---------------------------------------------------------------------
# 1. RENEWAL PREDICTION
# ---------------------------------------------------------------------
merged = leases.merge(properties[["property_id", "occupancy_pct", "same_property_noi_growth_pct"]],
                       on="property_id", suffixes=("", "_property"))

NUMERIC_FEATURES = ["rent_increase_offered_pct", "maintenance_avg_response_days",
                     "maintenance_requests_count", "tenure_months", "concession_months",
                     "occupancy_pct", "same_property_noi_growth_pct"]
CAT_FEATURES = ["property_class", "unit_type"]

X = pd.get_dummies(merged[NUMERIC_FEATURES + CAT_FEATURES], columns=CAT_FEATURES, drop_first=True)
y = merged["renewed"]

renewal_feature_columns = X.columns.tolist()
with open("models/renewal_feature_columns.json", "w") as f:
    json.dump(renewal_feature_columns, f, indent=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Renewal base rate -- train: {y_train.mean():.2%}, test: {y_test.mean():.2%}")

renewal_model = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                        class_weight="balanced", random_state=42)
renewal_model.fit(X_train, y_train)
proba_test = renewal_model.predict_proba(X_test)[:, 1]
preds_test = (proba_test >= 0.5).astype(int)

renewal_metrics = {
    "roc_auc": round(roc_auc_score(y_test, proba_test), 4),
    "pr_auc": round(average_precision_score(y_test, proba_test), 4),
    "f1": round(f1_score(y_test, preds_test), 4),
    "accuracy": round(accuracy_score(y_test, preds_test), 4),
    "base_rate_baseline_accuracy": round(max(y_test.mean(), 1 - y_test.mean()), 4),
}
print("Renewal model metrics:", json.dumps(renewal_metrics, indent=2))
with open("reports/renewal_model_metrics.json", "w") as f:
    json.dump(renewal_metrics, f, indent=2)

# SHAP explainability
explainer = shap.TreeExplainer(renewal_model)
shap_values = explainer.shap_values(X_test)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values
plt.figure()
shap.summary_plot(sv, X_test, show=False, max_display=10)
plt.tight_layout()
plt.savefig("reports/renewal_shap_summary.png", dpi=150)
plt.close()

# Score full lease book
X_full = X.copy()
merged["renewal_probability"] = np.round(renewal_model.predict_proba(X_full)[:, 1], 4)
merged["renewal_risk_tier"] = pd.cut(
    merged["renewal_probability"], bins=[-1, 0.4, 0.6, 1.01],
    labels=["High Renewal Risk", "Uncertain", "Likely to Renew"]
)

joblib.dump(renewal_model, "models/renewal_model.joblib")

# ---------------------------------------------------------------------
# 2. RESIDENT FEEDBACK NLP
# ---------------------------------------------------------------------
X_text = leases["feedback_text"]
y_theme = leases["feedback_theme"]
y_sentiment = leases["feedback_sentiment"]

Xt_train, Xt_test, ytheme_train, ytheme_test, ysent_train, ysent_test = train_test_split(
    X_text, y_theme, y_sentiment, test_size=0.25, random_state=42, stratify=y_theme
)

vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english")
Xt_train_vec = vectorizer.fit_transform(Xt_train)
Xt_test_vec = vectorizer.transform(Xt_test)

theme_model = LogisticRegression(max_iter=1000)
theme_model.fit(Xt_train_vec, ytheme_train)
theme_preds = theme_model.predict(Xt_test_vec)
theme_accuracy = accuracy_score(ytheme_test, theme_preds)

sentiment_model = LogisticRegression(max_iter=1000)
sentiment_model.fit(Xt_train_vec, ysent_train)
sentiment_preds = sentiment_model.predict(Xt_test_vec)
sentiment_accuracy = accuracy_score(ysent_test, sentiment_preds)

nlp_metrics = {
    "theme_classification_accuracy": round(theme_accuracy, 4),
    "sentiment_classification_accuracy": round(sentiment_accuracy, 4),
    "n_test": len(Xt_test),
    "limitation": "Feedback text is template-generated, not real free-text resident "
                   "comments. This makes classification easier than a real-world NLP "
                   "task -- these accuracy figures should NOT be read as representative "
                   "of performance on genuine, unstructured resident feedback.",
}
print("NLP metrics:", json.dumps(nlp_metrics, indent=2))
with open("reports/nlp_model_metrics.json", "w") as f:
    json.dump(nlp_metrics, f, indent=2)

joblib.dump(vectorizer, "models/feedback_vectorizer.joblib")
joblib.dump(theme_model, "models/feedback_theme_model.joblib")
joblib.dump(sentiment_model, "models/feedback_sentiment_model.joblib")

# ---------------------------------------------------------------------
# 3. PROPERTY SEGMENTATION (5 categories per project brief)
# ---------------------------------------------------------------------
SEG_FEATURES = ["occupancy_pct", "noi_margin", "same_property_noi_growth_pct",
                 "maintenance_avg_response_hours", "capex_per_unit_ytd"]
X_seg = StandardScaler().fit_transform(properties[SEG_FEATURES])

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
properties["cluster"] = kmeans.fit_predict(X_seg)

centers = pd.DataFrame(kmeans.cluster_centers_, columns=SEG_FEATURES)
centers_raw = properties.groupby(properties["cluster"])[SEG_FEATURES].mean()  # unstandardized, for readable rules

# Sequential rule-based assignment -- deliberately NOT a single blind
# composite rank, because capex has a genuinely two-sided meaning (heavy
# capex can mean "already turning the asset around" OR "underperforming
# despite investment" depending on the results it's producing). Each rule
# claims the clearest-fitting remaining cluster before falling through to
# the next, rather than forcing a 1-D ranking onto a 2-D reality.
remaining = list(centers_raw.index)
cluster_label_map = {}

# 1. Repositioning Candidate: worst occupancy + negative growth --
#    fundamentally underperforming regardless of capital already spent.
worst_growth = centers_raw.loc[remaining, "same_property_noi_growth_pct"].idxmin()
cluster_label_map[worst_growth] = "Repositioning Candidate"
remaining.remove(worst_growth)

# 2. Renovation Opportunity: heaviest capex currently being deployed
#    among what's left -- capital actively going in, results developing.
heaviest_capex = centers_raw.loc[remaining, "capex_per_unit_ytd"].idxmax()
cluster_label_map[heaviest_capex] = "Renovation Opportunity"
remaining.remove(heaviest_capex)

# 3. Stable Performer: best occupancy + margin among what's left --
#    "stable" means already performing well on the core two metrics, not
#    necessarily fastest-growing (a cluster can be excellent and simply
#    near its natural ceiling, which is a different story than growth).
perf_score = (
    centers_raw.loc[remaining, "occupancy_pct"].rank()
    + centers_raw.loc[remaining, "noi_margin"].rank()
)
best_stable = perf_score.idxmax()
cluster_label_map[best_stable] = "Stable Performer"
remaining.remove(best_stable)

# 4. Disposition-Review Candidate: weakest NOI margin among what's left --
#    a consistent underperformer not receiving capital priority.
weakest_margin = centers_raw.loc[remaining, "noi_margin"].idxmin()
cluster_label_map[weakest_margin] = "Disposition-Review Candidate"
remaining.remove(weakest_margin)

# 5. Whatever's left: solid results achieved with only moderate capital --
#    the "improving through operations, not yet through renovation" case.
for cid in remaining:
    cluster_label_map[cid] = "Operating-Improvement Candidate"

properties["segment"] = properties["cluster"].map(cluster_label_map)

print("\nSegment sizes:")
print(properties["segment"].value_counts())
print("\nSegment characteristics:")
print(properties.groupby("segment")[SEG_FEATURES].mean().round(2))

joblib.dump(kmeans, "models/property_segmentation_kmeans.joblib")

# ---------------------------------------------------------------------
# Save scored outputs
# ---------------------------------------------------------------------
merged.to_csv("data/scored_leases.csv", index=False)
properties.to_csv("data/scored_properties.csv", index=False)
print("\nDone. Scored data -> data/scored_leases.csv, data/scored_properties.csv")
