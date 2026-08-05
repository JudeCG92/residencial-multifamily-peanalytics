"""
app.py -- ResidenceAlpha: Multifamily Asset Performance & Renovation Strategy

Four views:
  1. Portfolio Overview     - fund-level operating performance and segmentation
  2. Property Explorer      - single-property detail, segment, lease renewal risk
  3. Lease Explorer         - single-lease renewal prediction with SHAP explanation
  4. Renovation ROI Calculator - live sensitivity on capex, rent premium, occupancy lift
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap

st.set_page_config(page_title="ResidenceAlpha", layout="wide", page_icon="\U0001F3E0")

BASE_DIR = Path(__file__).resolve().parent.parent


@st.cache_resource
def load_models():
    renewal_model = joblib.load(BASE_DIR / "models" / "renewal_model.joblib")
    with open(BASE_DIR / "models" / "renewal_feature_columns.json") as f:
        renewal_features = json.load(f)
    return renewal_model, renewal_features


@st.cache_data
def load_data():
    leases = pd.read_csv(BASE_DIR / "data" / "scored_leases.csv")
    properties = pd.read_csv(BASE_DIR / "data" / "scored_properties.csv")
    return leases, properties


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


renewal_model, RENEWAL_FEATURES = load_models()
leases, properties = load_data()
explainer = load_explainer(renewal_model)

NUMERIC_FEATURES = ["rent_increase_offered_pct", "maintenance_avg_response_days",
                     "maintenance_requests_count", "tenure_months", "concession_months",
                     "occupancy_pct", "same_property_noi_growth_pct"]
CAT_FEATURES = ["property_class", "unit_type"]


def build_renewal_features(frame: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(frame[NUMERIC_FEATURES + CAT_FEATURES], columns=CAT_FEATURES, drop_first=True)
    for col in RENEWAL_FEATURES:
        if col not in X.columns:
            X[col] = 0
    return X[RENEWAL_FEATURES]


SEGMENT_ORDER = ["Stable Performer", "Operating-Improvement Candidate",
                  "Renovation Opportunity", "Repositioning Candidate",
                  "Disposition-Review Candidate"]
SEGMENT_COLORS = {"Stable Performer": "#2ca02c", "Operating-Improvement Candidate": "#3a6fb0",
                   "Renovation Opportunity": "#f5c542", "Repositioning Candidate": "#ff7f0e",
                   "Disposition-Review Candidate": "#d62728"}

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.title("\U0001F3E0 ResidenceAlpha")
st.sidebar.caption("Multifamily Asset Performance & Renovation Strategy")
page = st.sidebar.radio("View", ["Portfolio Overview", "Property Explorer", "Lease Explorer", "Renovation ROI Calculator"])
st.sidebar.divider()
st.sidebar.caption(
    "All property and lease data is synthetic, generated to match realistic "
    "multifamily operating patterns. Resident feedback comments are "
    "template-generated for NLP demonstration purposes, not real resident "
    "quotes. No real property or resident data is used."
)

# ---------------------------------------------------------------------
# PAGE 1 -- Portfolio Overview
# ---------------------------------------------------------------------
if page == "Portfolio Overview":
    st.title("Portfolio Overview")

    total_units = properties["total_units"].sum()
    w_occ = np.average(properties["occupancy_pct"], weights=properties["total_units"])
    w_noi_margin = np.average(properties["noi_margin"], weights=properties["total_units"])
    total_valuation = properties["valuation"].sum()
    avg_renewal_prob = leases["renewal_probability"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Units", f"{total_units:,}")
    c2.metric("Weighted Occupancy", f"{w_occ:.1f}%")
    c3.metric("Weighted NOI Margin", f"{w_noi_margin:.1%}")
    c4.metric("Portfolio Valuation", f"${total_valuation/1e6:,.0f}M")
    c5.metric("Avg Renewal Probability", f"{avg_renewal_prob:.1%}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Portfolio Segments")
        seg_counts = properties["segment"].value_counts().reindex(SEGMENT_ORDER).fillna(0)
        fig = px.bar(seg_counts, x=seg_counts.index, y=seg_counts.values,
                     color=seg_counts.index, color_discrete_map=SEGMENT_COLORS,
                     labels={"x": "Segment", "y": "Number of Properties"})
        fig.update_layout(showlegend=False, xaxis_tickangle=-20)
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.subheader("Lease Renewal Risk Distribution")
        risk_counts = leases["renewal_risk_tier"].value_counts().reindex(
            ["High Renewal Risk", "Uncertain", "Likely to Renew"]).fillna(0)
        fig = px.pie(values=risk_counts.values, names=risk_counts.index, hole=0.45,
                     color=risk_counts.index,
                     color_discrete_map={"High Renewal Risk": "#d62728", "Uncertain": "#f5c542", "Likely to Renew": "#2ca02c"})
        st.plotly_chart(fig, width="stretch")

    st.subheader("Occupancy vs. NOI Margin by Property, colored by Segment")
    fig = px.scatter(properties, x="occupancy_pct", y="noi_margin", color="segment",
                      size="total_units", hover_data=["property_id", "market"],
                      color_discrete_map=SEGMENT_COLORS,
                      labels={"occupancy_pct": "Occupancy (%)", "noi_margin": "NOI Margin"})
    st.plotly_chart(fig, width="stretch")

    st.subheader("Resident Feedback Themes")
    theme_counts = leases["feedback_theme"].value_counts()
    fig = px.bar(theme_counts, x=theme_counts.values, y=theme_counts.index, orientation="h",
                 labels={"x": "Number of Comments", "y": ""})
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------
# PAGE 2 -- Property Explorer
# ---------------------------------------------------------------------
elif page == "Property Explorer":
    st.title("Property Explorer")

    prop_id = st.selectbox("Select a property", properties["property_id"].sort_values())
    prop = properties[properties["property_id"] == prop_id].iloc[0]
    prop_leases = leases[leases["property_id"] == prop_id]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segment", prop["segment"])
    c2.metric("Occupancy", f"{prop['occupancy_pct']:.1f}%")
    c3.metric("NOI Margin", f"{prop['noi_margin']:.1%}")
    c4.metric("Avg Renewal Probability", f"{prop_leases['renewal_probability'].mean():.1%}")

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Property Detail")
        detail = {
            "Market": prop["market"], "Class": prop["property_class"],
            "Year Built": str(int(prop["year_built"])), "Total Units": str(int(prop["total_units"])),
            "Effective Rent (PSF)": f"${prop['effective_rent_psf']:.2f}",
            "Loss to Lease": f"{prop['loss_to_lease_pct']:.1f}%",
            "Concession": f"{prop['concession_pct']:.1f}%",
            "Same-Property NOI Growth": f"{prop['same_property_noi_growth_pct']:.1f}%",
            "Maintenance Avg Response": f"{prop['maintenance_avg_response_hours']:.0f} hrs",
            "Renovation Status": prop["renovation_status"],
            "Capex/Unit YTD": f"${prop['capex_per_unit_ytd']:,.0f}",
            "Cap Rate": f"{prop['cap_rate']:.2%}",
            "Valuation": f"${prop['valuation']:,.0f}",
        }
        st.table(pd.DataFrame(detail.items(), columns=["Field", "Value"]).set_index("Field"))
    with right:
        st.subheader("Feedback Theme Breakdown (this property)")
        theme_counts = prop_leases["feedback_theme"].value_counts()
        if len(theme_counts) > 0:
            fig = px.bar(theme_counts, x=theme_counts.values, y=theme_counts.index, orientation="h",
                         labels={"x": "Comments", "y": ""})
            st.plotly_chart(fig, width="stretch")
        st.subheader("Sentiment mix")
        sent_counts = prop_leases["feedback_sentiment"].value_counts()
        st.write(f"Positive: {sent_counts.get('positive', 0)}  |  Negative: {sent_counts.get('negative', 0)}")

    st.divider()
    st.subheader("Leases at this property, by renewal risk")
    display_cols = ["lease_id", "unit_type", "tenure_months", "rent_increase_offered_pct",
                     "maintenance_requests_count", "renewal_probability", "renewal_risk_tier"]
    st.dataframe(
        prop_leases[display_cols].sort_values("renewal_probability").reset_index(drop=True),
        width="stretch"
    )

# ---------------------------------------------------------------------
# PAGE 3 -- Lease Explorer
# ---------------------------------------------------------------------
elif page == "Lease Explorer":
    st.title("Lease Explorer")

    lease_id = st.selectbox("Select a lease", leases["lease_id"].sort_values())
    lease = leases[leases["lease_id"] == lease_id].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Renewal Probability", f"{lease['renewal_probability']:.1%}")
    c2.metric("Risk Tier", lease["renewal_risk_tier"])
    c3.metric("Actual Outcome (synthetic)", "Renewed" if lease["renewed"] == 1 else "Did Not Renew")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Lease Detail")
        detail = {
            "Property": lease["property_id"], "Market": lease["market"],
            "Unit Type": lease["unit_type"], "Tenure": f"{lease['tenure_months']:.1f} months",
            "Current Rent": f"${lease['current_rent']:,.0f}",
            "Market Rent": f"${lease['market_rent']:,.0f}",
            "Rent Increase Offered": f"{lease['rent_increase_offered_pct']:.1f}%",
            "Concession Months": str(int(lease["concession_months"])),
            "Maintenance Requests": str(int(lease["maintenance_requests_count"])),
            "Maintenance Avg Response": f"{lease['maintenance_avg_response_days']:.1f} days",
        }
        st.table(pd.DataFrame(detail.items(), columns=["Field", "Value"]).set_index("Field"))
        st.subheader("Resident Feedback")
        st.write(f"**Theme:** {lease['feedback_theme']}  |  **Sentiment:** {lease['feedback_sentiment']}")
        st.info(lease["feedback_text"])

    with right:
        st.subheader("What's driving this renewal prediction")
        single = leases[leases["lease_id"] == lease_id]
        X_single = build_renewal_features(single)
        shap_vals = explainer.shap_values(X_single)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]  # legacy SHAP: list of per-class arrays
        elif shap_vals.ndim == 3:
            sv = shap_vals[0, :, 1]  # (n_samples, n_features, n_classes) -- take class 1 (renewed)
        else:
            sv = shap_vals[0]
        contrib = pd.Series(sv, index=RENEWAL_FEATURES).sort_values(key=abs, ascending=True).tail(8)
        fig = go.Figure(go.Bar(
            x=contrib.values, y=contrib.index, orientation="h",
            marker_color=["#2ca02c" if v > 0 else "#d62728" for v in contrib.values],
        ))
        fig.update_layout(xaxis_title="Impact on renewal probability (SHAP value)", yaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption("Green = pushes toward renewal. Red = pushes toward non-renewal.")

# ---------------------------------------------------------------------
# PAGE 4 -- Renovation ROI Calculator
# ---------------------------------------------------------------------
else:
    st.title("Renovation ROI Calculator")
    st.caption("Transparent formula, not a black box: costs, incremental revenue, and payback are shown explicitly below.")

    prop_id = st.selectbox("Select a property to model a renovation for", properties["property_id"].sort_values())
    prop = properties[properties["property_id"] == prop_id].iloc[0]
    st.caption(f"Segment: **{prop['segment']}**  |  Current occupancy: {prop['occupancy_pct']:.1f}%  |  Current effective rent: ${prop['effective_rent_psf']:.2f}/sqft")

    c1, c2, c3, c4, c5 = st.columns(5)
    capex_per_unit = c1.number_input("Renovation cost per unit ($)", min_value=0, max_value=50000, value=8000, step=500)
    pct_units = c2.slider("% of units renovated", 0, 100, 100)
    rent_premium_pct = c3.slider("Expected rent premium (%)", 0, 30, 8)
    occupancy_lift_pts = c4.slider("Expected occupancy lift (pts)", 0, 10, 2)
    downtime_months = c5.slider("Downtime during renovation (months)", 0, 6, 1)

    units_renovated = prop["total_units"] * (pct_units / 100)
    total_renovation_cost = capex_per_unit * units_renovated

    downtime_lost_revenue = (
        prop["effective_rent_psf"] * prop["avg_unit_sqft"] * units_renovated * (downtime_months / 12)
    )

    new_effective_rent_psf = prop["effective_rent_psf"] * (1 + rent_premium_pct / 100)
    new_occupancy_pct = min(100, prop["occupancy_pct"] + occupancy_lift_pts)
    new_annual_revenue = new_effective_rent_psf * prop["avg_unit_sqft"] * prop["total_units"] * (new_occupancy_pct / 100) * 12
    incremental_annual_revenue = new_annual_revenue - prop["annual_revenue"]
    incremental_noi = incremental_annual_revenue * (1 - prop["opex_ratio"])

    all_in_cost = total_renovation_cost + downtime_lost_revenue
    payback_years = all_in_cost / incremental_noi if incremental_noi > 0 else np.nan
    value_created = incremental_noi / prop["cap_rate"] - all_in_cost
    roi_pct = (value_created / all_in_cost * 100) if all_in_cost > 0 else np.nan

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("All-in Cost", f"${all_in_cost:,.0f}")
    c2.metric("Incremental Annual NOI", f"${incremental_noi:,.0f}")
    c3.metric("Payback Period", f"{payback_years:.1f} yrs" if np.isfinite(payback_years) else "N/A")
    c4.metric("Value Created (at cap rate)", f"${value_created:,.0f}", delta=f"{roi_pct:.0f}% ROI" if np.isfinite(roi_pct) else None)

    st.subheader("How this is calculated")
    st.markdown(f"""
    1. **All-in cost** = renovation cost per unit \u00d7 units renovated + lost rent during downtime
       = `${capex_per_unit:,.0f} \u00d7 {units_renovated:.0f} + ${downtime_lost_revenue:,.0f}` = **${all_in_cost:,.0f}**
    2. **New annual revenue** = new rent (${new_effective_rent_psf:.2f}/sqft) \u00d7 avg unit sqft \u00d7 total units \u00d7 new occupancy ({new_occupancy_pct:.1f}%)
    3. **Incremental NOI** = incremental revenue \u00d7 (1 \u2212 opex ratio of {prop['opex_ratio']:.1%}) = **${incremental_noi:,.0f}/year**
    4. **Value created** = incremental NOI \u00f7 cap rate ({prop['cap_rate']:.2%}) \u2212 all-in cost
    """)
    st.caption(
        "Modeling simplification: assumes the current opex ratio holds after renovation "
        "(renovated units are not assumed to carry different operating costs), and that "
        "the full portfolio's cap rate applies to the incremental NOI. Real underwriting "
        "would validate both assumptions against comparable renovated assets."
    )

    st.divider()
    st.subheader("Portfolio-wide: which properties would benefit most from this same scenario?")
    scan = properties.copy()
    scan_units = scan["total_units"] * (pct_units / 100)
    scan_cost = capex_per_unit * scan_units + (scan["effective_rent_psf"] * scan["avg_unit_sqft"] * scan_units * (downtime_months / 12))
    scan_new_rent = scan["effective_rent_psf"] * (1 + rent_premium_pct / 100)
    scan_new_occ = np.minimum(100, scan["occupancy_pct"] + occupancy_lift_pts)
    scan_new_rev = scan_new_rent * scan["avg_unit_sqft"] * scan["total_units"] * (scan_new_occ / 100) * 12
    scan_incr_noi = (scan_new_rev - scan["annual_revenue"]) * (1 - scan["opex_ratio"])
    scan["roi_pct"] = np.where(scan_cost > 0, (scan_incr_noi / scan["cap_rate"] - scan_cost) / scan_cost * 100, np.nan)
    st.dataframe(
        scan[["property_id", "segment", "occupancy_pct", "roi_pct"]].sort_values("roi_pct", ascending=False).head(10),
        width="stretch"
    )
