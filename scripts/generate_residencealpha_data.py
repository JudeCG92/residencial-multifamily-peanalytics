"""
generate_residencealpha_data.py

Creates two linked synthetic datasets for ResidenceAlpha:
  1. data/properties.csv -- 40 multifamily properties with portfolio-level
     operating metrics (occupancy, NOI, rent, capex, renovation status).
  2. data/leases.csv -- ~1,200 individual unit-leases nested inside those
     properties, each with resident-level detail, maintenance history, a
     synthetic resident feedback comment, and a renewal outcome.

Nothing here is real -- no real property, resident, or lease exists. This
mirrors the same design principle used in CRE Sentinel and RE ValueWorks:
generate DATA THAT BEHAVES like a real multifamily portfolio, with genuine
randomness in the outcome that matters (lease renewal), so a model trained
on it has something real to learn rather than reverse-engineering a formula.

Resident feedback comments are template-generated from theme + sentiment
combinations, NOT written to impersonate real residents. They exist purely
to give the NLP layer realistic short-text classification material.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_PROPERTIES = 40
TODAY = pd.Timestamp("2026-01-01")

MARKETS = ["Atlanta", "Dallas", "Phoenix", "Charlotte", "Denver", "Austin", "Tampa", "Nashville"]
CLASSES = ["A", "B", "C"]
CLASS_WEIGHTS = [0.30, 0.45, 0.25]

UNIT_TYPES = ["Studio", "1BR", "2BR", "3BR"]
UNIT_TYPE_WEIGHTS = [0.12, 0.42, 0.36, 0.10]
UNIT_SQFT = {"Studio": 520, "1BR": 750, "2BR": 1080, "3BR": 1350}

FEEDBACK_THEMES = ["Maintenance", "Noise", "Cleanliness", "Amenities", "Safety",
                    "Management Responsiveness", "Lighting"]

# Template bank: (theme, sentiment) -> list of phrasings. Combined with light
# randomization so comments aren't identical, but themes stay classifiable.
FEEDBACK_TEMPLATES = {
    ("Maintenance", "negative"): [
        "Maintenance request took {n} days to get resolved.",
        "Still waiting on a repair I submitted weeks ago.",
        "Maintenance team is slow to respond to requests.",
    ],
    ("Maintenance", "positive"): [
        "Maintenance fixed my issue the same day, very responsive.",
        "Quick turnaround on my maintenance request.",
    ],
    ("Noise", "negative"): [
        "Noise from neighboring units is a constant issue.",
        "Thin walls make it hard to get any quiet.",
    ],
    ("Noise", "positive"): [
        "Building is quiet, never had a noise complaint.",
    ],
    ("Cleanliness", "negative"): [
        "Common areas are not kept clean.",
        "Hallways and trash areas need more attention.",
    ],
    ("Cleanliness", "positive"): [
        "Property is always clean and well-maintained.",
    ],
    ("Amenities", "negative"): [
        "Gym equipment has been broken for months.",
        "Pool area feels outdated and underused.",
    ],
    ("Amenities", "positive"): [
        "Love the amenities, especially the fitness center.",
        "Amenity spaces are a great value for the rent.",
    ],
    ("Safety", "negative"): [
        "Front gate has been broken, security is a concern.",
        "Parking area doesn't feel very secure at night.",
    ],
    ("Safety", "positive"): [
        "Always feel safe on the property, good lighting and gated access.",
    ],
    ("Management Responsiveness", "negative"): [
        "Hard to get a response from the leasing office.",
        "Management is difficult to reach when issues come up.",
    ],
    ("Management Responsiveness", "positive"): [
        "Management team is responsive and easy to work with.",
    ],
    ("Lighting", "negative"): [
        "Parking lot lighting is poor at night.",
        "Common area lighting could be improved.",
    ],
    ("Lighting", "positive"): [
        "Well-lit throughout, common areas and parking included.",
    ],
}


def generate_properties(n=N_PROPERTIES):
    market = np.random.choice(MARKETS, size=n)
    prop_class = np.random.choice(CLASSES, size=n, p=CLASS_WEIGHTS)
    year_built = np.random.randint(1985, 2022, size=n)

    total_units = np.clip(np.random.normal(220, 70, size=n), 60, 450).astype(int)

    class_rent_psf = {"A": 1.95, "B": 1.55, "C": 1.20}
    asking_rent_psf = np.array([
        np.clip(np.random.normal(class_rent_psf[c], 0.15), 0.8, 2.6) for c in prop_class
    ])

    class_occ = {"A": 93, "B": 92, "C": 89}
    occupancy_pct = np.clip(
        np.array([np.random.normal(class_occ[c], 4) for c in prop_class]), 65, 100
    )

    loss_to_lease_pct = np.clip(np.random.normal(4, 2, size=n), -2, 12)
    effective_rent_psf = asking_rent_psf * (1 - loss_to_lease_pct / 100)
    concession_pct = np.clip(np.random.normal(2.5, 2, size=n), 0, 10)

    avg_unit_sqft = np.random.normal(820, 90, size=n)
    revenue = effective_rent_psf * avg_unit_sqft * total_units * (occupancy_pct / 100) * 12
    class_opex_ratio = {"A": 0.42, "B": 0.46, "C": 0.50}
    opex_ratio = np.array([np.clip(np.random.normal(class_opex_ratio[c], 0.04), 0.3, 0.65) for c in prop_class])
    noi = revenue * (1 - opex_ratio)
    noi_margin = 1 - opex_ratio

    same_property_noi_growth_pct = np.clip(np.random.normal(3.0, 2.5, size=n), -6, 10)

    maintenance_avg_response_hours = np.clip(np.random.normal(30, 15, size=n), 4, 96)

    renovation_status = np.random.choice(
        ["None", "In Progress", "Completed (last 12mo)"], size=n, p=[0.55, 0.15, 0.30]
    )
    capex_per_unit_ytd = np.where(
        renovation_status == "None", np.random.normal(600, 300, size=n),
        np.random.normal(4500, 1800, size=n)
    ).clip(min=100)

    cap_rate = np.array([np.clip(np.random.normal({"A": 0.052, "B": 0.058, "C": 0.066}[c], 0.006), 0.04, 0.09) for c in prop_class])
    valuation = noi / cap_rate

    df = pd.DataFrame({
        "property_id": [f"PR{str(i+1).zfill(3)}" for i in range(n)],
        "market": market, "property_class": prop_class, "year_built": year_built,
        "total_units": total_units, "avg_unit_sqft": np.round(avg_unit_sqft, 0),
        "occupancy_pct": np.round(occupancy_pct, 1),
        "asking_rent_psf": np.round(asking_rent_psf, 2),
        "effective_rent_psf": np.round(effective_rent_psf, 2),
        "loss_to_lease_pct": np.round(loss_to_lease_pct, 2),
        "concession_pct": np.round(concession_pct, 2),
        "annual_revenue": np.round(revenue, 0),
        "opex_ratio": np.round(opex_ratio, 4),
        "noi": np.round(noi, 0), "noi_margin": np.round(noi_margin, 4),
        "same_property_noi_growth_pct": np.round(same_property_noi_growth_pct, 2),
        "maintenance_avg_response_hours": np.round(maintenance_avg_response_hours, 1),
        "renovation_status": renovation_status,
        "capex_per_unit_ytd": np.round(capex_per_unit_ytd, 0),
        "cap_rate": np.round(cap_rate, 4),
        "valuation": np.round(valuation, 0),
    })
    return df


def generate_leases(properties_df, leases_per_property_avg=30):
    rows = []
    lease_counter = 1
    for _, prop in properties_df.iterrows():
        n_leases = max(8, int(np.random.normal(leases_per_property_avg, 8)))
        unit_type = np.random.choice(UNIT_TYPES, size=n_leases, p=UNIT_TYPE_WEIGHTS)

        tenure_months = np.clip(np.random.exponential(14, size=n_leases), 1, 96)
        lease_start = [TODAY - timedelta(days=int(t * 30.4)) for t in tenure_months]

        base_rent_psf = prop["effective_rent_psf"] * np.random.normal(1.0, 0.06, size=n_leases)
        unit_sqft = np.array([UNIT_SQFT[ut] for ut in unit_type]) * np.random.normal(1.0, 0.05, size=n_leases)
        current_rent = base_rent_psf * unit_sqft

        market_rent = current_rent * np.random.normal(1.04, 0.05, size=n_leases)
        rent_increase_offered_pct = np.clip((market_rent - current_rent) / current_rent * 100, -5, 20)

        concession_months = np.random.choice([0, 0, 0, 1, 2], size=n_leases)

        maintenance_requests_count = np.random.poisson(prop["maintenance_avg_response_hours"] / 20, size=n_leases)
        maintenance_avg_response_days = np.clip(
            np.random.normal(prop["maintenance_avg_response_hours"] / 24, 1.0, size=n_leases), 0.2, 10
        )

        # ---- Renewal probability: genuine logistic model + noise ----
        logit = (
            1.1
            - rent_increase_offered_pct * 0.09
            - maintenance_avg_response_days * 0.12
            + np.log1p(tenure_months) * 0.18
            - (maintenance_requests_count > 3).astype(float) * 0.35
            + (concession_months > 0).astype(float) * 0.15
            + (prop["occupancy_pct"] - 90) * 0.02
            + np.random.normal(0, 0.6, size=n_leases)
        )
        renew_prob = 1 / (1 + np.exp(-logit))
        renewed = np.random.binomial(1, np.clip(renew_prob, 0.02, 0.97))

        # ---- Resident feedback: theme + sentiment sampled with a mild
        # correlation to maintenance/renewal experience, then templated ----
        themes = np.random.choice(FEEDBACK_THEMES, size=n_leases)
        sentiment_bias = np.clip(0.5 - maintenance_avg_response_days * 0.03 + (renewed * 0.15), 0.05, 0.9)
        sentiments = np.array([
            "positive" if np.random.random() < p else "negative" for p in sentiment_bias
        ])
        feedback_text = []
        for th, sent, mreq in zip(themes, sentiments, maintenance_avg_response_days):
            options = FEEDBACK_TEMPLATES[(th, sent)]
            template = np.random.choice(options)
            feedback_text.append(template.format(n=int(max(1, mreq))))

        for i in range(n_leases):
            rows.append({
                "lease_id": f"L{str(lease_counter).zfill(5)}",
                "property_id": prop["property_id"],
                "market": prop["market"], "property_class": prop["property_class"],
                "unit_type": unit_type[i],
                "unit_sqft": round(unit_sqft[i], 0),
                "tenure_months": round(tenure_months[i], 1),
                "lease_start_date": lease_start[i].date(),
                "current_rent": round(current_rent[i], 0),
                "market_rent": round(market_rent[i], 0),
                "rent_increase_offered_pct": round(rent_increase_offered_pct[i], 2),
                "concession_months": int(concession_months[i]),
                "maintenance_requests_count": int(maintenance_requests_count[i]),
                "maintenance_avg_response_days": round(maintenance_avg_response_days[i], 2),
                "feedback_theme": themes[i],
                "feedback_sentiment": sentiments[i],
                "feedback_text": feedback_text[i],
                "renewed": int(renewed[i]),
            })
            lease_counter += 1

    return pd.DataFrame(rows)


if __name__ == "__main__":
    properties = generate_properties()
    leases = generate_leases(properties)

    properties.to_csv("/home/claude/residence-alpha/data/properties.csv", index=False)
    leases.to_csv("/home/claude/residence-alpha/data/leases.csv", index=False)

    print(f"Properties: {len(properties)}  |  Leases: {len(leases)}")
    print(f"Overall renewal rate: {leases['renewed'].mean():.2%}")
    print(f"Occupancy range: {properties['occupancy_pct'].min():.1f}% - {properties['occupancy_pct'].max():.1f}%")
    print(properties.groupby("property_class")[["occupancy_pct", "noi_margin"]].mean().round(3))
