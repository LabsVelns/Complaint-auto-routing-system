"""
data_generator.py
-----------------
Generates a realistic synthetic complaint dataset for training.
Each complaint has: text, officer, priority, eta_days

Officers are domain-specific — the model must learn to route
purely from complaint content (no department field in input).
"""

import pandas as pd
import numpy as np
import random
import json
import os

random.seed(42)
np.random.seed(42)

# ─── Officer roster ────────────────────────────────────────────────
OFFICERS = [
    {"id": "OFF001", "name": "Rajesh Kumar",    "domain": "infrastructure"},
    {"id": "OFF002", "name": "Priya Sharma",    "domain": "utilities"},
    {"id": "OFF003", "name": "Amit Verma",      "domain": "sanitation"},
    {"id": "OFF004", "name": "Sunita Patel",    "domain": "water_supply"},
    {"id": "OFF005", "name": "Deepak Singh",    "domain": "electricity"},
    {"id": "OFF006", "name": "Kavita Nair",     "domain": "healthcare"},
    {"id": "OFF007", "name": "Ravi Menon",      "domain": "transport"},
    {"id": "OFF008", "name": "Anita Gupta",     "domain": "noise_pollution"},
    {"id": "OFF009", "name": "Suresh Reddy",    "domain": "taxation"},
    {"id": "OFF010", "name": "Meena Joshi",     "domain": "housing"},
]

# Save officer roster for inference use
OFFICER_ROSTER = {o["id"]: o for o in OFFICERS}

# ─── Complaint templates per domain ────────────────────────────────
COMPLAINT_TEMPLATES = {

    "infrastructure": {
        "texts": [
            "The road on {street} has large potholes that have been causing accidents for {days} days. Several vehicles have been damaged.",
            "The bridge near {landmark} is showing cracks and looks structurally unsafe. Immediate inspection required.",
            "Street lights on {street} have not been working since {days} days causing safety issues at night.",
            "The footpath on {street} is completely broken and dangerous for pedestrians especially elderly people.",
            "A large pothole on {street} near {landmark} caused my vehicle to break down. This has been reported {count} times before.",
            "The road divider on {street} was damaged in an accident last week and has not been repaired creating traffic hazards.",
            "Flyover construction debris has not been cleared from {street} for {days} days causing major traffic issues.",
            "The public park benches at {landmark} are broken and unsafe for children and senior citizens.",
            "Speed breakers on {street} have been damaged and are not visible causing accidents at night.",
            "The drainage system on {street} collapsed causing road cave-in and major commute disruption.",
        ],
        "priority_weights": {"High": 0.4, "Medium": 0.4, "Low": 0.2},
        "eta_range": (3, 30),
    },

    "utilities": {
        "texts": [
            "Gas pipeline leak detected near {street}. Strong smell of gas in the area since {days} hours.",
            "Internet and cable services have been disrupted in {area} for {days} days affecting work from home.",
            "The public telephone booth at {landmark} is completely damaged and non-functional.",
            "Underground utility cables are exposed on {street} posing serious electrocution risk.",
            "The transformer on {street} is producing loud sparking sounds since last night. Very dangerous.",
            "Multiple utility poles on {street} are leaning and may fall during the next storm.",
            "The municipal wifi hotspot at {landmark} has not been working for {days} days.",
            "Street vendor electricity connections on {street} are illegally tapped causing overload issues.",
        ],
        "priority_weights": {"High": 0.5, "Medium": 0.35, "Low": 0.15},
        "eta_range": (1, 15),
    },

    "sanitation": {
        "texts": [
            "Garbage has not been collected from {area} for {days} days. The smell is unbearable and health hazard.",
            "Open defecation is happening near {landmark} due to lack of public toilets. Urgent action needed.",
            "The public toilet at {landmark} has been locked for {days} days leaving no facility for residents.",
            "Dead animals on {street} have not been removed for {days} days causing disease outbreak risk.",
            "Illegal dumping of construction waste at {landmark} is blocking the road and causing stench.",
            "Stray dogs are feeding on garbage dumped near {street} creating health and safety risks.",
            "The drain near {street} is completely clogged with garbage causing flooding during rains.",
            "Mosquito breeding in stagnant water near {landmark} due to poor drainage in {area}.",
            "The sanitation workers in {area} have not come for {days} days after the festival. Garbage piling up.",
            "Broken dustbins on {street} are causing litter to spread across the road and nearby shops.",
        ],
        "priority_weights": {"High": 0.35, "Medium": 0.45, "Low": 0.2},
        "eta_range": (1, 10),
    },

    "water_supply": {
        "texts": [
            "No water supply in {area} for {days} days. Residents are buying expensive tankers just to survive.",
            "Water coming from taps in {area} is brownish and smells bad. Definitely contaminated.",
            "The water pipeline on {street} is leaking heavily causing waterlogging and water wastage.",
            "Low water pressure in {area} since {days} days. Water does not reach above first floor.",
            "The water tank at {landmark} has overflowed for {days} consecutive days wasting thousands of litres.",
            "Sewage water is mixing with drinking water pipeline near {street}. Very serious health risk.",
            "Water supply timing in {area} has changed without notice causing major inconvenience.",
            "A burst pipe on {street} has been flooding the road for {days} days. No repair work started.",
            "The borewell at {landmark} has dried up leaving the community with no water source.",
            "Water meter readings in {area} are incorrect. Getting bills for double the actual usage.",
        ],
        "priority_weights": {"High": 0.5, "Medium": 0.35, "Low": 0.15},
        "eta_range": (1, 14),
    },

    "electricity": {
        "texts": [
            "Power outage in {area} for {days} days. Medicines in refrigerators are spoiling. Elderly patients suffering.",
            "High voltage fluctuation on {street} damaged our appliances. Need compensation and immediate fix.",
            "Electricity bill for {area} is three times higher than usual without any increase in usage.",
            "Dangling live electric wire on {street} near {landmark} is a life-threatening hazard.",
            "The transformer in {area} exploded last night. No power since then. No response from electricity board.",
            "Illegal electricity connections tapped from main line on {street} causing frequent outages.",
            "The electric meter box outside {landmark} is open and exposed to rain. Very dangerous.",
            "Streetlights in {area} are on during daytime wasting electricity for {days} days.",
            "Repeated tripping of main supply in {area} causing power cuts {count} times a day.",
            "The substation on {street} is making loud humming noise and smells of burning for {days} days.",
        ],
        "priority_weights": {"High": 0.5, "Medium": 0.3, "Low": 0.2},
        "eta_range": (1, 10),
    },

    "healthcare": {
        "texts": [
            "The government hospital at {landmark} does not have doctors available for {days} days. Patients suffering.",
            "Expired medicines are being distributed at the health centre near {area}. Serious negligence.",
            "Dengue outbreak in {area} - {count} cases confirmed. No fumigation or awareness drive started.",
            "The ambulance service in {area} took {days} hours to respond causing a patient's death.",
            "No sanitation in government hospital at {landmark}. Toilets not cleaned, wards have foul smell.",
            "Medical oxygen cylinders at {landmark} hospital are empty. ICU patients at risk.",
            "Fake medical practitioners operating clinic on {street} without valid license.",
            "The vaccination camp scheduled at {landmark} was cancelled without notice for {count} weeks.",
            "Unhygienic food being sold near {landmark} hospital. Multiple food poisoning cases reported.",
            "Doctor at {landmark} clinic demanding extra cash for treatments covered under government scheme.",
        ],
        "priority_weights": {"High": 0.6, "Medium": 0.3, "Low": 0.1},
        "eta_range": (1, 7),
    },

    "transport": {
        "texts": [
            "Bus route {count} has been cancelled for {days} days. Thousands of commuters stranded daily.",
            "Auto rickshaw drivers in {area} are refusing to go by meter and overcharging passengers.",
            "The railway station at {landmark} has no proper lighting in the parking area causing theft.",
            "Bus driver on route {count} was drunk and driving rashly. Passengers feared for their lives.",
            "No bus shelter on {street} causing hardship for commuters in rain and extreme heat.",
            "Traffic signal at {street} junction has been non-functional for {days} days causing accidents.",
            "Illegal parking of commercial vehicles on {street} blocks traffic for hours every day.",
            "The pedestrian crossing near {landmark} school has no zebra markings. Children in danger.",
            "Potholes on the main highway near {landmark} causing frequent accidents and vehicle damage.",
            "E-rickshaw operators in {area} are overloading passengers dangerously beyond capacity.",
        ],
        "priority_weights": {"High": 0.35, "Medium": 0.45, "Low": 0.2},
        "eta_range": (2, 21),
    },

    "noise_pollution": {
        "texts": [
            "Construction work on {street} continues all night violating noise pollution norms. No sleep for {days} days.",
            "A factory near {landmark} is running heavy machinery at night causing unbearable noise pollution.",
            "Loudspeakers at {landmark} play music beyond permitted decibel levels daily disturbing residents.",
            "Firecrackers being burst in {area} every night for {days} days causing distress to elderly and children.",
            "Bar and restaurant on {street} plays loud music past midnight regularly violating rules.",
            "A temple near {area} uses loudspeakers at 4am daily. Residents have complained {count} times.",
            "Illegal DJ parties in open ground near {landmark} till 3am disturbing entire neighbourhood.",
            "Honking near {landmark} school zone is continuous despite no-horn boards. Students cannot concentrate.",
        ],
        "priority_weights": {"High": 0.2, "Medium": 0.5, "Low": 0.3},
        "eta_range": (2, 14),
    },

    "taxation": {
        "texts": [
            "Received property tax notice for {area} property with incorrect assessment value. Need correction.",
            "Tax deducted twice for {area} property this quarter. Requesting refund of excess amount.",
            "My GST registration application has been pending for {days} days without any response.",
            "Received income tax notice with wrong PAN number. My returns are filed but notice not withdrawn.",
            "Property tax amount for {area} is higher than neighbouring properties of same size without reason.",
            "Road tax was paid {days} days ago but vehicle registration portal shows as pending.",
            "Tax exemption for senior citizen in {area} was removed without notice or reason this year.",
            "New tax levied on {area} residents without proper public notification or justification.",
        ],
        "priority_weights": {"High": 0.2, "Medium": 0.5, "Low": 0.3},
        "eta_range": (7, 45),
    },

    "housing": {
        "texts": [
            "Government housing allotted to someone else despite my name on the official list for {area}.",
            "Construction of building on {street} is illegal as it violates floor space index regulations.",
            "The housing society at {landmark} is denying flat possession despite full payment {days} days ago.",
            "Slum demolition in {area} was done without proper notice or rehabilitation arrangements for {count} families.",
            "My house in {area} was flooded due to nearby illegal construction blocking the natural drain.",
            "Building near {landmark} is under construction without valid permits blocking emergency exit roads.",
            "Common areas in government housing at {landmark} are being illegally encroached upon.",
            "Rental agreement disputes in {area} — landlord has locked the property illegally during tenancy.",
            "Promised amenities in housing scheme at {landmark} were not delivered even after {days} days of possession.",
            "Encroachment on public land near {street} has reduced the road width to dangerous levels.",
        ],
        "priority_weights": {"High": 0.3, "Medium": 0.4, "Low": 0.3},
        "eta_range": (5, 60),
    },
}

# ─── Filler data ────────────────────────────────────────────────────
STREETS   = ["MG Road","Gandhi Nagar","Nehru Street","Rajiv Chowk","Shastri Marg","Civil Lines","Sector 14","Lal Bagh Road","Brigade Road","Anna Salai"]
LANDMARKS = ["City Hospital","Central Park","Old Bus Stand","Railway Station","Town Hall","District Court","Main Market","Municipal Office","Government School","Post Office"]
AREAS     = ["Sector 4","Ward 7","Krishna Nagar","Gomti Nagar","Banjara Hills","Koramangala","Salt Lake","Andheri West","Powai","Whitefield"]

def fill_template(text):
    return text.format(
        street   = random.choice(STREETS),
        landmark = random.choice(LANDMARKS),
        area     = random.choice(AREAS),
        days     = random.randint(1, 30),
        count    = random.randint(2, 20),
    )

def assign_priority(weights):
    choices = list(weights.keys())
    probs   = list(weights.values())
    return random.choices(choices, weights=probs, k=1)[0]

def assign_eta(eta_range, priority):
    lo, hi = eta_range
    base   = random.randint(lo, hi)
    # High priority → shorter ETA
    if priority == "High":
        return max(1, int(base * 0.6))
    elif priority == "Low":
        return int(base * 1.4)
    return base

def generate_dataset(n_per_domain=120):
    records = []
    domains = list(COMPLAINT_TEMPLATES.keys())

    # Map domains to officer IDs
    domain_to_officer = {o["domain"]: o["id"] for o in OFFICERS}

    for domain, cfg in COMPLAINT_TEMPLATES.items():
        officer_id = domain_to_officer[domain]
        templates  = cfg["texts"]

        for i in range(n_per_domain):
            template = random.choice(templates)
            text     = fill_template(template)
            priority = assign_priority(cfg["priority_weights"])
            eta      = assign_eta(cfg["eta_range"], priority)

            # Add slight text augmentation variety
            augmentations = [
                text,
                text + " Please take urgent action.",
                text + " This is not the first time this has happened.",
                text + f" We have been suffering for a long time.",
                "URGENT: " + text if priority == "High" else text,
                text + " Kindly resolve at the earliest.",
            ]
            final_text = random.choice(augmentations)

            records.append({
                "complaint_text": final_text,
                "domain":         domain,
                "officer_id":     officer_id,
                "priority":       priority,
                "eta_days":       eta,
            })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def generate_historical_complaints(n=500):
    """
    Generate a separate pool of 'historical' complaints
    used only for similarity search (FAISS index).
    """
    records = []
    domains = list(COMPLAINT_TEMPLATES.keys())
    domain_to_officer = {o["domain"]: o["id"] for o in OFFICERS}

    for _ in range(n):
        domain   = random.choice(domains)
        cfg      = COMPLAINT_TEMPLATES[domain]
        template = random.choice(cfg["texts"])
        text     = fill_template(template)
        priority = assign_priority(cfg["priority_weights"])
        eta      = assign_eta(cfg["eta_range"], priority)
        records.append({
            "complaint_id":   f"HIST{len(records)+1:04d}",
            "complaint_text": text,
            "domain":         domain,
            "officer_id":     domain_to_officer[domain],
            "priority":       priority,
            "eta_days":       eta,
            "status":         random.choice(["Resolved", "Resolved", "Closed"]),
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("Generating training dataset...")
    df_train = generate_dataset(n_per_domain=120)
    df_train.to_csv("data/complaints_train.csv", index=False)
    print(f"  Training set: {len(df_train)} records")
    print(f"  Domain distribution:\n{df_train['domain'].value_counts()}")
    print(f"  Priority distribution:\n{df_train['priority'].value_counts()}")

    print("\nGenerating historical complaints for similarity search...")
    df_hist = generate_historical_complaints(n=500)
    df_hist.to_csv("data/historical_complaints.csv", index=False)
    print(f"  Historical set: {len(df_hist)} records")

    print("\nSaving officer roster...")
    with open("data/officers.json", "w") as f:
        json.dump(OFFICER_ROSTER, f, indent=2)

    print("\nDone. Files saved to data/")
