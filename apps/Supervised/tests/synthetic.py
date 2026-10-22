"""
Synthetische Testdaten – nur für Tests und zum Ausprobieren, keine echten Daten.

    python tests/synthetic.py                 # schreibt demo_lending_club_synthetisch.csv (20 000 Zeilen)
    python tests/synthetic.py --rows 5000

lending_club_like(): gleiche Spalten/Formate wie der öffentliche LendingClub-Datensatz (lending_club_loan_two.csv)
house_like():        Regression (Hauspreise, ähnlich kc_house_data)
iris_like():         Mehrklassen (Iris aus scikit-learn)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SUB_GRADES = [g + str(i) for g in "ABCDEFG" for i in range(1, 6)]
ZIPS = ["22690", "05113", "00813", "11650", "30723", "70466", "29597", "48052", "86630", "93700"]
ZIP_EFFECT = {"11650": -3.5, "86630": -3.5, "93700": -3.5, "05113": 2.5, "00813": 2.5, "29597": 2.5}
STATES = ["OK", "SD", "WV", "MA", "VA", "TX", "CA", "NY", "FL", "AZ"]
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
            "small_business", "car", "medical", "moving", "vacation", "house", "wedding",
            "renewable_energy", "educational"]
PURPOSE_P = np.array([59, 21, 6, 5.3, 2.2, 1.4, 1.2, 1.05, 0.72, 0.62, 0.55, 0.46, 0.08, 0.04])
EMP_LEN = ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years",
           "8 years", "9 years", "10+ years"]
JOBS = [f"{a} {b}" for a in ["Senior", "Junior", "Lead", "Chief", "Assistant", "Head", "Staff", "Regional",
                             "Associate", "Principal"]
        for b in ["Teacher", "Manager", "Nurse", "Engineer", "Driver", "Analyst", "Clerk", "Supervisor",
                  "Technician", "Consultant", "Accountant", "Director"]]


def lending_club_like(n: int = 5000, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sub_idx = np.clip(rng.normal(9, 6.5, n).round().astype(int), 0, 34)
    sub_grade = np.asarray(SUB_GRADES)[sub_idx]
    term = rng.choice([" 36 months", " 60 months"], n, p=[0.76, 0.24])
    months = np.where(term == " 36 months", 36, 60)
    loan_amnt = np.clip((rng.lognormal(9.4, 0.6, n) / 25).round() * 25, 500, 40000)
    int_rate = np.round(5.3 + sub_idx * 0.72 + rng.normal(0, 0.6, n), 2)
    r = int_rate / 1200
    installment = np.round(loan_amnt * r / (1 - (1 + r) ** -months), 2)
    annual_inc = np.round(rng.lognormal(11.1, 0.5, n), 0)
    dti = np.round(np.clip(rng.normal(17, 8, n), 0, 45), 2)
    open_acc = rng.poisson(11, n).astype(float)
    total_acc = open_acc + rng.poisson(14, n)
    mort_acc = np.clip(np.round(total_acc * 0.12 - 1 + rng.normal(0, 1.0, n)), 0, None)
    pub_rec = rng.choice([0, 0, 0, 0, 0, 0, 0, 1, 1, 2], n).astype(float)
    pub_rec_bankruptcies = np.where(pub_rec > 0, rng.choice([0, 1], n, p=[0.3, 0.7]), 0).astype(float)
    revol_bal = np.round(rng.lognormal(9.3, 0.9, n), 0)
    revol_util = np.round(np.clip(rng.normal(53, 24, n), 0, 150), 1)
    home = rng.choice(["MORTGAGE", "RENT", "OWN", "OTHER", "NONE", "ANY"], n,
                      p=[0.5, 0.4, 0.0945, 0.003, 0.0015, 0.001])
    verification = rng.choice(["Verified", "Source Verified", "Not Verified"], n, p=[0.35, 0.33, 0.32])
    purpose = rng.choice(PURPOSES, n, p=PURPOSE_P / PURPOSE_P.sum())
    title = np.array([p.replace("_", " ").title() + (f" {rng.integers(1, 400)}" if rng.random() < 0.4 else "")
                      for p in purpose], dtype=object)
    issue_d = [f"{MONTHS[m]}-{y}" for m, y in zip(rng.integers(0, 12, n), rng.integers(2007, 2017, n))]
    earliest = [f"{MONTHS[m]}-{y}" for m, y in zip(rng.integers(0, 12, n), rng.integers(1960, 2012, n))]
    zips = rng.choice(ZIPS, n)
    address = [f"{rng.integers(1, 99999)} Example Street {i}\nSampletown, {rng.choice(STATES)} {z}"
               for i, z in enumerate(zips)]
    emp_title = rng.choice(JOBS, n).astype(object)
    emp_length = rng.choice(EMP_LEN, n).astype(object)

    zip_eff = np.array([ZIP_EFFECT.get(z, 0.0) for z in zips])
    logit = (2.6 - 0.11 * sub_idx - 0.5 * (months == 60) - 0.02 * (dti - 17)
             + 0.25 * np.log(annual_inc / 60000) - 0.006 * (revol_util - 53) + zip_eff
             + rng.normal(0, 0.5, n))
    repaid = rng.random(n) < 1 / (1 + np.exp(-logit))
    loan_status = np.where(repaid, "Fully Paid", "Charged Off")

    def holes(arr, share, minimum=2):
        arr = arr.astype(object) if arr.dtype.kind in "OU" else arr.astype(float)
        k = max(minimum, int(n * share))
        arr[rng.choice(n, k, replace=False)] = np.nan
        return arr

    df = pd.DataFrame({
        "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate, "installment": installment,
        "grade": [s[0] for s in sub_grade], "sub_grade": sub_grade,
        "emp_title": holes(emp_title, 0.058), "emp_length": holes(emp_length, 0.046),
        "home_ownership": home, "annual_inc": annual_inc, "verification_status": verification,
        "issue_d": issue_d, "loan_status": loan_status, "purpose": purpose, "title": holes(title, 0.0044),
        "dti": dti, "earliest_cr_line": earliest, "open_acc": open_acc, "pub_rec": pub_rec,
        "revol_bal": revol_bal, "revol_util": holes(revol_util, 0.0007),
        "total_acc": total_acc.astype(float), "initial_list_status": rng.choice(["f", "w"], n, p=[0.6, 0.4]),
        "application_type": rng.choice(["INDIVIDUAL", "JOINT", "DIRECT_PAY"], n, p=[0.996, 0.002, 0.002]),
        "mort_acc": holes(mort_acc, 0.095), "pub_rec_bankruptcies": holes(pub_rec_bankruptcies, 0.0013),
        "address": address,
    })
    return df


def house_like(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sqft = np.clip(rng.normal(2000, 700, n), 400, None).round()
    grade = np.clip(rng.normal(7.6, 1.2, n).round(), 3, 13)
    waterfront = (rng.random(n) < 0.01).astype(int)
    view = rng.choice([0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4], n)
    zipcode = rng.choice(np.arange(98001, 98071), n)
    yr_built = rng.integers(1900, 2016, n)
    price = (50_000 + 180 * sqft + 25_000 * (grade - 7) + 400_000 * waterfront + 30_000 * view
             + 800 * (zipcode % 13) - 300 * (2015 - yr_built) + rng.normal(0, 40_000, n))
    dates = [f"20{y}{m:02d}{d:02d}T000000" for y, m, d in
             zip(rng.integers(14, 16, n), rng.integers(1, 13, n), rng.integers(1, 29, n))]
    bathrooms = np.round(rng.uniform(1, 4.5, n) * 4) / 4
    bathrooms[rng.choice(n, 40, replace=False)] = np.nan
    return pd.DataFrame({
        "id": rng.permutation(np.arange(1_000_000, 1_000_000 + n)), "date": dates, "price": price.round(),
        "bedrooms": rng.integers(1, 7, n), "bathrooms": bathrooms, "sqft_living": sqft,
        "floors": rng.choice([1, 1.5, 2, 3], n), "waterfront": waterfront, "view": view,
        "grade": grade, "yr_built": yr_built, "zipcode": zipcode,
        "lat": np.round(47.5 + rng.normal(0, 0.14, n), 4), "long": np.round(-122.2 + rng.normal(0, 0.14, n), 4),
    })


def iris_like() -> pd.DataFrame:
    from sklearn.datasets import load_iris

    data = load_iris(as_frame=True)
    df = data.frame.rename(columns=lambda c: c.replace(" (cm)", "").replace(" ", "_"))
    df["species"] = np.asarray(data.target_names)[df.pop("target")]
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20_000)
    ap.add_argument("--out", default="demo_lending_club_synthetisch.csv")
    args = ap.parse_args()
    lending_club_like(args.rows).to_csv(args.out, index=False)
    print(f"{args.out} geschrieben ({args.rows} Zeilen, synthetisch)")
