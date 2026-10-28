"""
Synthetische Testdaten – nur für Tests und zum Ausprobieren, keine echten Daten.

    python tests/synthetic.py                 # schreibt demo_college_synthetisch.csv (777 Zeilen, wie im Kurs)
    python tests/synthetic.py --rows 2000

college_like():      gleiche Spalten/Formate wie College_Data aus dem K-Means-Kurs (erste Spalte = Name, Private Yes/No)
mall_like():         Kundensegmente (5 Gruppen nach Einkommen und Ausgaben) mit Kategorien und Lücken
lending_club_like(): gleiche Spalten/Formate wie der öffentliche LendingClub-Datensatz – für die Tests von Schritt 2
iris_like():         Iris aus scikit-learn (3 bekannte Arten)
anomaly_like():      Transaktionen mit wenigen Ausreißern (Class = 1) – für Isolation Forest, LOF, DBSCAN
FakeBybit:           simulierte öffentliche Bybit-API v5 (Kerzen, Funding, Open Interest, Ticker) für
                     tools/bybit_features.py – mit einem eingebauten Kurssturz
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


COLLEGE_WORDS = ["Abbey", "Belmont", "Cedar", "Dover", "Elm", "Fairview", "Glen", "Harbor", "Iron", "Juniper", "Kings",
                 "Lake", "Maple", "North", "Oak", "Pine", "Quarry", "River", "Stone", "Valley", "West", "York"]


def college_like(n: int = 777, seed: int = 101, share_private: float = 0.727) -> pd.DataFrame:
    """Wie College_Data (Kurs): Index = Name der Hochschule, Private = Yes/No, 17 Zahlenspalten. Private Hochschulen
    sind kleiner, teurer (Outstate) und haben mehr Spenden ehemaliger Studierender – die Gruppen überlappen aber.
    Eine Hochschule hat Grad.Rate 118 (wie Cazenovia College im Kurs)."""
    rng = np.random.default_rng(seed)
    private = rng.random(n) < share_private
    size = np.where(private, rng.lognormal(7.2, 0.75, n), rng.lognormal(8.7, 0.75, n))
    f_under = np.round(size).astype(int) + 100
    apps = np.round(f_under * rng.lognormal(0.55, 0.35, n)).astype(int)
    accept = np.round(apps * np.clip(rng.normal(0.75, 0.13, n), 0.15, 1.0)).astype(int)
    enroll = np.round(accept * np.clip(rng.normal(0.4, 0.12, n), 0.1, 0.95)).astype(int)
    top10 = np.clip(np.round(np.where(private, rng.normal(29, 18, n), rng.normal(22, 14, n))), 1, 96).astype(int)
    top25 = np.clip(np.round(top10 + rng.normal(28, 9, n)), 9, 100).astype(int)
    p_under = np.round(np.where(private, rng.lognormal(5.2, 1.2, n), rng.lognormal(7.0, 1.1, n))).astype(int)
    outstate = np.round(np.where(private, rng.normal(11800, 3700, n), rng.normal(6800, 2300, n))).clip(2340, 21700)
    room = np.round(np.where(private, rng.normal(4590, 1100, n), rng.normal(3750, 850, n))).clip(1780, 8120)
    books = np.round(rng.normal(550, 165, n)).clip(96, 2340)
    personal = np.round(np.where(private, rng.normal(1210, 620, n), rng.normal(1680, 680, n))).clip(250, 6800)
    phd = np.clip(np.round(rng.normal(72, 16, n)), 8, 100).astype(int)
    terminal = np.clip(np.round(phd + rng.normal(7, 7, n)), 24, 100).astype(int)
    sf = np.round(np.where(private, rng.normal(12.9, 3.5, n), rng.normal(17.1, 3.4, n)), 1).clip(2.5, 39.8)
    alumni = np.clip(np.round(np.where(private, rng.normal(25.9, 12, n), rng.normal(14.4, 7.5, n))), 0, 64).astype(int)
    expend = np.round(np.where(private, rng.normal(10500, 5000, n), rng.normal(7450, 2700, n))).clip(3186, 56233)
    grad = np.clip(np.round(np.where(private, rng.normal(69, 17, n), rng.normal(56, 15, n))), 10, 100).astype(int)
    grad[int(rng.integers(n))] = 118  # unmöglicher Wert – im Kurs auf 100 gesetzt
    w = len(COLLEGE_WORDS)
    names = [f"{COLLEGE_WORDS[i % w]} {'College' if p else 'State University'}" + (f" {i // w + 1}" if i >= w else "")
             for i, p in enumerate(private)]
    df = pd.DataFrame({
        "Private": np.where(private, "Yes", "No"), "Apps": apps, "Accept": accept, "Enroll": enroll,
        "Top10perc": top10, "Top25perc": top25, "F.Undergrad": f_under, "P.Undergrad": p_under,
        "Outstate": outstate.astype(int), "Room.Board": room.astype(int), "Books": books.astype(int),
        "Personal": personal.astype(int), "PhD": phd, "Terminal": terminal, "S.F.Ratio": sf,
        "perc.alumni": alumni, "Expend": expend.astype(int), "Grad.Rate": grad,
    }, index=pd.Index(names))
    return df


def mall_like(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Kundensegmente wie der bekannte Mall-Customers-Datensatz: 5 Gruppen nach Jahreseinkommen und Ausgaben-Score,
    dazu Alter, Geschlecht, Region (Kategorie mit Lücken) und ein paar Lücken im Einkommen."""
    rng = np.random.default_rng(seed)
    centers = np.array([[25, 20], [25, 80], [55, 50], [88, 17], [88, 82]], dtype=float)
    segment = rng.choice(len(centers), n, p=[0.11, 0.11, 0.39, 0.19, 0.20])
    income = np.round(centers[segment, 0] + rng.normal(0, 7, n), 1).clip(12, 140)
    spending = np.round(centers[segment, 1] + rng.normal(0, 7, n)).clip(1, 99)
    age = np.round(np.where(spending > 60, rng.normal(30, 6, n), rng.normal(45, 11, n))).clip(18, 70)
    df = pd.DataFrame({
        "CustomerID": np.arange(1, n + 1),
        "Gender": rng.choice(["Female", "Male"], n, p=[0.56, 0.44]),
        "Age": age.astype(int),
        "Annual Income (k$)": income,
        "Spending Score (1-100)": spending.astype(int),
        "Region": rng.choice(["North", "South", "East", "West"], n).astype(object),
        "Segment": np.asarray(["A", "B", "C", "D", "E"])[segment],
    })
    df.loc[rng.choice(n, 12, replace=False), "Annual Income (k$)"] = np.nan
    df.loc[rng.choice(n, 8, replace=False), "Region"] = np.nan
    return df


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


def iris_like() -> pd.DataFrame:
    from sklearn.datasets import load_iris

    data = load_iris(as_frame=True)
    df = data.frame.rename(columns=lambda c: c.replace(" (cm)", "").replace(" ", "_"))
    df["species"] = np.asarray(data.target_names)[df.pop("target")]
    return df



def anomaly_like(n: int = 1200, share: float = 0.02, seed: int = 3) -> pd.DataFrame:
    """Transaktionen wie bei Betrugsdaten: drei normale Muster und wenige Ausreißer (Class = 1, Anteil share).
    ID = eindeutige ganze Zahl (Bezeichnung), channel = Kategorie, merchant_age mit 2 % Lücken."""
    rng = np.random.default_rng(seed)
    k = max(2, int(round(n * share)))
    m = n - k
    #               amount  items  hour  merchant_age  distance_km
    centers = np.array([[45.0, 1.5, 13.0, 6.0, 4.0], [280.0, 3.0, 18.0, 9.0, 12.0], [18.0, 1.0, 9.0, 2.0, 1.5]])
    spread = np.array([[12.0, 0.6, 2.5, 1.5, 1.5], [60.0, 1.0, 2.0, 2.0, 4.0], [5.0, 0.3, 1.5, 0.7, 0.6]])
    pick = rng.choice(3, m, p=[0.5, 0.2, 0.3])
    normal = centers[pick] + rng.normal(size=(m, 5)) * spread[pick]
    odd = np.column_stack([rng.uniform(900, 2500, k), rng.integers(6, 15, k), rng.uniform(1, 4.5, k),
                           rng.uniform(0, 0.3, k), rng.uniform(150, 900, k)])
    X = np.vstack([normal, odd])
    channel = np.r_[np.array(["online", "store", "atm"])[pick], rng.choice(["online", "atm"], k)]
    df = pd.DataFrame({"amount": X[:, 0].clip(0.5).round(2), "items": X[:, 1].clip(1).round().astype(int),
                       "hour": X[:, 2].clip(0, 23.9).round(1), "merchant_age": X[:, 3].clip(0).round(2),
                       "distance_km": X[:, 4].clip(0).round(1), "channel": channel,
                       "Class": np.r_[np.zeros(m, dtype=int), np.ones(k, dtype=int)]})
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.loc[rng.random(n) < 0.02, "merchant_age"] = np.nan
    df.insert(0, "ID", 100_000 + np.arange(n))
    return df


HOUR_MS = 3_600_000
OI_STEPS = {"5min": 5, "15min": 15, "30min": 30, "1h": 60, "4h": 240, "1d": 1440}  # intervalTime → Minuten


class FakeBybit:
    """Simulierte öffentliche Bybit-API v5 – nur für Tests, keine echten Kurse. Gleiche Antwortformen wie die echte
    API: Kerzen und Funding die neuesten zuerst, höchstens limit je Seite, start/end inklusive, Open Interest mit
    nextPageCursor im Raster intervalTime (simuliert alle 5 Minuten). Die jüngste Kerze läuft noch (now liegt in ihr).

    shock = (Symbol, Kerze): dort ein Kurssturz (−9 %) mit 25-fachem Umsatz, großer Spanne, Funding-Sprung und
    Rückgang des Open Interest (ab ¼ der Kerze: Liquidationen) – den soll die Anomalieerkennung finden. Kerze als
    Index ab dem Ende (−200 = 200 Kerzen vor der jüngsten)."""

    def __init__(self, symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"), minutes: int = 60, n: int = 2400,
                 now_ms: int = 1_790_000_000_000 + 30 * 60_000, seed: int = 7, shock=("SOLUSDT", -200)):
        from tools.bybit_features import BybitError  # noqa: PLC0415 – nur bei Bedarf (Pfad setzt der Test)

        self.error = BybitError
        self.now_ms, self.minutes = now_ms, minutes
        bar = minutes * 60_000
        self.bar = bar
        first = (now_ms // bar) * bar - (n - 1) * bar
        starts = first + bar * np.arange(n)
        rng = np.random.default_rng(seed)
        common = rng.normal(0, 0.006, n)
        self.kl, self.fund, self.oi, self.shock_start = {}, {}, {}, None
        f_ts = np.arange(first - first % (8 * HOUR_MS), now_ms + 1, 8 * HOUR_MS)
        o_ts = np.arange(first - first % HOUR_MS, now_ms + 1, HOUR_MS)
        o5 = np.arange(o_ts[0], now_ms + 1, 5 * 60_000)  # Open Interest alle 5 min (stündlich wie zuvor gezogen)
        for j, sym in enumerate(symbols):
            r = common * (1.0 + 0.3 * j) + (0 if j == 0 else rng.normal(0, 0.004, n))
            turn = np.exp(rng.normal(np.log(8e6 / (j + 1)), 0.25, n))
            wick = np.abs(rng.normal(0, 0.002, n))
            rate = 1e-4 + rng.normal(0, 1.5e-5, len(f_ts))
            oi_h = 2e5 * (j + 1) * np.exp(np.cumsum(rng.normal(0, 0.002, len(o_ts))))
            oi = np.exp(np.interp(o5, o_ts, np.log(oi_h)))  # zwischen den vollen Stunden: log-linear
            if shock and sym == shock[0]:
                i = shock[1] % n
                self.shock_start = int(starts[i])
                r[i], turn[i], wick[i] = -0.09, turn[i] * 25, 0.03
                rate[f_ts > starts[i]] = rate[f_ts > starts[i]] - 6e-4 * np.exp(
                    -(f_ts[f_ts > starts[i]] - starts[i]) / (3 * 8 * HOUR_MS))
                oi[o5 >= starts[i] + bar // 4] *= 0.85  # Liquidationen während der Sturz-Kerze
            close = 50.0 * (j + 1) * np.exp(np.cumsum(r))
            open_ = np.r_[close[0] / np.exp(r[0]), close[:-1]]
            high = np.maximum(open_, close) * (1 + wick)
            low = np.minimum(open_, close) * (1 - wick)
            self.kl[sym] = [[str(int(t)), f"{o:.6f}", f"{h:.6f}", f"{lo:.6f}", f"{c:.6f}", f"{v / c:.4f}", f"{v:.2f}"]
                            for t, o, h, lo, c, v in zip(starts, open_, high, low, close, turn)]
            self.fund[sym] = [(int(t), float(x)) for t, x in zip(f_ts, rate)]
            self.oi[sym] = [(int(t), float(x)) for t, x in zip(o5, oi)]
        self.calls = 0
        self.requests = []
        self._oi_pages = {}  # (Symbol, Raster, von, bis) → sortierte Zeilen – Blättern ohne neu zu sortieren

    def get(self, path: str, **p) -> dict:
        self.calls += 1
        self.requests.append((path, dict(p)))
        if path == "/v5/market/tickers":
            items = [{"symbol": s, "turnover24h": f"{sum(float(x[6]) for x in rows[-24:]):.2f}",
                      "volume24h": "1"} for s, rows in self.kl.items()]
            items += [{"symbol": "USDCUSDT", "turnover24h": "9e12", "volume24h": "1"},  # Stablecoin → auslassen
                      {"symbol": "BTCUSDT-26DEC26", "turnover24h": "9e12", "volume24h": "1"}]  # Future mit Laufzeit
            return {"category": p.get("category"), "list": items}
        sym = p.get("symbol")
        if sym not in self.kl:
            raise self.error("Bybit error 10001: params error: symbol invalid")
        limit = int(p.get("limit") or 200)
        if path == "/v5/market/kline":
            assert 1 <= limit <= 1000
            lo, hi = int(p.get("start") or 0), int(p.get("end") or 2**62)
            rows = sorted((x for x in self.kl[sym] if lo <= int(x[0]) <= hi), key=lambda x: -int(x[0]))
            return {"category": p.get("category"), "symbol": sym, "list": rows[:limit]}
        if path == "/v5/market/funding/history":
            assert 1 <= limit <= 200
            assert not (p.get("startTime") is not None and p.get("endTime") is None), "nur startTime → API-Fehler"
            lo, hi = int(p.get("startTime") or 0), int(p.get("endTime") or self.now_ms)
            rows = sorted((x for x in self.fund[sym] if lo <= x[0] <= hi), key=lambda x: -x[0])[:limit]
            return {"category": p.get("category"), "list": [{"symbol": sym, "fundingRate": f"{x:.8f}",
                                                             "fundingRateTimestamp": str(t)} for t, x in rows]}
        if path == "/v5/market/open-interest":
            assert 1 <= limit <= 200
            assert p.get("intervalTime") in OI_STEPS
            lo, hi = int(p.get("startTime") or 0), int(p.get("endTime") or self.now_ms)
            step = OI_STEPS[p["intervalTime"]] * 60_000
            key = (sym, step, lo, hi)
            if key not in self._oi_pages:
                self._oi_pages[key] = sorted((x for x in self.oi[sym] if lo <= x[0] <= hi and x[0] % step == 0),
                                             key=lambda x: -x[0])
            rows = self._oi_pages[key]
            off = int(p.get("cursor") or 0)
            page = rows[off:off + limit]
            nxt = str(off + limit) if off + limit < len(rows) else ""
            return {"symbol": sym, "category": p.get("category"), "nextPageCursor": nxt,
                    "list": [{"openInterest": f"{x:.4f}", "timestamp": str(t)} for t, x in page]}
        raise AssertionError(f"unbekannter Endpunkt {path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=777)
    ap.add_argument("--out", default="demo_college_synthetisch.csv")
    args = ap.parse_args()
    college_like(args.rows).to_csv(args.out)  # erste Spalte ohne Namen = Name der Hochschule (wie College_Data)
    print(f"{args.out} geschrieben ({args.rows} Zeilen, synthetisch)")
