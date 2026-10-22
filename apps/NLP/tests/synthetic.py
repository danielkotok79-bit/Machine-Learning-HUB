"""
Synthetische Testdaten – nur für Tests und zum Ausprobieren, keine echten Daten.

    python tests/synthetic.py                 # schreibt demo_yelp_synthetisch.csv (10 000 Bewertungen, wie im Kurs)
    python tests/synthetic.py --rows 3000
    python tests/synthetic.py --sms           # schreibt demo_sms_synthetisch.txt (Tab, ohne Kopfzeile)

yelp_like(): gleiche Spalten/Formate wie yelp.csv aus dem NLP-Projekt des Kurses (stars 1–5, text, cool/useful/funny)
sms_like():  wie SMSSpamCollection aus dem Kurs (label ham/spam, message; mit doppelten Nachrichten)
news_like(): Mehrklassen – deutsche Kurzmeldungen zu 4 Themen (Sport, Politik, Wirtschaft, Technik)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

POS = ["great", "amazing", "delicious", "friendly", "love", "best", "perfect", "excellent", "fantastic",
       "recommend", "awesome", "fresh", "tasty", "wonderful", "favorite"]
NEG = ["terrible", "rude", "worst", "awful", "cold", "slow", "disgusting", "never", "bad", "horrible",
       "dirty", "bland", "overpriced", "disappointing", "waited"]
NEUTRAL = ["food", "place", "service", "ordered", "restaurant", "table", "menu", "time", "staff", "chicken",
           "pizza", "coffee", "salad", "price", "lunch", "dinner", "bar", "drinks", "location", "night"]
GLUE = ["the", "and", "was", "we", "it", "very", "a", "to", "of", "for", "with", "this", "but", "they", "is",
        "i", "my", "our", "here", "so"]
STAR_P = np.array([749, 927, 1461, 3526, 3337]) / 10000  # Verteilung wie im Yelp-Datensatz des Kurses
ID_CHARS = np.array(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"))


def _ids(rng, n: int, length: int = 22) -> list[str]:
    return ["".join(row) for row in rng.choice(ID_CHARS, size=(n, length))]


def _review(rng, stars: int) -> str:
    """Bewertung aus Füllwörtern, Themen und Stimmungswörtern – je mehr Sterne, desto mehr positive Wörter."""
    n_words = int(np.clip(rng.lognormal(3.6, 0.6), 4, 400))
    p_pos = {1: 0.02, 2: 0.04, 3: 0.08, 4: 0.12, 5: 0.15}[stars]
    p_neg = {1: 0.15, 2: 0.12, 3: 0.07, 4: 0.03, 5: 0.015}[stars]
    kinds = rng.choice(4, n_words, p=[p_pos, p_neg, 0.25, 1 - p_pos - p_neg - 0.25])
    words = [rng.choice(POS) if k == 0 else rng.choice(NEG) if k == 1 else rng.choice(NEUTRAL) if k == 2
             else rng.choice(GLUE) for k in kinds]
    words[0] = words[0].capitalize()
    text = " ".join(words)
    if rng.random() < 0.3:
        text = text.replace(" and ", ", and ", 1) + rng.choice([".", "!", "!!", "..."])
    else:
        text += "."
    return text


def yelp_like(n: int = 10000, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stars = rng.choice([1, 2, 3, 4, 5], n, p=STAR_P)
    text = [_review(rng, int(s)) for s in stars]
    length = np.array([len(t) for t in text])
    days = rng.integers(0, 3000, n)
    date = (pd.Timestamp("2005-01-01") + pd.to_timedelta(days, unit="D")).strftime("%Y-%m-%d")
    useful = rng.poisson(0.5 + length / 900)
    return pd.DataFrame({
        "business_id": _ids(rng, n),
        "date": date,
        "review_id": _ids(rng, n),
        "stars": stars,
        "text": text,
        "type": "review",
        "user_id": _ids(rng, n),
        "cool": rng.poisson(0.3 + length / 1200),
        "useful": useful,
        "funny": rng.poisson(0.2 + length / 1500),
    })


SPAM = ["free", "win", "prize", "call", "txt", "claim", "urgent", "cash", "mobile", "offer", "reply", "now",
        "stop", "award", "guaranteed", "£1000", "500", "nokia", "ringtone", "winner"]
HAM = ["ok", "lol", "home", "later", "love", "going", "come", "sorry", "today", "dinner", "tomorrow", "u", "i",
       "you", "me", "what", "how", "work", "night", "good", "see", "time", "got", "know", "will", "just"]
SMS_GLUE = ["to", "the", "a", "is", "in", "and", "for", "your", "my", "on", "at", "it", "of", "be"]


def _sms(rng, spam: bool) -> str:
    n_words = int(np.clip(rng.lognormal(2.6 if spam else 2.3, 0.45), 2, 60))
    own, other = (SPAM, HAM) if spam else (HAM, SPAM)
    kinds = rng.choice(3, n_words, p=[0.45, 0.05, 0.5])
    words = [rng.choice(own) if k == 0 else rng.choice(other) if k == 1 else rng.choice(SMS_GLUE) for k in kinds]
    text = " ".join(words)
    if spam:
        text = text.upper() if rng.random() < 0.2 else text.capitalize()
        text += rng.choice(["!", "! Call 09061701461", ". T&C apply", " now!!"])
    elif rng.random() < 0.5:
        text = text.capitalize() + rng.choice([".", "..", "?", " :)"])
    return text


def sms_like(n: int = 5572, seed: int = 101, spam_share: float = 0.134, dup_share: float = 0.07) -> pd.DataFrame:
    """label (ham/spam) + message. dup_share: Anteil wortgleicher Nachrichten (wie in SMSSpamCollection)."""
    rng = np.random.default_rng(seed)
    labels = np.where(rng.random(n) < spam_share, "spam", "ham")
    msgs = [_sms(rng, lab == "spam") for lab in labels]
    for i in np.where(rng.random(n) < dup_share)[0]:  # Vorlagen: dieselbe Nachricht mehrfach
        j = int(rng.integers(0, n))
        msgs[i], labels[i] = msgs[j], labels[j]
    return pd.DataFrame({"label": labels, "message": msgs})


NEWS = {
    "Sport": ["Spiel", "Tor", "Mannschaft", "Trainer", "Liga", "Saison", "Sieg", "Niederlage", "Fans", "Stadion",
              "Meister", "Turnier"],
    "Politik": ["Regierung", "Minister", "Wahl", "Partei", "Gesetz", "Bundestag", "Koalition", "Opposition",
                "Abstimmung", "Kanzler", "Reform", "Debatte"],
    "Wirtschaft": ["Unternehmen", "Umsatz", "Gewinn", "Aktie", "Börse", "Inflation", "Zinsen", "Markt", "Euro",
                   "Bank", "Export", "Prozent"],
    "Technik": ["Software", "Computer", "Daten", "Internet", "Smartphone", "App", "Netzwerk", "Chip", "Update",
                "Roboter", "Cloud", "Sicherheitslücke"],
}
NEWS_GLUE = ["der", "die", "das", "und", "mit", "am", "nach", "für", "im", "über", "wird", "hat", "neue", "einer"]


def news_like(n: int = 1200, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    topics = rng.choice(list(NEWS), n, p=[0.3, 0.25, 0.25, 0.2])
    texts = []
    for t in topics:
        k = int(rng.integers(6, 25))
        others = [w for o, ws in NEWS.items() if o != t for w in ws]
        words = [rng.choice(NEWS[t]) if r < 0.4 else rng.choice(others) if r < 0.48 else rng.choice(NEWS_GLUE)
                 for r in rng.random(k)]
        texts.append(" ".join(words).capitalize() + ".")
    return pd.DataFrame({"titel": [f"Meldung {i + 1}" for i in range(n)], "text": texts, "thema": topics})


def write_sms_file(path, n: int = 5572, seed: int = 101) -> None:
    """Wie SMSSpamCollection: Tabulator, keine Kopfzeile, label zuerst."""
    sms_like(n, seed).to_csv(path, sep="\t", header=False, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--sms", action="store_true")
    a = ap.parse_args()
    if a.sms:
        write_sms_file("demo_sms_synthetisch.txt", n=a.rows or 5572)
        print("demo_sms_synthetisch.txt geschrieben")
    else:
        yelp_like(a.rows or 10000).to_csv("demo_yelp_synthetisch.csv", index=False)
        print("demo_yelp_synthetisch.csv geschrieben")
