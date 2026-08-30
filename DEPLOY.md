# Ανέβασμα στο Streamlit Community Cloud

## Πριν ξεκινήσεις — τι πρέπει να ξέρεις

Το Streamlit Community Cloud **ξαναχτίζει το container από το GitHub** σε κάθε
reboot, και οι εφαρμογές **κοιμούνται μετά από 12 ώρες** χωρίς επισκέψεις.
Ό,τι γράφεται στον δίσκο κατά την εκτέλεση χάνεται.

**Τι σημαίνει πρακτικά:**

| | Επιβιώνει; |
|---|---|
| Η γνώση που είναι στο `seed/vrgr_seed.db` (commit στο repo) | ✅ ναι |
| Ό,τι μαθαίνει το σύστημα **μετά** το τελευταίο commit | ❌ χάνεται στο restart |

Άρα το σύστημα **δεν μαθαίνει μόνο του στο cloud** — μαθαίνει όσο η εφαρμογή
είναι ξύπνια, και μηδενίζεται μετά. Για να κρατήσεις τη γνώση:

> Sidebar → **Αντίγραφο μνήμης** → **Κατέβασε** → αντικατέστησε το
> `seed/vrgr_seed.db` στο repo → commit → push

Είναι χειροκίνητο, αλλά λειτουργεί. Η οριστική λύση είναι εξωτερική βάση
(Postgres) — δεν είναι υλοποιημένη.

Όρια Community Cloud: **690MB–2.7GB RAM**, **0,078–2 πυρήνες**, ύπνος 12 ωρών.
Μια ανάλυση θέλει 3–5 λεπτά· στο χαμηλό άκρο των πόρων μπορεί να είναι πιο αργή.

---

## Βήμα 1 — GitHub repo (**ΙΔΙΩΤΙΚΟ**)

Το repo περιέχει τη βάση με την έρευνά σου. Πρέπει να είναι **private**.

```bash
cd /Users/tsivelekidis/Downloads/reels-viral-gr
gh repo create vrgr --private --source=. --push
```

Χωρίς `gh`: φτιάξε private repo στο github.com και μετά:

```bash
git remote add origin https://github.com/<ΤΟ_USERNAME_ΣΟΥ>/vrgr.git
git branch -M main && git push -u origin main
```

Το `.env` **δεν** ανεβαίνει — είναι gitignored. Επιβεβαιωμένο.

---

## Βήμα 2 — Deploy

1. Πήγαινε στο **share.streamlit.io** και συνδέσου με GitHub.
2. **New app** → διάλεξε το repo `vrgr`, branch `main`.
3. **Main file path**: `streamlit_app.py`
4. **Advanced settings** → **Python version**: `3.11` (ή νεότερη).

---

## Βήμα 3 — Κλειδιά

Στο **Advanced settings → Secrets**, επικόλλησε:

```toml
HIKER_API_KEY = "το_κλειδί_σου"
ANTHROPIC_API_KEY = "το_κλειδί_σου"

VRGR_VISION_MODEL = "claude-opus-5"
VRGR_WRITER_MODEL = "claude-opus-5"
VRGR_FAST_MODEL = "claude-haiku-4-5"

# Χαμηλότερο budget στο cloud: λιγότεροι πόροι, πιο αργές κλήσεις.
HIKER_BUDGET_PER_RUN = "80"
VRGR_MAX_FRAMES = "10"
```

Η εφαρμογή τα διαβάζει αυτόματα μέσω `st.secrets` — δεν χρειάζεται `.env`.

⚠️ **Κάνε rotate τα κλειδιά πριν το deploy.** Τα τωρινά έχουν κυκλοφορήσει.

---

## Βήμα 4 — Πρόσβαση

Στο **Settings → Sharing** της εφαρμογής, όρισε ποια emails έχουν πρόσβαση.
Χωρίς αυτό, όποιος βρει το link **ξοδεύει τα δικά σου credits** HikerAPI και
Anthropic.

---

## Ρουτίνα χρήσης

1. Ανοίγεις την εφαρμογή, ανεβάζεις Reel, παίρνεις λεζάντα + hashtags.
2. Δημοσιεύεις. Καταγράφεις το αποτέλεσμα στο **Ιστορικό**.
3. **Πριν κλείσεις**: sidebar → Αντίγραφο μνήμης → Κατέβασε.
4. Αντικαθιστάς το `seed/vrgr_seed.db`, commit, push.

Το βήμα 3-4 είναι που κάνει τη διαφορά ανάμεσα σε «γεννήτρια λεζαντών» και
«σύστημα που μαθαίνει». Χωρίς αυτό, κάθε restart σβήνει ό,τι έμαθε.

---

## Τοπικά (χωρίς κανένα από τα παραπάνω)

```bash
streamlit run streamlit_app.py     # ή: python3 app.py
```

Τοπικά η μνήμη είναι μόνιμη — δεν χρειάζεται κανένα αντίγραφο.
