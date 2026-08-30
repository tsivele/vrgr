#!/usr/bin/env bash
# Εγκατάσταση χωρίς brew / docker / node.
set -e
cd "$(dirname "$0")"

PY=${PY:-python3}
echo "→ Python: $($PY -V)"

echo "→ Εγκατάσταση εξαρτήσεων (user site)…"
$PY -m pip install --user --upgrade --quiet \
    "httpx>=0.27" "pydantic>=2.6" "numpy>=1.24" "anthropic>=0.40" \
    "imageio-ffmpeg>=0.5.1" "Pillow>=10.0"

if [ ! -f .env ]; then
    cp .env.example .env
    echo "→ Δημιουργήθηκε .env — ΒΑΛΕ ΤΑ KEYS ΣΟΥ ΕΚΕΙ."
fi

echo "→ Αρχικοποίηση βάσης…"
# Ο κώδικας ζει στο src/ και δεν εγκαθίσταται ως πακέτο — το PYTHONPATH
# είναι απαραίτητο, όπως ακριβώς κάνει και το run.sh.
PYTHONPATH="$PWD/src:$PYTHONPATH" $PY -m vrgr.cli init

echo
echo "✓ Έτοιμο.  Έλεγχος: ./run.sh doctor"
