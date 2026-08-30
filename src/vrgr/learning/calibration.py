"""
Βαθμονόμηση βαρών από πραγματικά αποτελέσματα (απαίτηση #13).

ΠΡΟΣΟΧΗ ΣΤΗ ΜΕΘΟΔΟΛΟΓΙΑ: με λίγα δείγματα, η «βελτιστοποίηση» βαρών είναι
υπερπροσαρμογή σε θόρυβο — χειρότερη από το να μην κάνεις τίποτα.

Γι' αυτό εδώ:
  • Απαιτούνται ≥12 μετρημένα αποτελέσματα πριν προταθεί οτιδήποτε.
  • Οι μεταβολές περιορίζονται σε ±25% ανά γύρο (shrinkage).
  • Η αλλαγή ΔΕΝ εφαρμόζεται αυτόματα — γράφεται πρόταση που εγκρίνεις.

Το τελευταίο είναι σκόπιμο: τα βάρη είναι η «κρίση» του συστήματος, και
δεν πρέπει να αλλάζει σιωπηλά πίσω από την πλάτη σου.
"""
from __future__ import annotations

import json
from pathlib import Path
from ..logging_setup import get_logger
from ..memory.repository import Repository

log = get_logger("learning.calibration")

MIN_OUTCOMES = 12
MAX_SHIFT = 0.25


def analyze(repo: Repository, config_dir: Path) -> dict:
    """Ελέγχει αν υπάρχουν αρκετά δεδομένα και τι προτείνουν."""
    outcomes = repo.outcomes()
    usable = [o for o in outcomes
              if o.get("predicted_score") is not None
              and o.get("outlier_score") is not None]
    report = {"n_outcomes": len(outcomes), "n_usable": len(usable),
              "ready": len(usable) >= MIN_OUTCOMES, "suggestions": [],
              "min_required": MIN_OUTCOMES}
    if not report["ready"]:
        report["message"] = (
            f"Χρειάζονται {MIN_OUTCOMES} μετρημένες δημοσιεύσεις για ασφαλή "
            f"βαθμονόμηση· υπάρχουν {len(usable)}. Μέχρι τότε τα βάρη μένουν "
            f"ως έχουν — η προσαρμογή σε λίγα δείγματα είναι υπερπροσαρμογή, "
            f"όχι μάθηση.")
        return report

    # Συστηματική μεροληψία: υπερεκτιμά ή υποεκτιμά το σύστημα;
    errors = [o["outlier_score"] - o["predicted_score"] for o in usable]
    bias = sum(errors) / len(errors)
    mae = sum(abs(e) for e in errors) / len(errors)
    report["bias"] = round(bias, 2)
    report["mean_abs_error"] = round(mae, 2)

    if abs(bias) >= 6.0:
        direction = "υπερεκτιμά" if bias < 0 else "υποεκτιμά"
        report["suggestions"].append({
            "type": "evidence_multiplier",
            "message": f"Το σύστημα {direction} συστηματικά κατά {abs(bias):.1f} "
                       f"πόντους. Πρότεινε προσαρμογή του εύρους του "
                       f"πολλαπλασιαστή τεκμηρίων.",
            "current_bias": round(bias, 2),
        })

    # Ποιοι πυλώνες συσχετίζονται όντως με επιτυχία;
    from .feedback import _spearman
    pillar_data: dict = {}
    for o in usable:
        run = repo.get_run(o["run_id"])
        if not run:
            continue
        try:
            payload = json.loads(run.get("result_json") or "{}")
            pillars = (payload.get("winner") or {}).get("score", {}).get("pillars", [])
        except json.JSONDecodeError:
            continue
        for p in pillars:
            pillar_data.setdefault(p["name"], {"raw": [], "actual": []})
            pillar_data[p["name"]]["raw"].append(p["raw"])
            pillar_data[p["name"]]["actual"].append(o["outlier_score"])

    weights_path = config_dir / "weights.json"
    current = json.loads(weights_path.read_text(encoding="utf-8")) \
        if weights_path.is_file() else {"pillars": {}}
    proposed = {}
    for name, data in pillar_data.items():
        if len(data["raw"]) < MIN_OUTCOMES:
            continue
        corr = _spearman(data["raw"], data["actual"])
        if corr is None:
            continue
        cur_w = (current.get("pillars", {}).get(name) or {}).get("weight")
        if cur_w is None:
            continue
        # Θετική συσχέτιση → αυξάνουμε το βάρος, με shrinkage.
        adjustment = max(-MAX_SHIFT, min(MAX_SHIFT, corr * 0.5))
        new_w = round(cur_w * (1.0 + adjustment), 4)
        proposed[name] = {"current": cur_w, "proposed": new_w,
                          "correlation": corr, "n": len(data["raw"])}

    if proposed:
        total = sum(v["proposed"] for v in proposed.values())
        missing = sum((v.get("weight") or 0)
                      for k, v in (current.get("pillars") or {}).items()
                      if k not in proposed)
        scale = (1.0 - missing) / total if total else 1.0
        for v in proposed.values():
            v["proposed"] = round(v["proposed"] * scale, 4)
        report["suggestions"].append({
            "type": "pillar_weights",
            "message": "Προτεινόμενα βάρη βάσει συσχέτισης με πραγματικά "
                       "αποτελέσματα (κανονικοποιημένα σε άθροισμα 1).",
            "weights": proposed,
        })
    report["message"] = (f"{len(usable)} αποτελέσματα αναλύθηκαν. "
                         f"Μέσο απόλυτο σφάλμα {mae:.1f} πόντοι.")
    return report


def apply_suggestion(config_dir: Path, weights: dict,
                     backup: bool = True) -> Path:
    """Εφαρμόζει προτεινόμενα βάρη — ΜΟΝΟ κατόπιν ρητής έγκρισης."""
    path = config_dir / "weights.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if backup:
        bak = path.with_suffix(f".backup-{int(__import__('time').time())}.json")
        bak.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        log.info("Αντίγραφο ασφαλείας: %s", bak.name)
    for name, spec in weights.items():
        if name in data.get("pillars", {}):
            data["pillars"][name]["weight"] = spec["proposed"]
            data["pillars"][name]["βαθμονομήθηκε"] = (
                f"συσχέτιση {spec['correlation']} σε {spec['n']} αποτελέσματα")
    data["version"] = int(data.get("version", 1)) + 1
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Τα βάρη ενημερώθηκαν (έκδοση %d)", data["version"])
    return path
