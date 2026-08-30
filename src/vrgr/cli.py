"""
Γραμμή εντολών.

    ./run.sh serve                         ΑΝΟΙΓΜΑ ΤΗΣ ΕΦΑΡΜΟΓΗΣ (ή: python3 app.py)
    ./run.sh doctor                        έλεγχος εγκατάστασης και κλειδιών
    ./run.sh analyze video.mp4             πλήρης ανάλυση → λεζάντα + hashtags
    ./run.sh feedback <run_id> <url>       καταγραφή πραγματικού αποτελέσματος
    ./run.sh seed                          τροφοδότηση μνήμης από benchmark accounts
    ./run.sh research @username            έρευνα ενός λογαριασμού
    ./run.sh memory                        κατάσταση μνήμης
    ./run.sh runs                          πρόσφατες εκτελέσεις
    ./run.sh balance                       υπόλοιπο credits HikerAPI
    ./run.sh calibrate                     αναφορά βαθμονόμησης
    ./run.sh verify-endpoints              επαλήθευση endpoints με το ζωντανό spec
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from .config import load_settings
from .errors import VRGRError
from .logging_setup import get_logger, setup_logging

log = get_logger("cli")


def _pipeline(settings):
    from .pipeline.orchestrator import Pipeline
    return Pipeline(settings)


# ── εντολές ───────────────────────────────────────────────────────────

def cmd_init(args, settings) -> int:
    from .memory.db import Database
    db = Database(settings.db_path)
    counts = db.counts()
    db.close()
    print(f"✓ Βάση έτοιμη: {settings.db_path}")
    print(f"  Πίνακες: {', '.join(counts)}")
    print(f"✓ Φάκελος δεδομένων: {settings.data_dir}")
    return 0


def cmd_doctor(args, settings) -> int:
    from .video import ffmpeg as FF
    ok = True
    print("\n── ΕΛΕΓΧΟΣ ΕΓΚΑΤΑΣΤΑΣΗΣ ──────────────────────────────────\n")

    print(f"  Python                {sys.version.split()[0]}")
    for module in ("httpx", "pydantic", "numpy", "anthropic"):
        try:
            __import__(module)
            print(f"  {module:21} ✓")
        except ImportError:
            print(f"  {module:21} ✗ ΛΕΙΠΕΙ → pip3 install --user {module}")
            ok = False

    tools = FF.available()
    if tools["ok"]:
        print(f"  ffmpeg                ✓ {Path(tools['ffmpeg']).name}")
    else:
        print("  ffmpeg                ✗ ΛΕΙΠΕΙ → pip3 install --user imageio-ffmpeg")
        ok = False
    probe_note = "✓" if tools["ffprobe"] else "— (προαιρετικό· το ffmpeg το καλύπτει)"
    print(f"  ffprobe               {probe_note}")

    print("\n── ΚΛΕΙΔΙΑ ───────────────────────────────────────────────\n")
    print(f"  HIKER_API_KEY         {'✓ ορισμένο' if settings.hiker.enabled else '✗ ΛΕΙΠΕΙ — χωρίς αυτό δεν γίνεται έρευνα'}")
    print(f"  ANTHROPIC_API_KEY     {'✓ ορισμένο' if settings.models.enabled else '✗ ΛΕΙΠΕΙ — απαραίτητο'}")
    if not settings.models.enabled:
        ok = False
    print(f"  Embeddings            {settings.embeddings.provider}")
    print(f"  ASR (ήχος)            {settings.video.asr_provider}"
          + ("  ⚠ χωρίς μεταγραφή, η ανάλυση βασίζεται μόνο σε καρέ"
             if settings.video.asr_provider == "none" else ""))

    print("\n── ΧΩΡΟΣ ΣΤΟΝ ΔΙΣΚΟ ──────────────────────────────────────\n")
    from .maintenance import usage
    for k, v in usage(settings).items():
        print(f"  {k:21} {v/1e6:>8.1f} MB")
    print("  (καθαρισμός: ./run.sh cleanup)")

    print("\n── ΜΝΗΜΗ ─────────────────────────────────────────────────\n")
    from .memory.db import Database
    db = Database(settings.db_path)
    for table, n in db.counts().items():
        print(f"  {table:21} {n:>8,}")
    db.close()

    if settings.hiker.enabled:
        print("\n── HIKERAPI ──────────────────────────────────────────────\n")
        from .cache.store import HttpCache
        from .clients.hiker.client import HikerClient
        client = HikerClient(settings.hiker, HttpCache(settings.cache_path))
        balance = client.balance()
        if balance is not None:
            print(f"  Σύνδεση               ✓")
            print(f"  Υπόλοιπο              {json.dumps(balance, ensure_ascii=False)}")
        else:
            print("  Σύνδεση               ✗ δεν απάντησε — έλεγξε το κλειδί")
            ok = False
        client.close()

    print()
    print("✓ Όλα έτοιμα." if ok else "⚠ Υπάρχουν προβλήματα — δες παραπάνω.")
    return 0 if ok else 1


def cmd_analyze(args, settings) -> int:
    from .report import render, render_compact
    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        print(f"✗ Δεν βρέθηκε το αρχείο: {video}")
        return 1

    pipe = _pipeline(settings)
    try:
        result = pipe.analyze(
            video, user_context=args.context or "",
            skip_research=args.no_research, n_captions=args.captions,
            benchmark_creators=(args.creators.split(",") if args.creators else None))
    except VRGRError as exc:
        print(f"\n✗ {exc}")
        return 1
    finally:
        pipe.close()

    output = render_compact(result) if args.compact else render(result, args.verbose)
    print(output)

    if args.json:
        path = settings.runs_dir / f"{result.run_id}.json"
        path.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n→ Πλήρες JSON: {path}")
    return 0


def cmd_feedback(args, settings) -> int:
    from .cache.store import HttpCache
    from .clients.hiker.client import HikerClient
    from .clients.llm.embeddings import build_embedder
    from .learning.feedback import FeedbackLoop
    from .memory.db import Database
    from .memory.patterns import PatternStore
    from .memory.repository import Repository

    db = Database(settings.db_path)
    repo = Repository(db, build_embedder(settings.embeddings))
    client = (HikerClient(settings.hiker, HttpCache(settings.cache_path))
              if settings.hiker.enabled else None)
    loop = FeedbackLoop(repo, PatternStore(db), client)

    manual = {k: v for k, v in (
        ("views", args.views), ("likes", args.likes),
        ("comments", args.comments), ("followers", args.followers)) if v}
    try:
        out = loop.record(args.run_id, args.url or "", manual or None)
    except VRGRError as exc:
        print(f"✗ {exc}")
        return 1
    finally:
        if client:
            client.close()

    print("\n── ΚΑΤΑΓΡΑΦΗ ΑΠΟΤΕΛΕΣΜΑΤΟΣ ───────────────────────────────\n")
    print(f"  Εκτέλεση        {out['run_id']}")
    print(f"  Προβολές        {out['views']:,}".replace(",", "."))
    if out.get("followers"):
        print(f"  Followers       {out['followers']:,}".replace(",", "."))
    if out.get("vf_ratio"):
        print(f"  V/F             {out['vf_ratio']:.1f}×  "
              f"{'✓ ΕΠΙΤΥΧΙΑ' if out['succeeded'] else '— κάτω από το κατώφλι'}")
    if out.get("predicted") is not None and out.get("actual") is not None:
        print(f"  Πρόβλεψη        {out['predicted']:.1f}")
        print(f"  Πραγματικό      {out['actual']:.1f}")
        print(f"  Σφάλμα          {out['error']:+.1f}")
    print(f"  Μοτίβα          {out['patterns_updated']} ενημερώθηκαν")

    summary = loop.summary()
    print(f"\n  Συνολικά: {summary['n_outcomes']} μετρημένες δημοσιεύσεις")
    if summary.get("correlation") is not None:
        print(f"  Συσχέτιση πρόβλεψης–πραγματικότητας: {summary['correlation']} "
              f"(μέσο σφάλμα {summary['mean_abs_error']})")
    if summary.get("success_rate") is not None:
        print(f"  Ποσοστό επιτυχίας: {summary['success_rate']:.0%} "
              f"(διάμεσο V/F {summary['median_vf']})")
    db.close()
    return 0


def cmd_seed(args, settings) -> int:
    """Τροφοδότηση μνήμης από τους benchmark λογαριασμούς."""
    if not settings.hiker.enabled:
        print("✗ Λείπει το HIKER_API_KEY.")
        return 1
    from .analysis import metrics as M
    from .cache.store import HttpCache
    from .clients.hiker.client import HikerClient
    from .clients.llm.embeddings import build_embedder
    from .memory.db import Database
    from .memory.repository import Repository
    from .research.collector import Collector, fill_hashtag_stat
    from .research.planner import load_seed_creators

    usernames = (args.creators.split(",") if args.creators
                 else load_seed_creators(settings.config_dir))
    usernames = [u.strip().lstrip("@") for u in usernames if u.strip()][:args.limit]
    if not usernames:
        print("✗ Δεν βρέθηκαν λογαριασμοί. Δες config/seed_creators.json")
        return 1

    db = Database(settings.db_path)
    repo = Repository(db, build_embedder(settings.embeddings))
    client = HikerClient(settings.hiker, HttpCache(settings.cache_path),
                         budget=args.budget)
    collector = Collector(client, repo)

    print(f"\nΤροφοδότηση μνήμης από {len(usernames)} λογαριασμούς "
          f"(budget {args.budget} μονάδες)…\n")
    totals = {"creators": 0, "posts": 0, "snapshots": 0, "outliers": 0}
    for i, username in enumerate(usernames, 1):
        try:
            creator, posts = collector.creator_reels(username)
        except VRGRError as exc:
            print(f"  [{i}/{len(usernames)}] @{username}: {exc}")
            break
        if not creator:
            print(f"  [{i}/{len(usernames)}] @{username}: δεν βρέθηκε")
            continue
        M.enrich(posts)
        outliers = M.rank_outliers(posts, min_score=45.0, greek_only=False)
        repo.upsert_creator(creator, args.niche, is_benchmark=True)
        stats = repo.save_posts(posts, args.niche)
        totals["creators"] += 1
        totals["posts"] += stats["new_posts"]
        totals["snapshots"] += stats["new_snapshots"]
        totals["outliers"] += len(outliers)
        best = max((p.normalized.vf_ratio or 0) for p in posts) if posts else 0
        print(f"  [{i}/{len(usernames)}] @{username:28} "
              f"{creator.followers:>8,} fol · {len(posts):>3} reels · "
              f"{len(outliers):>2} outliers · καλύτερο V/F {best:>6.1f}×"
              .replace(",", "."))

    print(f"\n✓ {totals['creators']} λογαριασμοί · {totals['posts']} νέα posts · "
          f"{totals['snapshots']} στιγμιότυπα · {totals['outliers']} outliers")
    print(f"  Κλήσεις API: {client.stats.calls} (cache: {client.stats.cache_hits})")
    client.close()
    db.close()
    return 0


def cmd_research(args, settings) -> int:
    if not settings.hiker.enabled:
        print("✗ Λείπει το HIKER_API_KEY.")
        return 1
    from .analysis import metrics as M
    from .cache.store import HttpCache
    from .clients.hiker.client import HikerClient
    from .research.collector import Collector, fill_hashtag_stat

    client = HikerClient(settings.hiker, HttpCache(settings.cache_path),
                         budget=args.budget)
    collector = Collector(client)
    target = args.target.strip()

    if target.startswith("#"):
        tag = target.lstrip("#")
        posts, stat = collector.hashtag(tag)
        trend = collector.hashtag_trend(tag)
        # Κανένα endpoint hashtag δεν επιστρέφει follower_count (μετρημένο σε
        # ζωντανό API: 0/28). Χωρίς εμπλουτισμό η στήλη V/F — δηλαδή ο λόγος
        # ύπαρξης της εντολής — θα ήταν κενή.
        collector.enrich_followers(posts, max_profiles=args.limit + 4)
        M.enrich(posts)
        if stat is not None:
            fill_hashtag_stat(stat, [p for p in posts if p.metrics.views])
        print(f"\n── #{tag} ─────────────────────────────────────────\n")
        if stat:
            print(f"  Μέγεθος (✓ μετρημένο)  {stat.media_count:,} posts"
                  .replace(",", ".") if stat.media_count else "  Μέγεθος  —")
            print(f"  Επίπεδο                {stat.tier}")
            if stat.difficulty:
                print(f"  Δυσκολία (≈ παράγωγη)  {stat.difficulty}/100")
            if stat.small_account_share is not None:
                print(f"  Μικροί λογαριασμοί     {stat.small_account_share:.0%} "
                      f"των κορυφαίων")
        if trend:
            print(f"  Τάση (≈ παράγωγη)      {trend['label']} "
                  f"(recent/top = {trend['ratio']})")
        _print_posts(posts, args.limit)
    else:
        username = target.lstrip("@")
        creator, posts = collector.creator_reels(username)
        if not creator:
            print(f"✗ Δεν βρέθηκε ο @{username}")
            client.close()
            return 1
        M.enrich(posts)
        print(f"\n── @{creator.username} ────────────────────────────\n")
        print(f"  {creator.full_name}")
        print(f"  Followers (✓)          {creator.followers:,}".replace(",", "."))
        print(f"  Ελληνικότητα (≈)       {creator.greek_confidence:.0%}")
        summary = M.corpus_summary(posts)
        print(f"  Reels στο δείγμα       {summary['n']}")
        if summary["median_views"]:
            print(f"  Διάμεσο προβολών (✓)   {summary['median_views']:,}"
                  .replace(",", "."))
        if summary["max_vf"]:
            print(f"  Καλύτερο V/F (≈)       {summary['max_vf']}×")
        _print_posts(posts, args.limit)

    print(f"\n  Κλήσεις API: {client.stats.calls} (cache: {client.stats.cache_hits})")
    client.close()
    return 0


def _print_posts(posts: list, limit: int) -> None:
    from .analysis import metrics as M
    ranked = sorted(posts, key=lambda p: -(p.normalized.outlier_score or 0))[:limit]
    if not ranked:
        print("\n  (κανένα post)")
        return
    print(f"\n  {'προβολές':>10}{'followers':>11}{'V/F':>8}{'σκορ':>7}  λογαριασμός / λεζάντα")
    for p in ranked:
        v = f"{p.metrics.views:,}".replace(",", ".") if p.metrics.views else "—"
        f = (f"{p.followers_at_observation:,}".replace(",", ".")
             if p.followers_at_observation else "—")
        vf = f"{p.normalized.vf_ratio:.1f}×" if p.normalized.vf_ratio else "—"
        sc = f"{p.normalized.outlier_score:.0f}" if p.normalized.outlier_score else "—"
        print(f"  {v:>10}{f:>11}{vf:>8}{sc:>7}  @{p.username} · "
              f"{(p.caption_body or '')[:44]}")


def cmd_memory(args, settings) -> int:
    from .clients.llm.embeddings import build_embedder
    from .memory.db import Database
    from .memory.patterns import PatternStore
    from .memory.repository import Repository
    from .memory.retrieval import MemoryRetriever

    db = Database(settings.db_path)
    embedder = build_embedder(settings.embeddings)
    retriever = MemoryRetriever(db, embedder)
    patterns = PatternStore(db)

    print("\n── ΚΑΤΑΣΤΑΣΗ ΜΝΗΜΗΣ ──────────────────────────────────────\n")
    stats = retriever.corpus_stats()
    print(f"  Posts                 {stats['posts']:>8,}".replace(",", "."))
    print(f"  Ελληνικά              {stats['greek_posts']:>8,}".replace(",", "."))
    print(f"  Στιγμιότυπα           {stats['snapshots']:>8,}".replace(",", "."))
    print(f"  Outliers (≥45)        {stats['outliers']:>8,}".replace(",", "."))
    pstats = patterns.stats()
    print(f"\n  Μοτίβα                {pstats['patterns']:>8}")
    print(f"  Αξιοποιήσιμα (n≥4)    {pstats['usable']:>8}")
    print(f"  Μέσα δείγματα         {pstats['avg_samples']:>8}")

    if args.query:
        print(f"\n── ΑΝΑΖΗΤΗΣΗ: «{args.query}» ──────────────────────\n")
        for row in retriever.search(args.query, limit=args.limit):
            print(f"  [{row['retrieval_score']:.3f}] @{row['username']} "
                  f"· {(row.get('views') or 0):,} προβολές "
                  f"· V/F {row.get('vf_ratio') or 0:.1f}×".replace(",", "."))
            print(f"      «{(row.get('caption_body') or '')[:90]}»")
            if row.get("hashtags"):
                print(f"      {' '.join('#' + t for t in row['hashtags'][:8])}")

    if args.patterns:
        print("\n── ΜΟΤΙΒΑ ΜΕ ΤΕΚΜΗΡΙΩΣΗ ──────────────────────────────\n")
        for kind in ("caption_structure", "hashtag"):
            found = patterns.by_kind(kind, args.niche or "", limit=12)
            if not found:
                continue
            print(f"  [{kind}]")
            for p in found:
                print(f"    {p.lower_bound():.2f} (μέσο {p.mean:.2f}, n={p.n:.0f}) "
                      f"{p.description_el or p.key}")
    db.close()
    return 0


def cmd_runs(args, settings) -> int:
    from .clients.llm.embeddings import build_embedder
    from .memory.db import Database
    from .memory.repository import Repository
    db = Database(settings.db_path)
    repo = Repository(db, build_embedder(settings.embeddings))
    runs = repo.recent_runs(args.limit)
    if not runs:
        print("Καμία εκτέλεση ακόμη. Ξεκίνα με: ./run.sh analyze video.mp4")
        db.close()
        return 0
    print("\n── ΠΡΟΣΦΑΤΕΣ ΕΚΤΕΛΕΣΕΙΣ ──────────────────────────────────\n")
    for r in runs:
        when = time.strftime("%d/%m %H:%M", time.localtime(r["created_at"]))
        print(f"  {r['run_id']}  {when}  {r['predicted_score'] or 0:5.1f} "
              f"({r['confidence']:7}) {r['niche'][:14]:14} «{(r['caption'] or '')[:44]}»")
    db.close()
    return 0


def cmd_balance(args, settings) -> int:
    if not settings.hiker.enabled:
        print("✗ Λείπει το HIKER_API_KEY.")
        return 1
    from .cache.store import HttpCache
    from .clients.hiker.client import HikerClient
    client = HikerClient(settings.hiker, HttpCache(settings.cache_path))
    balance = client.balance()
    print(json.dumps(balance, ensure_ascii=False, indent=2)
          if balance else "✗ Δεν ήταν δυνατή η ανάγνωση υπολοίπου.")
    client.close()
    return 0 if balance else 1


def cmd_calibrate(args, settings) -> int:
    from .clients.llm.embeddings import build_embedder
    from .learning.calibration import analyze as cal_analyze, apply_suggestion
    from .memory.db import Database
    from .memory.repository import Repository
    db = Database(settings.db_path)
    repo = Repository(db, build_embedder(settings.embeddings))
    report = cal_analyze(repo, settings.config_dir)
    print("\n── ΒΑΘΜΟΝΟΜΗΣΗ ───────────────────────────────────────────\n")
    print(f"  Αποτελέσματα: {report['n_usable']}/{report['min_required']} "
          f"απαιτούμενα")
    print(f"  {report['message']}")
    for s in report.get("suggestions", []):
        print(f"\n  • {s['message']}")
        if s["type"] == "pillar_weights":
            for name, spec in s["weights"].items():
                print(f"      {name:24} {spec['current']:.3f} → {spec['proposed']:.3f} "
                      f"(συσχέτιση {spec['correlation']}, n={spec['n']})")
            if args.apply:
                apply_suggestion(settings.config_dir, s["weights"])
                print("\n  ✓ Τα βάρη ενημερώθηκαν (κρατήθηκε αντίγραφο ασφαλείας).")
            else:
                print("\n  Για εφαρμογή: ./run.sh calibrate --apply")
    db.close()
    return 0


def cmd_cleanup(args, settings) -> int:
    from .cache.store import HttpCache
    from .maintenance import cleanup, usage
    before = usage(settings)
    cache = HttpCache(settings.cache_path)
    freed = cleanup(settings, cache=cache, aggressive=args.aggressive)
    cache.close()
    after = usage(settings)
    print("\n── ΣΥΝΤΗΡΗΣΗ ────────────────────────────────────────────\n")
    print(f"  {'':18}{'πριν':>10}{'μετά':>10}")
    for k in before:
        print(f"  {k:18}{before[k]/1e6:>9.1f}M{after[k]/1e6:>9.1f}M")
    print(f"\n  Ελευθερώθηκαν {freed['bytes']/1e6:.1f} MB "
          f"({freed['frames_dirs']} φάκελοι καρέ, {freed['uploads']} uploads, "
          f"{freed['cache_rows']} εγγραφές cache)")
    print("  Η βάση γνώσης ΔΕΝ αγγίζεται — είναι ό,τι έχει μάθει το σύστημα.")
    return 0


def cmd_serve(args, settings) -> int:
    from .web.server import serve
    serve(settings, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_verify_endpoints(args, settings) -> int:
    """Διασταύρωση του μητρώου με το ΖΩΝΤΑΝΟ openapi.json του HikerAPI."""
    import httpx
    from .clients.hiker import endpoints as E
    print("Λήψη openapi.json από το HikerAPI…")
    try:
        spec = httpx.get(f"{settings.hiker.base_url}/openapi.json",
                         timeout=60, headers={"User-Agent": settings.hiker.user_agent}
                         ).json()
    except Exception as exc:                          # noqa: BLE001
        print(f"✗ Αποτυχία λήψης: {type(exc).__name__}")
        return 1
    paths = spec.get("paths", {})
    problems = 0
    for path, ep in E.REGISTRY.items():
        op = paths.get(path, {}).get("get")
        if not op:
            print(f"  ✗ ΔΕΝ ΥΠΑΡΧΕΙ ΠΛΕΟΝ: {path}")
            problems += 1
            continue
        params = op.get("parameters", [])
        required = {p["name"] for p in params if p.get("required")}
        allowed = {p["name"] for p in params}
        if set(ep.required) != required:
            print(f"  ✗ {path}: υποχρεωτικά δικά μας={sorted(ep.required)} "
                  f"spec={sorted(required)}")
            problems += 1
        unknown = (set(ep.optional) | set(ep.required)) - allowed
        if unknown:
            print(f"  ✗ {path}: άγνωστες παράμετροι {sorted(unknown)}")
            problems += 1
        summary = (op.get("summary") or "").lower()
        if "deprecat" in summary or "instead" in summary:
            print(f"  ⚠ {path}: το spec προτείνει αλλαγή → {op.get('summary')[:90]}")
    print(f"\n{len(E.REGISTRY)} endpoints ελέγχθηκαν έναντι {len(paths)} του spec — "
          f"{problems} προβλήματα")
    return 1 if problems else 0


# ── είσοδος ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vrgr",
        description="Instagram Reels Viral Intelligence — ελληνική αγορά",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--log-level", default="", help="DEBUG|INFO|WARNING|ERROR")
    p.add_argument("--quiet", action="store_true", help="μόνο σφάλματα")
    sub = p.add_subparsers(dest="command", required=True)

    sv = sub.add_parser("serve", help="άνοιγμα της εφαρμογής στον browser")
    sv.add_argument("--port", type=int, default=8778)
    sv.add_argument("--no-browser", action="store_true")

    cl = sub.add_parser("cleanup", help="καθαρισμός προσωρινών αρχείων και cache")
    cl.add_argument("--aggressive", action="store_true",
                    help="πιο επιθετικά όρια ηλικίας")

    sub.add_parser("init", help="αρχικοποίηση βάσης")
    sub.add_parser("doctor", help="έλεγχος εγκατάστασης, κλειδιών, μνήμης")
    sub.add_parser("balance", help="υπόλοιπο credits HikerAPI")
    sub.add_parser("verify-endpoints", help="επαλήθευση endpoints με το ζωντανό spec")

    a = sub.add_parser("analyze", help="ανάλυση Reel → λεζάντα + hashtags")
    a.add_argument("video", help="διαδρομή αρχείου βίντεο")
    a.add_argument("--context", default="", help="επιπλέον πληροφορίες για το βίντεο")
    a.add_argument("--captions", type=int, default=8, help="πλήθος υποψηφίων λεζαντών")
    a.add_argument("--creators", default="", help="benchmark λογαριασμοί, χωρισμένοι με κόμμα")
    a.add_argument("--no-research", action="store_true",
                   help="χωρίς κλήσεις HikerAPI (μόνο μνήμη)")
    a.add_argument("--compact", action="store_true", help="μόνο η τελική απόφαση")
    a.add_argument("--verbose", action="store_true", help="αναλυτική έξοδος")
    a.add_argument("--json", action="store_true", help="αποθήκευση πλήρους JSON")

    f = sub.add_parser("feedback", help="καταγραφή πραγματικού αποτελέσματος")
    f.add_argument("run_id")
    f.add_argument("url", nargs="?", default="", help="URL του δημοσιευμένου Reel")
    f.add_argument("--views", type=int, help="χειροκίνητα, αν λείπει το URL")
    f.add_argument("--likes", type=int)
    f.add_argument("--comments", type=int)
    f.add_argument("--followers", type=int)

    s = sub.add_parser("seed", help="τροφοδότηση μνήμης από benchmark accounts")
    s.add_argument("--creators", default="", help="λογαριασμοί χωρισμένοι με κόμμα")
    s.add_argument("--limit", type=int, default=12, help="πόσοι λογαριασμοί")
    s.add_argument("--budget", type=int, default=90, help="όριο κλήσεων API")
    s.add_argument("--niche", default="", help="ετικέτα niche για τη μνήμη")

    r = sub.add_parser("research", help="έρευνα λογαριασμού ή hashtag")
    r.add_argument("target", help="@username ή #hashtag")
    r.add_argument("--limit", type=int, default=15)
    r.add_argument("--budget", type=int, default=25)

    m = sub.add_parser("memory", help="κατάσταση και αναζήτηση μνήμης")
    m.add_argument("--query", default="", help="σημασιολογική αναζήτηση")
    m.add_argument("--limit", type=int, default=10)
    m.add_argument("--patterns", action="store_true", help="εμφάνιση μοτίβων")
    m.add_argument("--niche", default="")

    ru = sub.add_parser("runs", help="πρόσφατες εκτελέσεις")
    ru.add_argument("--limit", type=int, default=15)

    c = sub.add_parser("calibrate", help="αναφορά βαθμονόμησης βαρών")
    c.add_argument("--apply", action="store_true", help="εφαρμογή προτεινόμενων βαρών")
    return p


COMMANDS = {
    "serve": cmd_serve, "cleanup": cmd_cleanup, "init": cmd_init, "doctor": cmd_doctor, "analyze": cmd_analyze,
    "feedback": cmd_feedback, "seed": cmd_seed, "research": cmd_research,
    "memory": cmd_memory, "runs": cmd_runs, "balance": cmd_balance,
    "calibrate": cmd_calibrate, "verify-endpoints": cmd_verify_endpoints,
}


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    level = args.log_level or ("ERROR" if args.quiet else settings.log_level)
    setup_logging(level, settings.log_format,
                  settings.data_dir / "vrgr.log")
    try:
        return COMMANDS[args.command](args, settings)
    except KeyboardInterrupt:
        print("\nΔιακόπηκε.")
        return 130
    except VRGRError as exc:
        print(f"\n✗ {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
