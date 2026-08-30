"""
VRGR — διεπαφή Streamlit.

    streamlit run streamlit_app.py

Δεν υπάρχει διπλή λογική: όλη η δουλειά γίνεται από το ίδιο `Pipeline` και
`JobManager` που χρησιμοποιεί και η stdlib εφαρμογή. Εδώ είναι μόνο η
παρουσίαση.

ΤΟ ΜΟΝΤΕΛΟ ΕΚΤΕΛΕΣΗΣ ΤΟΥ STREAMLIT: το script ξανατρέχει από την αρχή σε κάθε
αλληλεπίδραση. Μια ανάλυση 5 λεπτών ΔΕΝ μπορεί να τρέξει μέσα στο script —
τρέχει σε νήμα (ο `JobManager` το κάνει ήδη) και η πρόοδος διαβάζεται από ένα
`@st.fragment` που ανανεώνεται μόνο του, χωρίς να ξαναχτίζει όλη τη σελίδα.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vrgr.config import load_settings                       # noqa: E402
from vrgr.logging_setup import setup_logging                # noqa: E402

st.set_page_config(page_title="VRGR — Viral Intelligence GR",
                   page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

UPLOAD_CHUNK = 4 * 1024 * 1024


# ── πόροι που επιβιώνουν των reruns ───────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_settings():
    s = load_settings()
    setup_logging(s.log_level, s.log_format, s.data_dir / "vrgr.log")
    # Σε εφήμερο filesystem η βάση λείπει μετά από κάθε restart: την
    # ξεκινάμε από το αντίγραφο που ζει μέσα στο repo.
    from vrgr.seedstore import restore_if_missing
    st.session_state["_restored"] = restore_if_missing(s.db_path)
    return s


@st.cache_resource(show_spinner=False)
def get_jobs():
    """
    Ένας JobManager για όλη τη ζωή της διεργασίας.

    Χωρίς `cache_resource` θα χτιζόταν νέο Pipeline —δηλαδή νέα σύνδεση βάσης,
    νέος embedder, νέος HTTP client— σε ΚΑΘΕ κλικ.
    """
    from vrgr.web.jobs import JobManager
    return JobManager(get_settings())


@st.cache_data(ttl=120, show_spinner=False)
def cached_status() -> dict:
    s, jobs = get_settings(), get_jobs()
    pipe = jobs.pipeline()
    balance = pipe.hiker.balance() if pipe.hiker is not None else None
    from vrgr.maintenance import usage
    return {"memory": pipe.retriever.corpus_stats(),
            "patterns": pipe.patterns.stats(),
            "balance": balance, "disk": usage(s),
            "hiker": s.hiker.enabled, "anthropic": s.models.enabled,
            "asr": s.video.asr_provider}


def save_upload(uploaded) -> Path:
    """Γράφει το ανεβασμένο βίντεο στον δίσκο σε κομμάτια."""
    s = get_settings()
    up_dir = s.data_dir / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    import re
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(uploaded.name).name)[-80:]
    dest = up_dir / f"{int(time.time())}_{safe}"
    with open(dest, "wb") as fh:
        while True:
            chunk = uploaded.read(UPLOAD_CHUNK)
            if not chunk:
                break
            fh.write(chunk)
    return dest


# ── sidebar ───────────────────────────────────────────────────────────

def sidebar() -> None:
    st.sidebar.markdown("### VRGR · Viral Intelligence **GR**")
    try:
        info = cached_status()
    except Exception as exc:                        # noqa: BLE001
        st.sidebar.error(f"Αδύνατη ανάγνωση κατάστασης: {type(exc).__name__}")
        return

    c1, c2 = st.sidebar.columns(2)
    c1.metric("HikerAPI", "✓" if info["hiker"] else "✗")
    c2.metric("Anthropic", "✓" if info["anthropic"] else "✗")
    if not info["anthropic"]:
        st.sidebar.error("Λείπει το ANTHROPIC_API_KEY — η ανάλυση δεν θα δουλέψει.")
    if not info["hiker"]:
        st.sidebar.warning("Λείπει το HIKER_API_KEY — δεν θα γίνεται έρευνα δεδομένων.")

    b = info.get("balance") or {}
    if b.get("requests") is not None:
        st.sidebar.metric("Credits HikerAPI", f"{b['requests']:,}".replace(",", "."))

    m = info["memory"]
    st.sidebar.markdown("**Μνήμη**")
    st.sidebar.caption(
        f"{m['posts']:,} posts · {m['greek_posts']:,} ελληνικά · "
        f"{m['outliers']:,} outliers · {info['patterns']['patterns']} μοτίβα "
        f"({info['patterns']['usable']} αξιοποιήσιμα)".replace(",", "."))

    disk = info["disk"]
    st.sidebar.caption(f"Δίσκος: {disk['σύνολο']/1e6:.0f} MB "
                       f"(γνώση {disk['βάση_γνώσης']/1e6:.0f} MB)")
    if info["asr"] == "none":
        st.sidebar.caption("⚠ Χωρίς μεταγραφή ήχου — η ανάλυση βασίζεται σε καρέ.")

    memory_backup_controls()

    if st.sidebar.button("Καθαρισμός προσωρινών", use_container_width=True):
        from vrgr.maintenance import cleanup
        pipe = get_jobs().pipeline()
        freed = cleanup(get_settings(), cache=getattr(pipe, "_cache", None))
        cached_status.clear()
        st.sidebar.success(f"Ελευθερώθηκαν {freed['bytes']/1e6:.1f} MB")


def memory_backup_controls() -> None:
    """
    Λήψη και επαναφορά της μνήμης.

    Σε εφήμερο περιβάλλον αυτό ΔΕΝ είναι πολυτέλεια: είναι ο μόνος τρόπος να
    κρατήσεις ό,τι έμαθε το σύστημα. Κατεβάζεις τη βάση, την κάνεις commit
    στο `seed/vrgr_seed.db`, και η γνώση επιβιώνει του επόμενου restart.
    """
    from vrgr.seedstore import export_db, import_db, is_ephemeral
    s = get_settings()

    if is_ephemeral():
        st.sidebar.warning(
            "**Εφήμερος δίσκος.** Ό,τι μαθαίνει το σύστημα χάνεται στο επόμενο "
            "restart (ύπνος 12 ωρών ή redeploy). Κατέβασε αντίγραφο και κάν' το "
            "commit στο `seed/vrgr_seed.db` για να μείνει.", icon="⚠️")
        note = st.session_state.get("_restored")
        if note:
            st.sidebar.caption(note)

    with st.sidebar.expander("Αντίγραφο μνήμης"):
        try:
            tmp = s.data_dir / "memory_backup.db"
            export_db(s.db_path, tmp)
            st.download_button(
                f"Κατέβασε ({tmp.stat().st_size/1e6:.0f} MB)",
                data=tmp.read_bytes(), file_name="vrgr_seed.db",
                mime="application/octet-stream", use_container_width=True)
        except Exception as exc:                     # noqa: BLE001
            st.caption(f"Αδύνατη η εξαγωγή: {type(exc).__name__}")
        restore = st.file_uploader("Επαναφορά από αρχείο", type=["db"],
                                   key="restore_db")
        if restore is not None and st.button("Επαναφορά", use_container_width=True):
            try:
                msg = import_db(restore.getvalue(), s.db_path)
            except Exception as exc:                 # noqa: BLE001
                st.error(str(exc))
            else:
                get_jobs.clear()
                cached_status.clear()
                st.success(msg)
                st.rerun()


# ── καρτέλα: Ανάλυση ──────────────────────────────────────────────────

def tab_analyze() -> None:
    jobs = get_jobs()
    active = st.session_state.get("job_id")

    if not active:
        st.subheader("Ανάλυση Reel")
        st.caption("Το σύστημα αναλύει το βίντεο, ερευνά πραγματικά ελληνικά "
                   "δεδομένα Instagram, και **αποφασίζει** λεζάντα και hashtags.")
        uploaded = st.file_uploader(
            "Βίντεο", type=["mp4", "mov", "m4v", "webm", "avi", "mkv"],
            label_visibility="collapsed")
        c1, c2, c3 = st.columns([2, 2, 1])
        context = c1.text_input("Συμφραζόμενα (προαιρετικά)",
                                placeholder="π.χ. Reel για τη σελίδα μαγειρικής μου")
        creators = c2.text_input("Benchmark λογαριασμοί (προαιρετικά)",
                                 placeholder="user1, user2, user3")
        n_caps = c3.selectbox("Λεζάντες", [6, 8, 10], index=1)
        no_research = st.checkbox(
            "Χωρίς έρευνα HikerAPI (γρήγορο, αλλά χαμηλό σκορ λόγω έλλειψης τεκμηρίων)")

        if uploaded is not None:
            st.info(f"🎬 **{uploaded.name}** · {uploaded.size/1e6:.1f} MB")
        if st.button("Έναρξη ανάλυσης", type="primary", disabled=uploaded is None,
                     use_container_width=True):
            path = save_upload(uploaded)
            job = jobs.start_analysis(path, {
                "context": context,
                "captions": n_caps,
                "no_research": no_research,
                "creators": [c.strip().lstrip("@") for c in creators.split(",")
                             if c.strip()] or None,
            })
            st.session_state["job_id"] = job.id
            st.rerun()
        return

    job = jobs.get(active)
    if job is None:
        st.session_state.pop("job_id", None)
        st.rerun()
        return

    if job.status in ("queued", "running"):
        live_progress(active)
        return

    if job.status == "error":
        st.error(f"**Η ανάλυση απέτυχε**\n\n{job.error}")
        if st.button("Νέα προσπάθεια"):
            st.session_state.pop("job_id", None)
            st.rerun()
        return

    render_result(job.result)
    if st.button("Νέα ανάλυση", type="primary"):
        st.session_state.pop("job_id", None)
        cached_status.clear()
        st.rerun()


STEP_NAMES = ["Τεχνική ανάλυση βίντεο", "Οπτική ανάλυση καρέ με AI",
              "Επιλογή viral γωνίας", "Σχεδιασμός ελληνικής έρευνας",
              "Έρευνα HikerAPI", "Αποθήκευση στη μνήμη",
              "Αναζήτηση ιστορικών αναλόγων", "Εξόρυξη μοτίβων",
              "Παραγωγή λεζαντών με AI", "Κατασκευή χαρτοφυλακίων hashtags",
              "Σκοράρισμα συνδυασμών", "Τεκμήρια και μάθηση"]


@st.fragment(run_every=2)
def live_progress(job_id: str) -> None:
    """
    Ανανεώνεται μόνο του κάθε 2 δευτερόλεπτα.

    Ως fragment ξαναχτίζει ΜΟΝΟ αυτό το κομμάτι — όχι όλη τη σελίδα, που θα
    ξανάτρεχε κάθε query και θα έκανε τη διεπαφή να τρεμοπαίζει.
    """
    job = get_jobs().get(job_id)
    if job is None:
        return
    st.subheader("Σε εξέλιξη…")
    st.progress(min(1.0, job.step / max(1, job.total)),
                text=f"{job.step_label}"
                     + (f" — {job.step_detail}" if job.step_detail else ""))
    mins, secs = divmod(int(job.elapsed), 60)
    st.caption(f"Βήμα {job.step}/{job.total} · {mins}:{secs:02d} · "
               f"η πλήρης ανάλυση διαρκεί συνήθως 3–5 λεπτά")
    for i, name in enumerate(STEP_NAMES, start=1):
        if i < job.step:
            st.markdown(f"✅ &nbsp;{name}", unsafe_allow_html=True)
        elif i == job.step:
            st.markdown(f"🔄 &nbsp;**{name}**", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:#5f6874'>⬜ &nbsp;{name}</span>",
                        unsafe_allow_html=True)
    if job.status in ("done", "error"):
        st.rerun()


def render_result(result) -> None:
    if result is None or result.winner is None:
        st.error("Δεν παρήχθη αποτέλεσμα.")
        return
    w = result.winner
    sc = w.score
    tags = " ".join("#" + t for t in w.hashtag_set.tags)

    st.subheader("Η απόφαση")
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Viral Score", f"{sc.total:.0f}/100",
              help="Σχετική κατάταξη, όχι πρόβλεψη προβολών.")
    c2.metric("Βεβαιότητα", sc.confidence,
              help=f"Εύρος {sc.interval[0]:.0f}–{sc.interval[1]:.0f}")
    if result.research:
        c3.metric("Τεκμήρια",
                  f"{result.research.greek_posts} ελληνικά posts",
                  f"{len(result.research.outliers)} outliers")
    if sc.evidence_multiplier < 1.0:
        st.warning(f"Το σκορ περιορίστηκε σε ×{sc.evidence_multiplier:.2f} "
                   f"λόγω περιορισμένων τεκμηρίων.")

    st.markdown("**Λεζάντα**")
    st.code(w.caption.text, language=None, wrap_lines=True)
    st.caption(f"{w.caption.strategy} · {w.caption.length_chars} χαρακτ. · "
               f"{w.caption.emoji_count} emoji")

    st.markdown("**Hashtags**")
    st.code(tags, language=None, wrap_lines=True)
    dist = " · ".join(f"{k}: {v}" for k, v in sorted(w.hashtag_set.tier_distribution.items()))
    st.caption(f"Χαρτοφυλάκιο «{w.hashtag_set.strategy}» — {dist} · "
               f"ελληνικά {w.hashtag_set.greek_share:.0%}")

    st.markdown("**Έτοιμο για επικόλληση**")
    st.code(w.caption.text + "\n\n" + tags, language=None, wrap_lines=True)

    with st.expander("Γιατί κέρδισε αυτός ο συνδυασμός", expanded=True):
        st.write(result.why_won)
        ang = result.video.chosen_angle
        if ang:
            st.markdown(f"**Γωνία «{ang.name}»** ({ang.strategy}, ισχύς {ang.strength})")
            st.markdown(
                f"- **Γιατί σταματά το scroll:** {ang.why_greek_stops}\n"
                f"- **Γιατί σχολιάζει:** {ang.why_comment}\n"
                f"- **Γιατί το στέλνει:** {ang.why_share}\n"
                f"- **Τι προσθέτει η λεζάντα:** {ang.caption_should_add}")
        st.dataframe(
            [{"Πυλώνας": p.label_el, "Σκορ": round(p.raw),
              "Βάρος": p.weight, "Συνεισφορά": p.weighted}
             for p in sorted(sc.pillars, key=lambda x: -x.weighted)],
            hide_index=True, use_container_width=True)

    if result.evidence:
        with st.expander(f"Τεκμήρια από {len(result.evidence)} πραγματικά posts"):
            for e in result.evidence:
                vf = f"V/F {e.vf_ratio}×" if e.vf_ratio else "V/F —"
                head = f"**@{e.username}** — {e.views or 0:,} προβολές / " \
                       f"{e.followers or 0:,} followers · {vf}".replace(",", ".")
                if e.age_days is not None:
                    head += f" · {e.age_days:.0f} ημ."
                st.markdown(head)
                if e.caption_excerpt:
                    st.caption(f"«{e.caption_excerpt}»")
                st.caption(e.why_relevant)
                if e.url:
                    st.caption(e.url)
                st.divider()

    if result.backups:
        with st.expander(f"Εναλλακτικές λεζάντες ({len(result.backups)})"):
            for i, b in enumerate(result.backups, start=2):
                st.markdown(f"**{i}. {b.score.total:.0f}/100 · {b.caption.strategy}**")
                st.code(b.caption.text, language=None, wrap_lines=True)

    if result.backup_hashtag_sets:
        with st.expander("Εναλλακτικά σετ hashtags"):
            for b in result.backup_hashtag_sets:
                st.markdown(f"**«{b.hashtag_set.strategy}» · {b.score.total:.0f}/100**")
                st.code(" ".join("#" + t for t in b.hashtag_set.tags),
                        language=None, wrap_lines=True)
                st.caption(b.hashtag_set.rationale)

    gaps = list(result.warnings) + list(result.data_gaps)
    if gaps:
        with st.expander("Περιορισμοί και κενά δεδομένων"):
            for g in gaps:
                st.caption(f"· {g}")

    st.caption(f"Εκτέλεση `{result.run_id}` · {result.duration_s}s · "
               f"{result.api_calls} κλήσεις API · "
               f"μετά τη δημοσίευση κατέγραψε το αποτέλεσμα στο **Ιστορικό**.")


# ── καρτέλα: Ιστορικό ─────────────────────────────────────────────────

def tab_history() -> None:
    repo = get_jobs().pipeline().repo
    runs = repo.recent_runs(30)
    if not runs:
        st.info("Καμία ανάλυση ακόμη.")
        return
    st.caption("Μετά τη δημοσίευση, κατέγραψε το πραγματικό αποτέλεσμα — "
               "εκεί μαθαίνει το σύστημα.")
    for r in runs:
        when = time.strftime("%d/%m %H:%M", time.localtime(r["created_at"]))
        head = (f"{when} · {r['predicted_score'] or 0:.0f}/100 "
                f"({r['confidence']}) — {(r['caption'] or '')[:60]}…")
        with st.expander(head):
            st.code(r["caption"] or "", language=None, wrap_lines=True)
            import json
            try:
                st.code(" ".join("#" + t for t in json.loads(r["hashtags_json"])),
                        language=None, wrap_lines=True)
            except Exception:                        # noqa: BLE001
                pass
            st.caption(f"{r['niche']} · {r['angle_name']}")
            feedback_form(r["run_id"])


def feedback_form(run_id: str) -> None:
    with st.form(f"fb_{run_id}"):
        st.markdown("**Κατέγραψε το πραγματικό αποτέλεσμα**")
        url = st.text_input("URL δημοσιευμένου Reel",
                            placeholder="https://instagram.com/reel/…",
                            key=f"u_{run_id}")
        c1, c2, c3, c4 = st.columns(4)
        views = c1.number_input("Προβολές", min_value=0, step=1000, key=f"v_{run_id}")
        followers = c2.number_input("Followers", min_value=0, step=100, key=f"f_{run_id}")
        likes = c3.number_input("Likes", min_value=0, step=100, key=f"l_{run_id}")
        comments = c4.number_input("Σχόλια", min_value=0, step=10, key=f"c_{run_id}")
        st.caption("Με URL τα νούμερα έρχονται αυτόματα από το HikerAPI.")
        if st.form_submit_button("Καταγραφή"):
            from vrgr.learning.feedback import FeedbackLoop
            pipe = get_jobs().pipeline()
            loop = FeedbackLoop(pipe.repo, pipe.patterns, pipe.hiker)
            manual = {k: int(v) for k, v in
                      (("views", views), ("followers", followers),
                       ("likes", likes), ("comments", comments)) if v}
            try:
                out = loop.record(run_id, url, manual or None)
            except Exception as exc:                 # noqa: BLE001
                st.error(str(exc))
                return
            summary = loop.summary()
            st.success(
                f"Καταγράφηκε. V/F **{out['vf_ratio']:.1f}×** · "
                f"πρόβλεψη {out['predicted'] or 0:.0f} vs πραγματικό "
                f"{out['actual'] or 0:.0f} · {out['patterns_updated']} μοτίβα "
                f"ενημερώθηκαν."
                + (f"\n\nΣυσχέτιση πρόβλεψης–πραγματικότητας: "
                   f"**{summary['correlation']}** σε {summary['n_comparable']} μετρήσεις."
                   if summary.get("correlation") is not None else ""))
            cached_status.clear()


# ── καρτέλα: Έρευνα ───────────────────────────────────────────────────

def tab_research() -> None:
    st.caption("Δες τι πετυχαίνει ένας λογαριασμός ή ένα hashtag — με V/F, "
               "δηλαδή απόδοση σε σχέση με το μέγεθος του λογαριασμού.")
    c1, c2 = st.columns([4, 1])
    target = c1.text_input("Στόχος", placeholder="@λογαριασμός ή #hashtag",
                           label_visibility="collapsed")
    go = c2.button("Έρευνα", use_container_width=True)
    if not (go and target.strip()):
        return
    jobs = get_jobs()
    with st.spinner(f"Έρευνα «{target}» — μπορεί να πάρει ένα λεπτό…"):
        job = jobs.start_research(target.strip())
        while job.status in ("queued", "running"):
            time.sleep(1.0)
    if job.status == "error":
        st.error(job.error)
        return
    render_research(job.result)
    cached_status.clear()


def render_research(d: dict) -> None:
    if d["kind"] == "creator":
        c, s = d["creator"], d["summary"]
        st.subheader(f"@{c['username']}")
        cols = st.columns(4)
        cols[0].metric("Followers", f"{c['followers']:,}".replace(",", "."))
        cols[1].metric("Ελληνικότητα", f"{c['greek_confidence']:.0%}")
        cols[2].metric("Διάμεσο προβολών",
                       f"{s['median_views'] or 0:,}".replace(",", "."))
        cols[3].metric("Καλύτερο V/F", f"{s['max_vf'] or 0}×")
    else:
        st.subheader(f"#{d['target']}")
        stat = d.get("stat") or {}
        cols = st.columns(4)
        cols[0].metric("Μέγεθος",
                       f"{stat.get('media_count') or 0:,}".replace(",", "."))
        cols[1].metric("Επίπεδο", stat.get("tier") or "—")
        cols[2].metric("Δυσκολία",
                       f"{stat['difficulty']:.0f}/100" if stat.get("difficulty") else "—",
                       help="Υπολογισμένη από εμάς — το API δεν τη δίνει.")
        share = stat.get("small_account_share")
        cols[3].metric("Μικροί λογαριασμοί",
                       f"{share:.0%}" if share is not None else "—",
                       help="Ποσοστό των κορυφαίων με <50K followers — "
                            "«χωράει μικρός εδώ;»")
        if d.get("trend"):
            st.caption(f"Τάση: **{d['trend']['label']}** "
                       f"(recent/top = {d['trend']['ratio']})")
    rows = [{"Λογαριασμός": "@" + p["username"], "Προβολές": p["views"],
             "Followers": p["followers"],
             "V/F": f"{p['vf_ratio']}×" if p["vf_ratio"] else "—",
             "Σκορ": round(p["outlier_score"]) if p["outlier_score"] else None,
             "Λεζάντα": (p["caption"] or "")[:80]} for p in d["posts"]]
    st.dataframe(rows, hide_index=True, use_container_width=True)


# ── καρτέλα: Μνήμη ────────────────────────────────────────────────────

def tab_memory() -> None:
    pipe = get_jobs().pipeline()
    stats = pipe.retriever.corpus_stats()
    pstats = pipe.patterns.stats()
    cols = st.columns(5)
    cols[0].metric("Posts", f"{stats['posts']:,}".replace(",", "."))
    cols[1].metric("Ελληνικά", f"{stats['greek_posts']:,}".replace(",", "."))
    cols[2].metric("Στιγμιότυπα", f"{stats['snapshots']:,}".replace(",", "."))
    cols[3].metric("Outliers", f"{stats['outliers']:,}".replace(",", "."))
    cols[4].metric("Μοτίβα", f"{pstats['patterns']}",
                   f"{pstats['usable']} αξιοποιήσιμα")

    q = st.text_input("Σημασιολογική αναζήτηση",
                      placeholder="π.χ. «περιμένω μήνυμα»")
    if q.strip():
        for row in pipe.retriever.search(q.strip(), limit=15):
            st.markdown(
                f"**@{row['username']}** · {(row.get('views') or 0):,} προβολές"
                .replace(",", ".")
                + (f" · V/F {row['vf_ratio']:.1f}×" if row.get("vf_ratio") else "")
                + f" · ομοιότητα {row.get('similarity', 0):.2f}")
            st.caption(f"«{(row.get('caption_body') or '')[:160]}»")
            if row.get("hashtags"):
                st.caption(" ".join("#" + t for t in row["hashtags"][:10]))
            st.divider()

    st.markdown("### Μοτίβα που έχει μάθει")
    rows = []
    for kind in ("caption_structure", "hashtag"):
        for p in pipe.patterns.by_kind(kind, "", limit=15):
            rows.append({"Μοτίβο": p.description_el or p.key, "Είδος": kind,
                         "Δείγματα": round(p.n, 1), "Μέσο": round(p.mean, 3),
                         "Κάτω φράγμα": round(p.lower_bound(), 3)})
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.caption("Κανένα μοτίβο με αρκετά δείγματα ακόμη — χρειάζονται ≥4 "
                   "επιβεβαιώσεις. Χτίζεται με κάθε ανάλυση και κάθε καταγραφή "
                   "αποτελέσματος.")


# ── main ──────────────────────────────────────────────────────────────

def main() -> None:
    sidebar()
    t1, t2, t3, t4 = st.tabs(["Ανάλυση", "Ιστορικό", "Έρευνα", "Μνήμη"])
    with t1:
        tab_analyze()
    with t2:
        tab_history()
    with t3:
        tab_research()
    with t4:
        tab_memory()


main()
