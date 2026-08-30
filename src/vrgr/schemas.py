"""
Κανονικά μοντέλα δεδομένων.

Θεμελιώδης αρχή: κάθε αριθμός κουβαλά την ΠΡΟΕΛΕΥΣΗ του.
Το `Origin` δεν είναι διακοσμητικό — το report ΑΡΝΕΙΤΑΙ να παρουσιάσει
`INFERRED` τιμή ως γεγονός, και το σκορ τιμωρεί αποφάσεις με λίγα
`OBSERVED` στοιχεία (evidence multiplier).
"""
from __future__ import annotations

import hashlib
import time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Origin(str, Enum):
    OBSERVED = "OBSERVED"    # μετρήθηκε από HikerAPI — πραγματικό γεγονός
    DERIVED = "DERIVED"      # υπολογίστηκε ντετερμινιστικά από OBSERVED
    INFERRED = "INFERRED"    # εκτίμηση μοντέλου — ΔΕΝ είναι γεγονός


class Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# ═══════════════════════════════════════════════ HikerAPI → κανονικά

class Creator(Base):
    """Λογαριασμός Instagram, κανονικοποιημένος από οποιοδήποτε endpoint."""
    pk: str
    username: str
    full_name: str = ""
    followers: int = 0
    following: int = 0
    media_count: int = 0
    biography: str = ""
    is_private: bool = False
    is_verified: bool = False
    category: str = ""
    external_url: str = ""
    greek_confidence: float = 0.0
    fetched_at: float = Field(default_factory=time.time)
    origin: Origin = Origin.OBSERVED


class PostMetrics(Base):
    """Ωμοί μετρητές. `None` σημαίνει «δεν το δίνει το API» — ΟΧΙ μηδέν."""
    views: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None      # σχεδόν πάντα None σε ξένα posts
    saves: Optional[int] = None       # σχεδόν πάντα None σε ξένα posts
    plays: Optional[int] = None


class NormalizedMetrics(Base):
    """
    Μετρικές κανονικοποιημένες ως προς το μέγεθος λογαριασμού.

    Ο πυρήνας της απαίτησης #14: 1.5M views σε λογαριασμό 5K (V/F=300)
    διδάσκει πολλαπλάσια από 2M views σε λογαριασμό 10M (V/F=0.2).
    """
    vf_ratio: Optional[float] = None            # views / followers
    lf_ratio: Optional[float] = None            # likes / followers
    cf_ratio: Optional[float] = None            # comments / followers
    engagement_rate: Optional[float] = None     # (likes+comments) / views
    comment_rate: Optional[float] = None        # comments / views
    like_rate: Optional[float] = None           # likes / views
    comment_to_like: Optional[float] = None     # δείκτης «προκαλεί συζήτηση»
    viral_multiplier: Optional[float] = None    # views / διάμεσο του ίδιου creator
    outlier_score: Optional[float] = None       # 0–100 συνολικό «πόσο ξέφυγε»
    origin: Origin = Origin.DERIVED


class ObservedPost(Base):
    """Ένα πραγματικό post του Instagram όπως το είδαμε μια συγκεκριμένη στιγμή."""
    media_id: str
    code: str = ""
    creator_pk: str = ""
    username: str = ""
    followers_at_observation: Optional[int] = None
    caption: str = ""
    caption_body: str = ""
    hashtags: list = Field(default_factory=list)
    mentions: list = Field(default_factory=list)
    metrics: PostMetrics = Field(default_factory=PostMetrics)
    normalized: NormalizedMetrics = Field(default_factory=NormalizedMetrics)
    taken_at: Optional[int] = None
    duration_s: Optional[float] = None
    product_type: str = ""          # "clips" = Reel
    media_type: Optional[int] = None
    music_title: str = ""
    music_artist: str = ""
    is_original_audio: Optional[bool] = None
    location_name: str = ""
    language: str = ""
    greek_confidence: float = 0.0
    thumbnail_url: str = ""
    source_endpoint: str = ""
    observed_at: float = Field(default_factory=time.time)
    origin: Origin = Origin.OBSERVED

    @property
    def url(self) -> str:
        return f"https://www.instagram.com/reel/{self.code}/" if self.code else ""

    @property
    def age_days(self) -> Optional[float]:
        if not self.taken_at:
            return None
        return max(0.0, (time.time() - self.taken_at) / 86400.0)


class HashtagStat(Base):
    """
    Στοιχεία hashtag.

    ΠΡΟΣΟΧΗ: το HikerAPI δίνει ΜΟΝΟ `media_count` (OBSERVED).
    Δεν υπάρχει endpoint για «δυσκολία» ή «τάση» — αυτά τα υπολογίζουμε
    εμείς από δείγμα top posts και σημαίνονται ως DERIVED.
    """
    tag: str
    media_count: Optional[int] = None                 # OBSERVED
    media_count_origin: Origin = Origin.OBSERVED
    tier: str = "unknown"                             # broad|mid|niche|micro
    is_greek: bool = False
    sample_size: int = 0
    median_views_top: Optional[float] = None          # DERIVED
    median_followers_top: Optional[float] = None      # DERIVED
    small_account_share: Optional[float] = None       # DERIVED — «χωράει μικρός;»
    recency_days_median: Optional[float] = None       # DERIVED
    difficulty: Optional[float] = None                # DERIVED 0–100
    observed_at: float = Field(default_factory=time.time)


# ═══════════════════════════════════════════════ Ανάλυση βίντεο

class VideoTechnical(Base):
    """Ντετερμινιστικά μετρημένα — καμία συμμετοχή LLM."""
    path: str = ""
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    aspect_ratio: str = ""
    has_audio: bool = False
    size_bytes: int = 0
    scene_cuts: list = Field(default_factory=list)
    cut_count: int = 0
    cuts_per_second: float = 0.0
    avg_shot_len_s: Optional[float] = None
    hook_cut_count: int = 0            # κοψίματα στα πρώτα N δευτ.
    frame_paths: list = Field(default_factory=list)
    frame_times: list = Field(default_factory=list)
    origin: Origin = Origin.OBSERVED


class VideoContent(Base):
    """Τι δείχνει το βίντεο — έξοδος του vision μοντέλου (INFERRED)."""
    summary: str = ""
    main_subject: str = ""
    people_count: int = 0
    people_description: str = ""
    environment: str = ""
    actions: list = Field(default_factory=list)
    facial_expressions: str = ""
    emotions: list = Field(default_factory=list)
    mood: str = ""
    energy_level: str = ""             # low|medium|high
    visual_style: str = ""
    editing_style: str = ""
    pace: str = ""
    on_screen_text: list = Field(default_factory=list)
    spoken_transcript: str = ""
    audio_type: str = ""               # speech|music|both|silent
    hook_description: str = ""
    story_arc: str = ""
    humor: str = ""
    aesthetic: str = ""
    niche: str = ""
    sub_niche: str = ""
    target_audience: str = ""
    potential_audience: str = ""
    cultural_markers: list = Field(default_factory=list)
    origin: Origin = Origin.INFERRED


class ViralSignals(Base):
    """Δείκτες 0–100 για το δυναμικό του βίντεο — INFERRED."""
    curiosity_gap: int = 0
    emotional_trigger: int = 0
    relatability: int = 0
    shareability: int = 0
    commentability: int = 0
    save_potential: int = 0
    rewatch_potential: int = 0
    shock_factor: int = 0
    attention_hold: int = 0
    greek_cultural_fit: int = 0
    notes: str = ""
    origin: Origin = Origin.INFERRED

    def mean(self) -> float:
        vals = [self.curiosity_gap, self.emotional_trigger, self.relatability,
                self.shareability, self.commentability, self.save_potential,
                self.rewatch_potential, self.attention_hold, self.greek_cultural_fit]
        return sum(vals) / len(vals)


class ViralAngle(Base):
    """
    Μια υποψήφια «γωνία» — ο λόγος που κάποιος θα αντιδράσει.

    Δεν είναι περιγραφή του βίντεο· είναι ΣΤΡΑΤΗΓΙΚΗ απόφαση για το
    τι πρέπει να προσθέσει η λεζάντα ώστε να ενεργοποιηθεί αντίδραση.
    """
    name: str
    strategy: str                       # curiosity|relatability|humor|...
    rationale: str = ""
    why_greek_stops: str = ""           # γιατί σταματά το scroll Έλληνας χρήστης
    why_comment: str = ""
    why_share: str = ""
    caption_should_add: str = ""        # τι ΔΕΝ δείχνει το βίντεο
    target_segment: str = ""
    strength: int = 0                   # 0–100
    risk: str = ""
    origin: Origin = Origin.INFERRED


class VideoAnalysis(Base):
    technical: VideoTechnical = Field(default_factory=VideoTechnical)
    content: VideoContent = Field(default_factory=VideoContent)
    signals: ViralSignals = Field(default_factory=ViralSignals)
    angles: list = Field(default_factory=list)          # list[ViralAngle]
    chosen_angle: Optional[ViralAngle] = None
    analysis_notes: list = Field(default_factory=list)


# ═══════════════════════════════════════════════ Έρευνα

class ResearchQuery(Base):
    kind: str                    # hashtag|keyword|creator|location|music
    value: str
    reason: str = ""
    priority: int = 5            # 1 = υψηλότερη
    estimated_calls: int = 1


class ResearchBundle(Base):
    """Ό,τι μαζεύτηκε από HikerAPI σε μία εκτέλεση."""
    run_id: str = ""
    queries: list = Field(default_factory=list)
    posts: list = Field(default_factory=list)            # list[ObservedPost]
    creators: dict = Field(default_factory=dict)         # pk -> Creator
    hashtag_stats: dict = Field(default_factory=dict)    # tag -> HashtagStat
    outliers: list = Field(default_factory=list)         # list[ObservedPost]
    greek_posts: int = 0
    api_calls: int = 0
    cache_hits: int = 0
    errors: list = Field(default_factory=list)
    degraded: bool = False
    degraded_reason: str = ""


class MinedPatterns(Base):
    """Μοτίβα εξαγμένα από το corpus — DERIVED, όχι γνώμη μοντέλου."""
    sample_size: int = 0
    outlier_sample_size: int = 0
    caption_len_median: Optional[float] = None
    caption_len_p25: Optional[float] = None
    caption_len_p75: Optional[float] = None
    emoji_median: Optional[float] = None
    hashtag_count_median: Optional[float] = None
    question_share: Optional[float] = None
    cta_share: Optional[float] = None
    first_person_share: Optional[float] = None
    second_person_share: Optional[float] = None
    top_words: list = Field(default_factory=list)        # [(word, lift, n)]
    top_bigrams: list = Field(default_factory=list)
    top_hashtags: list = Field(default_factory=list)     # [(tag, lift, n)]
    hashtag_cooccurrence: list = Field(default_factory=list)
    greek_expressions: list = Field(default_factory=list)
    posting_hours: list = Field(default_factory=list)
    duration_sweet_spot: Optional[list] = None
    origin: Origin = Origin.DERIVED


# ═══════════════════════════════════════════════ Παραγωγή

class CaptionCandidate(Base):
    id: str = ""
    text: str
    strategy: str
    angle_name: str = ""
    rationale: str = ""
    hook_line: str = ""
    length_chars: int = 0
    word_count: int = 0
    emoji_count: int = 0
    has_question: bool = False
    has_cta: bool = False
    greek_ratio: float = 0.0
    similarity_to_corpus: float = 0.0     # anti-plagiarism
    origin: Origin = Origin.INFERRED

    def fill(self) -> "CaptionCandidate":
        from . import greek as _g
        self.length_chars = len(self.text)
        self.word_count = len(self.text.split())
        self.emoji_count = _g.count_emoji(self.text)
        # «;» είναι το ελληνικό ερωτηματικό — όχι απλό σημείο στίξης
        self.has_question = ("?" in self.text) or (";" in self.text)
        self.greek_ratio = _g.greek_ratio(self.text)
        if not self.id:
            self.id = hashlib.sha1(self.text.encode("utf-8")).hexdigest()[:10]
        if not self.hook_line:
            self.hook_line = self.text.strip().split("\n")[0][:80]
        return self


class HashtagCandidate(Base):
    tag: str
    tier: str = "unknown"
    category: str = ""            # topic|audience|content|greek|location|trend|community
    relevance: float = 0.0
    evidence_count: int = 0       # σε πόσα επιτυχημένα συγκρίσιμα posts εμφανίζεται
    media_count: Optional[int] = None
    difficulty: Optional[float] = None
    is_greek: bool = False
    score: float = 0.0
    reason: str = ""
    origin: Origin = Origin.DERIVED


class HashtagSet(Base):
    id: str = ""
    tags: list = Field(default_factory=list)          # list[str]
    candidates: list = Field(default_factory=list)    # list[HashtagCandidate]
    strategy: str = ""
    tier_distribution: dict = Field(default_factory=dict)
    greek_share: float = 0.0
    evidence_share: float = 0.0
    score: float = 0.0
    rationale: str = ""

    def fill(self) -> "HashtagSet":
        if not self.id:
            self.id = hashlib.sha1(",".join(self.tags).encode("utf-8")).hexdigest()[:10]
        return self


# ═══════════════════════════════════════════════ Σκοράρισμα

class PillarScore(Base):
    name: str
    label_el: str
    raw: float                     # 0–100
    weight: float
    weighted: float
    evidence_n: int = 0
    origin: Origin = Origin.INFERRED
    note: str = ""


class ViralScore(Base):
    total: float = 0.0                    # 0–100 μετά τον evidence multiplier
    raw_total: float = 0.0                # πριν
    evidence_multiplier: float = 1.0
    confidence: str = "χαμηλή"            # χαμηλή|μεσαία|υψηλή
    interval: list = Field(default_factory=list)      # [low, high]
    pillars: list = Field(default_factory=list)       # list[PillarScore]
    observed_posts_used: int = 0
    greek_posts_used: int = 0
    notes: list = Field(default_factory=list)


class ScoredCombo(Base):
    """Ζεύγος λεζάντας × σετ hashtags — η μονάδα βελτιστοποίησης (#11)."""
    caption: CaptionCandidate
    hashtag_set: HashtagSet
    score: ViralScore
    synergy: float = 0.0
    rank: int = 0


class EvidenceItem(Base):
    """Μια συγκεκριμένη παραπομπή σε πραγματικό post."""
    username: str
    url: str = ""
    followers: Optional[int] = None
    views: Optional[int] = None
    vf_ratio: Optional[float] = None
    caption_excerpt: str = ""
    hashtags: list = Field(default_factory=list)
    age_days: Optional[float] = None
    why_relevant: str = ""
    origin: Origin = Origin.OBSERVED


class AnalysisResult(Base):
    """Το τελικό προϊόν μιας εκτέλεσης."""
    run_id: str
    created_at: float = Field(default_factory=time.time)
    video: VideoAnalysis = Field(default_factory=VideoAnalysis)
    research: Optional[ResearchBundle] = None
    patterns: Optional[MinedPatterns] = None
    winner: Optional[ScoredCombo] = None
    backups: list = Field(default_factory=list)          # list[ScoredCombo]
    backup_hashtag_sets: list = Field(default_factory=list)
    evidence: list = Field(default_factory=list)         # list[EvidenceItem]
    why_won: str = ""
    memory_hits: int = 0
    warnings: list = Field(default_factory=list)
    data_gaps: list = Field(default_factory=list)
    duration_s: float = 0.0
    api_calls: int = 0


class Pattern(Base):
    """
    Κανόνας με Beta(α, β) βεβαιότητα.

    Beta αντί για σκέτο ποσοστό: το 2/2 (=100%) και το 41/60 (=68%) δεν είναι
    το ίδιο πράγμα, και το σύστημα πρέπει να ξέρει τη διαφορά.
    """
    key: str
    kind: str                      # caption_structure|hashtag|hook|timing|...
    niche: str = ""
    description_el: str = ""
    alpha: float = 1.0
    beta: float = 1.0
    last_seen: float = Field(default_factory=time.time)
    sample_ids: list = Field(default_factory=list)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def n(self) -> float:
        return self.alpha + self.beta - 2.0

    @property
    def confidence(self) -> float:
        """Πόσο εμπιστευόμαστε τη μέση τιμή — αντιστρόφως ανάλογο της διασποράς."""
        a, b = self.alpha, self.beta
        var = (a * b) / (((a + b) ** 2) * (a + b + 1))
        return round(max(0.0, 1.0 - (var ** 0.5) * 3.4641), 3)

    def lower_bound(self) -> float:
        """Συντηρητική εκτίμηση (~5ο εκατοστημόριο) — αυτή χρησιμοποιεί το σκορ."""
        a, b = self.alpha, self.beta
        m = a / (a + b)
        sd = ((a * b) / (((a + b) ** 2) * (a + b + 1))) ** 0.5
        return max(0.0, m - 1.645 * sd)
