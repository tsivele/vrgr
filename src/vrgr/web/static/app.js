'use strict';

// ── βοηθητικά ────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => { const n = document.createElement(tag);
  if (cls) n.className = cls; if (html !== undefined) n.innerHTML = html; return n; };
const esc = (s) => String(s ?? '').replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (n) => (n === null || n === undefined) ? '—'
  : Number(n).toLocaleString('el-GR');
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  const d = await r.json().catch(() => ({ error: 'Μη έγκυρη απόκριση' }));
  if (!r.ok) throw new Error(d.error || `Σφάλμα ${r.status}`);
  return d;
};
function copy(text, btn) {
  navigator.clipboard.writeText(text).then(() => {
    const old = btn.textContent; btn.textContent = 'Αντιγράφηκε ✓';
    setTimeout(() => btn.textContent = old, 1600);
  });
}
function copyBox(title, body, isTags) {
  const box = el('div', 'copybox');
  const b = el('button', 'ghost', 'Αντιγραφή');
  b.onclick = () => copy(body, b);
  box.appendChild(b);
  box.appendChild(isTags ? el('div', 'tags', esc(body)) : el('pre', '', esc(body)));
  return box;
}

// ── καρτέλες ─────────────────────────────────────────────────────────
document.querySelectorAll('nav button').forEach(b => b.onclick = () => {
  document.querySelectorAll('nav button').forEach(x => x.classList.toggle('on', x === b));
  ['analyze','history','research','memory'].forEach(t =>
    $('tab-' + t).classList.toggle('hide', t !== b.dataset.tab));
  if (b.dataset.tab === 'history') loadRuns();
  if (b.dataset.tab === 'memory') loadMemory();
});

// ── κατάσταση συστήματος ─────────────────────────────────────────────
async function loadStatus() {
  try {
    const s = await api('/api/status');
    const c = $('chips'); c.innerHTML = '';
    const chip = (label, val, cls) =>
      c.appendChild(el('span', 'chip ' + (cls || ''), `${label} <b>${val}</b>`));
    chip('HikerAPI', s.keys.hiker ? '✓' : '✗', s.keys.hiker ? 'ok' : 'err');
    chip('Anthropic', s.keys.anthropic ? '✓' : '✗', s.keys.anthropic ? 'ok' : 'err');
    if (s.balance && s.balance.requests !== undefined)
      chip('Credits', num(s.balance.requests));
    chip('Μνήμη', `${num(s.memory.posts)} posts`);
    chip('Ελληνικά', num(s.memory.greek_posts));
    chip('Outliers', num(s.memory.outliers));
    if (s.asr === 'none')
      chip('Ήχος', 'χωρίς μεταγραφή');
  } catch (e) { /* η μπάρα κατάστασης δεν πρέπει να σπάει τη σελίδα */ }
}
loadStatus();

// ── ανέβασμα ─────────────────────────────────────────────────────────
let chosen = null;
const drop = $('drop'), file = $('file');
drop.onclick = () => file.click();
['dragenter','dragover'].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.add('over'); }));
['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => { if (e.dataTransfer.files[0]) pick(e.dataTransfer.files[0]); });
file.onchange = () => { if (file.files[0]) pick(file.files[0]); };

function pick(f) {
  const okExt = /\.(mp4|mov|m4v|webm|avi|mkv)$/i.test(f.name);
  const box = $('picked'); box.classList.remove('hide');
  if (!okExt) {
    box.innerHTML = `<div class="err">Μη υποστηριζόμενος τύπος: ${esc(f.name)}</div>`;
    chosen = null; $('go').disabled = true; return;
  }
  if (f.size > 400 * 1024 * 1024) {
    box.innerHTML = `<div class="err">Το αρχείο είναι ${(f.size/1e6).toFixed(0)} MB — όριο 400 MB</div>`;
    chosen = null; $('go').disabled = true; return;
  }
  chosen = f;
  box.innerHTML = `<div class="ok">🎬 <b>${esc(f.name)}</b> · ${(f.size/1e6).toFixed(1)} MB</div>`;
  $('go').disabled = false;
}

$('go').onclick = async () => {
  if (!chosen) return;
  $('go').disabled = true; $('uperr').innerHTML = '';
  const fd = new FormData();
  fd.append('video', chosen);
  fd.append('context', $('context').value);
  fd.append('creators', $('creators').value);
  fd.append('captions', $('captions').value);
  if ($('noresearch').checked) fd.append('no_research', '1');
  try {
    const r = await fetch('/api/analyze', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Αποτυχία ανεβάσματος');
    $('uploader').classList.add('hide');
    $('result').innerHTML = '';
    watch(d.job_id);
  } catch (e) {
    $('uperr').innerHTML = `<div class="err">${esc(e.message)}</div>`;
    $('go').disabled = false;
  }
};

// ── παρακολούθηση προόδου ────────────────────────────────────────────
const STEP_NAMES = ['Τεχνική ανάλυση βίντεο','Οπτική ανάλυση καρέ με AI',
  'Επιλογή viral γωνίας','Σχεδιασμός ελληνικής έρευνας','Έρευνα HikerAPI',
  'Αποθήκευση στη μνήμη','Αναζήτηση ιστορικών αναλόγων','Εξόρυξη μοτίβων',
  'Παραγωγή λεζαντών με AI','Κατασκευή χαρτοφυλακίων hashtags',
  'Σκοράρισμα συνδυασμών','Τεκμήρια και μάθηση'];

function renderSteps(current) {
  const box = $('psteps'); box.innerHTML = '';
  STEP_NAMES.forEach((name, i) => {
    const n = i + 1;
    const cls = n < current ? 'step done' : (n === current ? 'step now' : 'step');
    const mark = n < current ? '✓' : '';
    box.appendChild(el('div', cls, `<span class="dot">${mark}</span><span>${esc(name)}</span>`));
  });
}

function watch(jobId) {
  $('progress').classList.remove('hide');
  renderSteps(0);
  const t0 = Date.now();
  const tick = async () => {
    let j;
    try { j = await api('/api/job/' + jobId); }
    catch (e) { $('ptitle').innerHTML = `<span class="err">${esc(e.message)}</span>`; return; }
    const pct = Math.round(100 * j.step / (j.total || 12));
    $('pbar').style.width = pct + '%';
    $('ptitle').textContent = j.step_label + (j.step_detail ? ` — ${j.step_detail}` : '');
    const mins = Math.floor(j.elapsed / 60), secs = Math.round(j.elapsed % 60);
    $('pmeta').textContent = `Βήμα ${j.step}/${j.total} · ${mins}:${String(secs).padStart(2,'0')} · `
      + 'η πλήρης ανάλυση διαρκεί συνήθως 3–5 λεπτά';
    renderSteps(j.step);
    if (j.status === 'done') {
      const full = await api('/api/job/' + jobId + '?result=1');
      $('progress').classList.add('hide');
      showResult(full.result);
      loadStatus();
      return;
    }
    if (j.status === 'error') {
      $('progress').classList.add('hide');
      $('result').innerHTML = `<div class="panel"><div class="err">
        <b>Η ανάλυση απέτυχε</b><br>${esc(j.error)}</div>
        <button class="ghost" onclick="location.reload()">Νέα προσπάθεια</button></div>`;
      return;
    }
    setTimeout(tick, 1500);
  };
  tick();
}

// ── αποτέλεσμα ───────────────────────────────────────────────────────
function showResult(r) {
  const out = $('result'); out.innerHTML = '';
  if (!r || !r.winner) {
    out.innerHTML = `<div class="panel"><div class="err">
      Το αποτέλεσμα δεν είναι διαθέσιμο. Αν ο server επανεκκινήθηκε, βρες το
      στην καρτέλα <b>Ιστορικό</b> — οι αναλύσεις αποθηκεύονται στη βάση.</div></div>`;
    return;
  }
  const w = r.winner;
  if (!w) { out.innerHTML = '<div class="panel"><div class="err">Δεν παρήχθη αποτέλεσμα.</div></div>'; return; }
  const tags = w.hashtag_set.tags.map(t => '#' + t).join(' ');

  // ΑΠΟΦΑΣΗ
  const p = el('div', 'panel');
  p.appendChild(el('h2', '', 'Η απόφαση'));
  const sc = el('div', 'score');
  const col = w.score.total >= 70 ? 'var(--good)' : w.score.total >= 50 ? 'var(--warn)' : 'var(--bad)';
  sc.innerHTML = `<b style="color:${col}">${Math.round(w.score.total)}</b><span class="of">/100</span>
    <span class="band">εύρος ${Math.round(w.score.interval[0])}–${Math.round(w.score.interval[1])}<br>
    βεβαιότητα: <b>${esc(w.score.confidence)}</b></span>`;
  p.appendChild(sc);
  p.appendChild(el('h3', '', 'Λεζάντα'));
  p.appendChild(copyBox('', w.caption.text, false));
  p.appendChild(el('div', 'small muted',
    `${esc(w.caption.strategy)} · ${w.caption.length_chars} χαρακτ. · ${w.caption.emoji_count} emoji`));
  p.appendChild(el('h3', '', 'Hashtags'));
  p.appendChild(copyBox('', tags, true));
  const dist = Object.entries(w.hashtag_set.tier_distribution || {})
    .map(([k, v]) => `${k}: ${v}`).join(' · ');
  p.appendChild(el('div', 'small muted',
    `Χαρτοφυλάκιο «${esc(w.hashtag_set.strategy)}» — ${esc(dist)} · ελληνικά ${Math.round(w.hashtag_set.greek_share*100)}%`));
  p.appendChild(el('h3', '', 'Έτοιμο για επικόλληση'));
  p.appendChild(copyBox('', r.ready_to_paste, false));
  out.appendChild(p);

  // ΓΙΑΤΙ
  const why = el('div', 'panel');
  why.appendChild(el('h2', '', 'Γιατί αυτός ο συνδυασμός'));
  why.appendChild(el('p', 'sub', esc(r.why_won)));
  const ang = r.video.chosen_angle;
  if (ang) {
    why.appendChild(el('h3', '', 'Viral γωνία'));
    why.appendChild(el('div', '', `<b>«${esc(ang.name)}»</b> <span class="pill">${esc(ang.strategy)}</span>
      <span class="pill">ισχύς ${ang.strength}</span>`));
    why.appendChild(el('div', 'small muted', `
      <div style="margin-top:8px"><b>Γιατί σταματά το scroll:</b> ${esc(ang.why_greek_stops)}</div>
      <div><b>Γιατί σχολιάζει:</b> ${esc(ang.why_comment)}</div>
      <div><b>Γιατί το στέλνει:</b> ${esc(ang.why_share)}</div>
      <div><b>Τι προσθέτει η λεζάντα:</b> ${esc(ang.caption_should_add)}</div>`));
  }
  why.appendChild(el('h3', '', 'Ανάλυση σκορ'));
  const tb = el('table', '', '<tr><th>Πυλώνας</th><th class="num">Σκορ</th><th style="width:110px"></th><th class="num">Συνεισφορά</th></tr>');
  w.score.pillars.slice().sort((a,b) => b.weighted - a.weighted).forEach(pl => {
    tb.insertRow().innerHTML = `<td>${esc(pl.label_el)}</td>
      <td class="num">${Math.round(pl.raw)}</td>
      <td><div class="pbar"><i style="width:${pl.raw}%"></i></div></td>
      <td class="num mono">${pl.weighted.toFixed(2)}</td>`;
  });
  why.appendChild(tb);
  (w.score.notes || []).forEach(n => why.appendChild(el('div', 'warn', esc(n))));
  out.appendChild(why);

  // ΤΕΚΜΗΡΙΑ
  if ((r.evidence || []).length) {
    const ev = el('div', 'panel');
    ev.appendChild(el('h2', '', 'Τεκμήρια από πραγματικά posts'));
    if (r.research)
      ev.appendChild(el('p', 'sub',
        `${num(r.research.posts.length)} posts εξετάστηκαν · ${num(r.research.greek_posts)} ελληνικά · `
        + `${num(r.research.outliers.length)} με δυσανάλογη απόδοση · ${num(r.api_calls)} κλήσεις API`));
    r.evidence.forEach(e => {
      const d = el('div', 'ev');
      d.innerHTML = `<div><b>@${esc(e.username)}</b>
        ${e.url ? `<a href="${esc(e.url)}" target="_blank" rel="noopener">↗</a>` : ''}
        <span class="muted small"> ${num(e.views)} προβολές / ${num(e.followers)} followers</span>
        ${e.vf_ratio ? `<span class="vf"> V/F ${e.vf_ratio}×</span>` : ''}
        ${e.age_days != null ? `<span class="muted small"> · ${Math.round(e.age_days)} ημ.</span>` : ''}</div>
        ${e.caption_excerpt ? `<div class="q">«${esc(e.caption_excerpt)}»</div>` : ''}
        <div class="small muted">${esc(e.why_relevant)}</div>`;
      ev.appendChild(d);
    });
    out.appendChild(ev);
  }

  // ΕΝΑΛΛΑΚΤΙΚΑ
  if ((r.backups || []).length) {
    const bk = el('div', 'panel');
    bk.appendChild(el('h2', '', 'Εναλλακτικές λεζάντες'));
    r.backups.forEach((b, i) => {
      const d = el('details');
      d.innerHTML = `<summary>${i+2}. ${Math.round(b.score.total)}/100 · ${esc(b.caption.strategy)}</summary>`;
      d.appendChild(copyBox('', b.caption.text, false));
      bk.appendChild(d);
    });
    (r.backup_hashtag_sets || []).forEach(b => {
      const d = el('details');
      d.innerHTML = `<summary>Hashtags: «${esc(b.hashtag_set.strategy)}» · ${Math.round(b.score.total)}/100</summary>`;
      d.appendChild(copyBox('', b.hashtag_set.tags.map(t => '#'+t).join(' '), true));
      bk.appendChild(d);
    });
    out.appendChild(bk);
  }

  // ΠΕΡΙΟΡΙΣΜΟΙ
  const gaps = (r.warnings || []).concat(r.data_gaps || []);
  if (gaps.length) {
    const g = el('div', 'panel');
    g.appendChild(el('h2', '', 'Περιορισμοί και κενά δεδομένων'));
    gaps.forEach(x => g.appendChild(el('div', 'small muted', '· ' + esc(x))));
    out.appendChild(g);
  }

  const foot = el('div', 'panel');
  foot.innerHTML = `<div class="small muted">Εκτέλεση <b>${esc(r.run_id)}</b> ·
    ${r.duration_s}s · Μετά τη δημοσίευση κατέγραψε το αποτέλεσμα στο <b>Ιστορικό</b>.</div>`;
  const again = el('button', 'ghost', 'Νέα ανάλυση');
  again.onclick = () => location.reload();
  foot.appendChild(el('div', '', '<br>')); foot.appendChild(again);
  out.appendChild(foot);
  out.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── ιστορικό + ανατροφοδότηση ────────────────────────────────────────
async function loadRuns() {
  const box = $('runs'); box.innerHTML = '<span class="spin"></span>';
  try {
    const d = await api('/api/runs?limit=25');
    if (!d.runs.length) { box.innerHTML = '<p class="muted">Καμία ανάλυση ακόμη.</p>'; return; }
    box.innerHTML = '';
    d.runs.forEach(r => {
      const det = el('details');
      const when = new Date(r.created_at * 1000).toLocaleString('el-GR',
        { day:'2-digit', month:'2-digit', hour:'2-digit', minute:'2-digit' });
      det.innerHTML = `<summary>${when} · <b>${Math.round(r.predicted_score||0)}/100</b>
        <span class="muted">(${esc(r.confidence||'')})</span> — ${esc((r.caption||'').slice(0,60))}…</summary>`;
      const body = el('div', '', `<div class="small muted" style="margin:10px 0">
        ${esc(r.niche||'')} · ${esc(r.angle_name||'')}</div>`);
      body.appendChild(copyBox('', r.caption || '', false));

      // Οι αναλύσεις ζουν στη βάση, όχι μόνο στη μνήμη του server: μια
      // εκτέλεση από την περασμένη εβδομάδα πρέπει να ανοίγει πλήρης.
      const view = el('button', 'ghost', 'Δες πλήρες αποτέλεσμα');
      view.onclick = async () => {
        view.disabled = true;
        try {
          const d2 = await api('/api/run/' + r.run_id);
          document.querySelectorAll('nav button')[0].click();
          $('uploader').classList.add('hide');
          showResult(d2.result);
        } catch (e) {
          view.insertAdjacentHTML('afterend', `<div class="err">${esc(e.message)}</div>`);
        }
        view.disabled = false;
      };
      body.appendChild(view);

      const fb = el('div', '');
      fb.innerHTML = `<h3>Κατέγραψε το πραγματικό αποτέλεσμα</h3>
        <div class="grid">
          <div><label>URL δημοσιευμένου Reel</label><input placeholder="https://instagram.com/reel/…" data-f="url"></div>
          <div><label>Προβολές</label><input type="number" data-f="views" placeholder="π.χ. 120000"></div>
          <div><label>Followers τη στιγμή</label><input type="number" data-f="followers" placeholder="π.χ. 8300"></div>
          <div><label>Likes</label><input type="number" data-f="likes"></div>
          <div><label>Σχόλια</label><input type="number" data-f="comments"></div>
        </div>
        <div class="small muted" style="margin:8px 0">Με URL τα νούμερα έρχονται αυτόματα από το HikerAPI.</div>`;
      const btn = el('button', '', 'Καταγραφή');
      const msg = el('div', '');
      btn.onclick = async () => {
        btn.disabled = true; msg.innerHTML = '<span class="spin"></span>';
        const body2 = { run_id: r.run_id };
        fb.querySelectorAll('[data-f]').forEach(i => {
          if (i.value) body2[i.dataset.f] = i.dataset.f === 'url' ? i.value : Number(i.value); });
        try {
          const res = await api('/api/feedback', { method: 'POST',
            headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body2) });
          msg.innerHTML = `<div class="ok">Καταγράφηκε. V/F <b>${res.vf_ratio ? res.vf_ratio.toFixed(1) : '—'}×</b>
            · πρόβλεψη ${res.predicted != null ? Math.round(res.predicted) : '—'}
            · πραγματικό ${res.actual != null ? Math.round(res.actual) : '—'}
            · ${res.patterns_updated} μοτίβα ενημερώθηκαν.
            ${res.summary && res.summary.correlation != null
              ? `<br>Συσχέτιση πρόβλεψης–πραγματικότητας: <b>${res.summary.correlation}</b> σε ${res.summary.n_comparable} μετρήσεις.` : ''}</div>`;
        } catch (e) { msg.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
        btn.disabled = false;
      };
      fb.appendChild(btn); fb.appendChild(msg);
      body.appendChild(fb); det.appendChild(body); box.appendChild(det);
    });
  } catch (e) { box.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}

// ── έρευνα ───────────────────────────────────────────────────────────
$('rgo').onclick = async () => {
  const t = $('rtarget').value.trim();
  if (!t) return;
  const out = $('rout'); out.innerHTML = '<span class="spin"></span> Έρευνα σε εξέλιξη…';
  try {
    const d = await api('/api/research', { method: 'POST',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ target: t }) });
    const poll = async () => {
      const j = await api('/api/job/' + d.job_id + '?result=1');
      if (j.status === 'running' || j.status === 'queued') { setTimeout(poll, 1200); return; }
      if (j.status === 'error') { out.innerHTML = `<div class="err">${esc(j.error)}</div>`; return; }
      renderResearch(out, j.result);
      loadStatus();
    };
    poll();
  } catch (e) { out.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
};
$('rtarget').addEventListener('keydown', e => { if (e.key === 'Enter') $('rgo').click(); });

function renderResearch(out, d) {
  out.innerHTML = '';
  if (d.kind === 'creator') {
    const c = d.creator, s = d.summary;
    out.appendChild(el('div', '', `<h3>@${esc(c.username)}</h3>
      <div class="small muted">${esc(c.full_name || '')}</div>
      <div style="margin:8px 0">
        <span class="pill">${num(c.followers)} followers</span>
        <span class="pill">ελληνικότητα ${Math.round(c.greek_confidence*100)}%</span>
        <span class="pill">διάμεσο ${num(s.median_views)} προβολές</span>
        ${s.max_vf ? `<span class="pill">καλύτερο V/F ${s.max_vf}×</span>` : ''}</div>`));
  } else {
    const st = d.stat || {};
    out.appendChild(el('div', '', `<h3>#${esc(d.target)}</h3>
      <div style="margin:8px 0">
        <span class="pill">${num(st.media_count)} posts</span>
        <span class="pill">επίπεδο ${esc(st.tier || '—')}</span>
        ${st.difficulty ? `<span class="pill">δυσκολία ${st.difficulty}/100</span>` : ''}
        ${d.trend ? `<span class="pill">τάση: ${esc(d.trend.label)}</span>` : ''}
        ${st.small_account_share != null ? `<span class="pill">μικροί λογαριασμοί ${Math.round(st.small_account_share*100)}%</span>` : ''}
      </div>`));
  }
  if (!d.posts.length) { out.appendChild(el('p', 'muted', 'Κανένα post.')); return; }
  const t = el('table', '', `<tr><th>Λογαριασμός</th><th class="num">Προβολές</th>
    <th class="num">Followers</th><th class="num">V/F</th><th class="num">Σκορ</th><th>Λεζάντα</th></tr>`);
  d.posts.forEach(p => {
    t.insertRow().innerHTML = `<td>${p.url ? `<a href="${esc(p.url)}" target="_blank" rel="noopener">@${esc(p.username)}</a>` : '@'+esc(p.username)}</td>
      <td class="num">${num(p.views)}</td><td class="num">${num(p.followers)}</td>
      <td class="num ${p.vf_ratio && p.vf_ratio > 5 ? 'vf' : ''}">${p.vf_ratio ? p.vf_ratio.toFixed(1)+'×' : '—'}</td>
      <td class="num">${p.outlier_score != null ? Math.round(p.outlier_score) : '—'}</td>
      <td class="small muted">${esc((p.caption || '').slice(0, 70))}</td>`;
  });
  out.appendChild(t);
}

// ── μνήμη ────────────────────────────────────────────────────────────
async function loadMemory(q) {
  const out = $('mout'); out.innerHTML = '<span class="spin"></span>';
  try {
    const d = await api('/api/memory' + (q ? '?q=' + encodeURIComponent(q) : ''));
    out.innerHTML = '';
    const s = d.stats, ps = d.patterns_stats;
    out.appendChild(el('div', '', `
      <span class="pill">${num(s.posts)} posts</span>
      <span class="pill">${num(s.greek_posts)} ελληνικά</span>
      <span class="pill">${num(s.snapshots)} στιγμιότυπα</span>
      <span class="pill">${num(s.outliers)} outliers</span>
      <span class="pill">${num(ps.patterns)} μοτίβα (${num(ps.usable)} αξιοποιήσιμα)</span>`));
    if (d.results && d.results.length) {
      out.appendChild(el('h3', '', 'Αποτελέσματα αναζήτησης'));
      d.results.forEach(r => {
        const d2 = el('div', 'ev');
        d2.innerHTML = `<div><b>@${esc(r.username)}</b>
          <span class="muted small">${num(r.views)} προβολές</span>
          ${r.vf_ratio ? `<span class="vf"> V/F ${r.vf_ratio.toFixed(1)}×</span>` : ''}
          <span class="muted small"> · ομοιότητα ${(r.similarity||0).toFixed(2)}</span></div>
          <div class="q">«${esc((r.caption_body||'').slice(0,150))}»</div>
          <div class="small" style="color:var(--accent)">${(r.hashtags||[]).slice(0,10).map(t=>'#'+esc(t)).join(' ')}</div>`;
        out.appendChild(d2);
      });
    }
    if (d.patterns && d.patterns.length) {
      out.appendChild(el('h3', '', 'Μοτίβα που έχει μάθει'));
      const t = el('table', '', `<tr><th>Μοτίβο</th><th class="num">Δείγματα</th>
        <th class="num">Μέσο</th><th class="num">Κάτω φράγμα</th></tr>`);
      d.patterns.forEach(p => {
        t.insertRow().innerHTML = `<td>${esc(p.description || p.key)}</td>
          <td class="num">${p.n}</td><td class="num">${p.mean.toFixed(2)}</td>
          <td class="num">${p.lower.toFixed(2)}</td>`;
      });
      out.appendChild(t);
    } else {
      out.appendChild(el('p', 'small muted',
        'Κανένα μοτίβο με αρκετά δείγματα ακόμη — χρειάζονται ≥4 επιβεβαιώσεις. '
        + 'Χτίζεται με κάθε ανάλυση και κάθε καταγραφή αποτελέσματος.'));
    }
  } catch (e) { out.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}
$('mgo').onclick = () => loadMemory($('mq').value.trim());
$('mq').addEventListener('keydown', e => { if (e.key === 'Enter') $('mgo').click(); });
