/* ROMA project page: teaser controls, reveal animations and hand-drawn charts.
   All figures are drawn here from the numbers reported in the paper. */
(() => {
  'use strict';

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const SVGNS = 'http://www.w3.org/2000/svg';
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

  function make(ns, tag, attrs = {}, kids = []) {
    const el = ns ? document.createElementNS(SVGNS, tag) : document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') el.setAttribute('class', v);
      else if (k === 'html') el.innerHTML = v;
      else if (k === 'text') el.textContent = v;
      else if (k === 'style') el.setAttribute('style', v);
      else el.setAttribute(k, v);
    }
    for (const kid of [].concat(kids)) if (kid != null) el.append(kid);
    return el;
  }
  const h = (tag, attrs, ...kids) => make(false, tag, attrs, kids);
  const s = (tag, attrs, ...kids) => make(true, tag, attrs, kids);
  const fmt = (n) => n.toLocaleString('en-US');

  /* ------------------------------------------------------------------ *
   * Teaser: bare <video>, custom pause/resume, back/forward 1.5 s, replay
   * ------------------------------------------------------------------ */
  (function teaser() {
    const v = $('#teaser');
    if (!v) return;
    const play = $('#ctl-play');
    const label = $('span', play);
    const STEP = 1.5;

    v.muted = true;
    v.removeAttribute('controls');
    v.addEventListener('contextmenu', (e) => e.preventDefault());

    const sync = () => {
      const paused = v.paused || v.ended;
      play.classList.toggle('is-paused', paused);
      label.textContent = paused ? 'Resume' : 'Pause';
      play.setAttribute('aria-label', paused ? 'Resume' : 'Pause');
    };
    const start = () => v.play().catch(() => sync());

    play.addEventListener('click', () => {
      if (v.ended) v.currentTime = 0;
      if (v.paused || v.ended) start(); else v.pause();
    });
    $('#ctl-back').addEventListener('click', () => {
      v.currentTime = Math.max(0, v.currentTime - STEP);
    });
    $('#ctl-fwd').addEventListener('click', () => {
      if (!isFinite(v.duration)) return;
      v.currentTime = Math.min(v.duration, v.currentTime + STEP);
    });
    $('#ctl-replay').addEventListener('click', () => {
      v.currentTime = 0;
      start();
    });
    ['play', 'pause', 'ended', 'playing', 'loadeddata'].forEach((ev) => v.addEventListener(ev, sync));

    if (reduceMotion) v.pause();
    sync();
  })();

  /* ------------------------------------------------------------------ *
   * Nav state, reveal-on-scroll, count-up
   * ------------------------------------------------------------------ */
  const nav = $('#nav');
  const onScroll = () => nav.classList.toggle('scrolled', scrollY > 8);
  addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  const links = new Map($$('.nav-links a').map((a) => [a.getAttribute('href').slice(1), a]));
  const secObs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      links.forEach((a) => a.classList.remove('active'));
      const a = links.get(e.target.id);
      if (a) a.classList.add('active');
    });
  }, { rootMargin: '-45% 0px -50% 0px' });
  $$('main section[id]').forEach((sec) => secObs.observe(sec));

  function countUp(el) {
    const target = parseFloat(el.dataset.count);
    const dec = parseInt(el.dataset.dec || '0', 10);
    const prefix = el.dataset.prefix || '';
    const suffix = el.textContent.trim().endsWith('%') ? '%' : '';
    if (reduceMotion) return;
    const t0 = performance.now();
    const dur = 1100;
    const tick = (t) => {
      const p = Math.min(1, (t - t0) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = prefix + (target * eased).toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec }) + suffix;
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  const revealObs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      e.target.classList.add('in');
      $$('[data-count]', e.target).forEach(countUp);
      revealObs.unobserve(e.target);
    });
  }, { threshold: 0.12 });
  $$('.reveal').forEach((el) => revealObs.observe(el));

  /* ------------------------------------------------------------------ *
   * Chart plumbing: charts animate in when first seen, and again on redraw
   * ------------------------------------------------------------------ */
  const seen = new WeakSet();
  const goObs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      seen.add(e.target);
      go(e.target);
      goObs.unobserve(e.target);
    });
  }, { threshold: 0.25 });

  function go(el) {
    el.classList.remove('go');
    void el.offsetWidth;
    requestAnimationFrame(() => el.classList.add('go'));
  }
  function mount(el, render) {
    el.replaceChildren();
    render(el);
    if (seen.has(el)) go(el);
    else if (!el._watched) { el._watched = true; goObs.observe(el); }
  }
  function tabs(el, items, onSelect, initial = 0) {
    const btns = items.map((it, i) => h('button', { type: 'button', role: 'tab', class: 'tab', text: it.tab, 'aria-selected': String(i === initial) }));
    const select = (i) => {
      btns.forEach((b, j) => b.setAttribute('aria-selected', String(i === j)));
      onSelect(items[i], i);
    };
    btns.forEach((b, i) => b.addEventListener('click', () => select(i)));
    el.replaceChildren(...btns);
    select(initial);
  }

  /* ---------------- donut ---------------- */
  function donut(el, segs, centerBig, centerSmall) {
    const total = segs.reduce((a, d) => a + d.value, 0);
    const R = 48;
    let off = 0;
    const circles = segs.map((d, i) => {
      const seg = (d.value / total) * 100;
      const c = s('circle', {
        cx: 60, cy: 60, r: R, pathLength: 100, stroke: d.color,
        'stroke-dashoffset': -off, style: `--seg:${Math.max(seg - 0.7, 0.1)};--d:${i * 0.12}s`,
      });
      c.append(s('title', { text: `${d.label}: ${fmt(d.value)}` }));
      off += seg;
      return c;
    });
    const ring = h('div', { class: 'donut' },
      s('svg', { viewBox: '0 0 120 120', 'aria-hidden': 'true' }, ...circles),
      h('div', { class: 'center-label' }, h('b', { text: centerBig }), h('small', { text: centerSmall })));
    const legend = h('ul', { class: 'donut-legend' }, ...segs.map((d) =>
      h('li', { style: `--c:${d.color}` }, h('i'),
        h('span', {}, h('b', { text: fmt(d.value) }), ' ', d.label, ' ',
          h('small', { text: `(${((d.value / total) * 100).toFixed(1)}%)` })))));
    el.append(ring, legend);
  }

  /* ---------------- horizontal bars ---------------- */
  function hbars(el, rows, { max, color, fmtv = (v) => fmt(v) } = {}) {
    const top = max ?? Math.max(...rows.map((r) => r.value));
    rows.forEach((r, i) => {
      el.append(h('div', { class: 'hbar' },
        h('span', { text: r.label }),
        h('div', { class: 'track' }, h('i', { style: `--v:${(r.value / top) * 100};--d:${i * 0.045}s;--c:${r.color || color}` })),
        h('b', { text: fmtv(r.value) })));
    });
  }

  /* ------------------------------------------------------------------ *
   * ROMI-2K statistics (paper Figs. 16, 17, 19)
   * ------------------------------------------------------------------ */
  const SUBSETS = [
    {
      ex: 'hand', tab: 'Handheld', color: 'var(--blue)', big: '1,657', small: 'objects',
      materials: [['plastic', 908], ['paper', 488], ['metal', 255], ['fabric', 105], ['wood', 38], ['rubber', 32], ['glass', 29],
        ['silicone', 27], ['ceramic', 17], ['sponge', 16], ['leather', 16], ['foam', 14], ['cotton', 3], ['plants', 3]],
      has: 688, none: 969,
    },
    {
      ex: 'table', tab: 'Tabletop · train', color: 'var(--purple)', big: '1,269', small: 'instances',
      materials: [['plastic', 664], ['paper', 313], ['metal', 154], ['glass', 50], ['fabric', 40], ['ceramic', 36], ['wood', 25],
        ['sponge', 19], ['leather', 11], ['rubber', 9], ['silicone', 8], ['nylon', 3], ['wax', 2]],
      has: 693, none: 576,
    },
    {
      ex: 'bench', tab: 'ROMA Bench', color: 'var(--orange)', big: '325', small: 'instances',
      materials: [['plastic', 141], ['paper', 69], ['ceramic', 24], ['metal', 23], ['sponge', 22], ['silicone', 16], ['fabric', 11],
        ['leather', 10], ['glass', 6], ['rubber', 4]],
      has: 173, none: 152,
    },
  ];
  /* ------------------------------------------------------------------ *
   * Looping example clips: no player chrome, play only while on screen
   * ------------------------------------------------------------------ */
  const clipObs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      const v = e.target;
      if (e.isIntersecting) {
        if (!v.getAttribute('src')) v.src = v.dataset.src;
        if (!reduceMotion) v.play().catch(() => {});
      } else if (v.getAttribute('src')) v.pause();
    });
  }, { rootMargin: '200px 0px' });

  function clip(v, rate = 1) {
    v.muted = true; v.loop = true; v.playsInline = true;
    v.removeAttribute('controls');
    v.disablePictureInPicture = true;
    const setRate = () => { v.playbackRate = rate; };
    v.addEventListener('loadedmetadata', setRate);
    v.addEventListener('play', setRate);
    v.addEventListener('contextmenu', (e) => e.preventDefault());
    clipObs.observe(v);
    return v;
  }
  $$('video[data-src]').forEach((v) => clip(v));

  /* ---------------- dataset examples (assest/examples) ---------------- */
  const EXDIR = 'assest/examples/';
  const MODS = { vision: 'Vision', audio: 'Audio', touch: 'Touch', force: 'Force' };
  function tableLike(dir) {
    return [
      { type: 'image', file: `${dir}/Initial_180.png`, mod: 'vision', label: 'Initial scene' },
      { type: 'video', file: `${dir}/collide_ThirdCamera.mp4`, action: 'collide', mod: 'vision', note: 'third view' },
      { type: 'video', file: `${dir}/collide_WristCamera.mp4`, action: 'collide', mod: 'vision', note: 'wrist view' },
      { type: 'video', file: `${dir}/clench_GelSightL.mp4`, action: 'squeeze', mod: 'touch', note: 'left sensor' },
      { type: 'video', file: `${dir}/clench_GelSightR.mp4`, action: 'squeeze', mod: 'touch', note: 'right sensor' },
      { type: 'audio', file: `${dir}/audio_enhanced_cut.wav`, action: 'shake', mod: 'audio' },
      { type: 'video', file: `${dir}/rotate_force.mp4`, action: 'rotate', mod: 'force', rate: 2, badge: '2× speed', fit: 'contain' },
    ];
  }
  const EXAMPLES = {
    hand: [
      { type: 'video', file: 'hand/collide_ThirdCamera.mp4', action: 'collide', mod: 'vision', note: 'third view' },
      { type: 'video', file: 'hand/collide_WristCamera.mp4', action: 'collide', mod: 'vision', note: 'wrist view' },
      { type: 'video', file: 'hand/clench_GelSight1.mp4', action: 'squeeze', mod: 'touch', note: 'sensor 1' },
      { type: 'video', file: 'hand/clench_GelSight2.mp4', action: 'squeeze', mod: 'touch', note: 'sensor 2' },
      { type: 'audio', file: 'hand/shake_mic.wav', action: 'shake', mod: 'audio' },
    ],
    table: tableLike('table'),
    bench: tableLike('bench'),
  };

  let playingAudio = null;
  function audioTile(it) {
    const bars = Array.from({ length: 28 }, (_, i) => {
      const hgt = 18 + 62 * Math.abs(Math.sin(i * 1.7) * Math.cos(i * 0.55 + 1));
      return s('rect', { x: 8 + i * 6.2, y: 50 - hgt / 2, width: 3.4, height: hgt, rx: 1.7 });
    });
    const audio = h('audio', { src: EXDIR + it.file, preload: 'none' });
    const icons = '<svg viewBox="0 0 24 24" class="i-play"><path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.5-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5z"/></svg>'
      + '<svg viewBox="0 0 24 24" class="i-pause"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>';
    const btn = h('button', { type: 'button', class: 'audio-tile', 'aria-label': 'Play audio example' },
      s('svg', { viewBox: '0 0 190 100', preserveAspectRatio: 'none', 'aria-hidden': 'true', class: 'wave' }, ...bars),
      h('span', { class: 'ap', html: icons }),
      audio);
    const sync = () => btn.classList.toggle('is-playing', !audio.paused);
    ['play', 'pause', 'ended'].forEach((ev) => audio.addEventListener(ev, sync));
    btn.addEventListener('click', () => {
      if (!audio.paused) { audio.pause(); return; }
      if (playingAudio && playingAudio !== audio) playingAudio.pause();
      playingAudio = audio; audio.currentTime = 0; audio.play().catch(() => {});
    });
    return btn;
  }
  function renderExamples(key) {
    const grid = $('#data-examples');
    if (!grid) return;
    if (playingAudio) { playingAudio.pause(); playingAudio = null; }
    grid.querySelectorAll('video').forEach((v) => clipObs.unobserve(v));
    const figs = EXAMPLES[key].map((it) => {
      let media;
      if (it.type === 'video') {
        const v = h('video', { 'data-src': EXDIR + it.file, preload: 'none', 'aria-label': `${it.action} ${MODS[it.mod]} example` });
        if (it.fit) v.style.objectFit = it.fit;
        media = clip(v, it.rate || 1);
      } else if (it.type === 'image') {
        media = h('img', { src: EXDIR + it.file, alt: 'Initial scene', loading: 'lazy' });
      } else media = audioTile(it);
      const note = it.note ? `, ${it.note}` : '';
      const cap = h('figcaption', { html: it.action
        ? `<code>&lt;${it.action}&gt;</code> <span class="m ${it.mod}">(${MODS[it.mod]}${note})</span>${it.badge ? `<em class="rate">${it.badge}</em>` : ''}`
        : `${it.label} <span class="m ${it.mod}">(${MODS[it.mod]})</span>` });
      return h('figure', { class: 'ex' }, h('div', { class: 'ex-media' }, media), cap);
    });
    grid.className = key === 'hand' ? 'ex-grid rows' : 'ex-grid split';
    if (key === 'hand') {
      grid.replaceChildren(h('div', { class: 'ex-row' }, ...figs.slice(0, 2)), h('div', { class: 'ex-row' }, ...figs.slice(2)));
    } else {
      figs[0].classList.add('ex-lead');
      grid.replaceChildren(...figs);
    }
  }

  const dataTabs = $('#data-tabs');
  if (dataTabs) {
    tabs(dataTabs, SUBSETS, (d) => {
      renderExamples(d.ex);
      mount($('#chart-materials'), (el) => {
        el.classList.add('hbars');
        hbars(el, d.materials.map(([label, value]) => ({ label, value })), { color: d.color });
      });
      mount($('#chart-contents'), (el) => donut(el, [
        { label: 'with contents', value: d.has, color: d.color },
        { label: 'empty', value: d.none, color: '#d9d5c8' },
      ], d.big, d.small));
    });
  }

  /* ------------------------------------------------------------------ *
   * Beta-sampling curves (Sec. 4.2, alpha = 2.5)
   * ------------------------------------------------------------------ */
  const beta = $('#chart-beta');
  if (beta) {
    mount(beta, (el) => {
      const a = 2.5, W = 440, x0 = 30, x1 = 424, yb = 172, ys = 116 / a;
      const X = (t) => x0 + t * (x1 - x0);
      const pts = (f) => Array.from({ length: 81 }, (_, i) => { const t = i / 80; return [X(t), yb - f(t) * ys]; });
      const line = (p) => 'M' + p.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' L');
      const early = pts((t) => a * Math.pow(1 - t, a - 1));
      const late = pts((t) => a * Math.pow(t, a - 1));
      const area = (p) => `${line(p)} L${x1},${yb} L${x0},${yb} Z`;
      el.append(s('svg', { viewBox: `0 0 ${W} 214`, role: 'img', 'aria-label': 'Beta(1, 2.5) and Beta(2.5, 1) sampling densities over training progress' },
        s('line', { class: 'axis', x1: x0, y1: yb, x2: x1, y2: yb }),
        s('path', { class: 'area', d: area(early), fill: 'rgba(43,127,224,.16)' }),
        s('path', { class: 'area', d: area(late), fill: 'rgba(122,99,204,.18)' }),
        s('path', { d: line(early), fill: 'none', stroke: 'var(--blue)', 'stroke-width': 3, 'stroke-linecap': 'round' }),
        s('path', { d: line(late), fill: 'none', stroke: 'var(--purple)', 'stroke-width': 3, 'stroke-linecap': 'round' }),
        s('text', { class: 'lbl', x: x0 + 8, y: 26, fill: 'var(--blue)', style: 'fill:#1a5fb4', text: 'Handheld + pseudo-scenes' }),
        s('text', { x: x0 + 8, y: 42, text: 'Beta(1, α): early in training' }),
        s('text', { class: 'lbl', x: x1 - 4, y: 26, 'text-anchor': 'end', style: 'fill:#5a44ae', text: 'Real scenes (with force)' }),
        s('text', { x: x1 - 4, y: 42, 'text-anchor': 'end', text: 'Beta(α, 1): late in training' }),
        s('text', { x: x0, y: yb + 22, text: 'start of training' }),
        s('text', { x: x1, y: yb + 22, 'text-anchor': 'end', text: 'end of training' }),
        s('text', { x: (x0 + x1) / 2, y: yb + 22, 'text-anchor': 'middle', text: 'sorted sample position →' })));
    });
  }

  /* ------------------------------------------------------------------ *
   * ROMA Bench: task split donut + attribute chord (paper Fig. 18)
   * ------------------------------------------------------------------ */
  const tasks = $('#chart-tasks');
  if (tasks) {
    mount(tasks, (el) => donut(el, [
      { label: 'Single-chain', value: 741, color: 'var(--blue)' },
      { label: 'Multi-chain', value: 1036, color: 'var(--orange)' },
      { label: 'Intent-driven', value: 323, color: 'var(--green)' },
    ], '2,100', 'tasks'));
  }

  function drawChord(el) {
    // Counts from the ROMA Bench annotations. Multi-chain (level-2) tasks link two attributes and
    // draw the ribbons; single-chain (level-1) tasks involve one attribute and only lengthen its arc.
    const names = ['Hardness', 'Roughness', 'Texture', 'Material', 'Weight', 'Inside'];
    const colors = ['#2b7fe0', '#3fa46a', '#12a3b8', '#7a63cc', '#e0527a', '#f5921e'];
    const n = names.length;
    const solo = [99, 78, 42, 291, 153, 78];
    const m = [
      [0, 25, 61, 85, 49, 59],
      [25, 0, 75, 17, 22, 30],
      [61, 75, 0, 35, 102, 94],
      [85, 17, 35, 0, 119, 126],
      [49, 22, 102, 119, 0, 137],
      [59, 30, 94, 126, 137, 0],
    ];
    const paired = m.map((row) => row.reduce((a, b) => a + b, 0));
    const totals = solo.map((v, i) => v + paired[i]); // 378, 247, 409, 673, 582, 524

    const cx = 290, cy = 236, R = 168, r = 156, gap = 0.07;
    const k = (2 * Math.PI - n * gap) / totals.reduce((a, b) => a + b, 0);
    const P = (rad, a) => [cx + rad * Math.cos(a), cy + rad * Math.sin(a)];
    const pt = (p) => p.map((v) => v.toFixed(2)).join(',');

    let a = -Math.PI / 2 + gap / 2;
    const arcs = totals.map((t, i) => {
      const a0 = a; a += t * k; const a1 = a; a += gap;
      return { i, a0, a1, mid: (a0 + a1) / 2 };
    });
    // within each arc, put the clockwise-nearest partner last so ribbons do not tangle
    const sub = Array.from({ length: n }, () => Array(n));
    arcs.forEach(({ i, a0 }) => {
      let cur = a0;
      const order = [...Array(n).keys()].filter((j) => j !== i).sort((p, q) => ((q - i + n) % n) - ((p - i + n) % n));
      order.forEach((j) => { sub[i][j] = [cur, cur + m[i][j] * k]; cur += m[i][j] * k; });
    });

    const defs = s('defs');
    const ribbons = [];
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const [p0, p1] = sub[i][j], [q0, q1] = sub[j][i];
      const id = `rg-${i}-${j}`;
      const A = P(r, (p0 + p1) / 2), B = P(r, (q0 + q1) / 2);
      defs.append(s('linearGradient', { id, gradientUnits: 'userSpaceOnUse', x1: A[0], y1: A[1], x2: B[0], y2: B[1] },
        s('stop', { offset: 0, 'stop-color': colors[i] }), s('stop', { offset: 1, 'stop-color': colors[j] })));
      const d = `M${pt(P(r, p0))} A${r},${r} 0 0 1 ${pt(P(r, p1))} Q${cx},${cy} ${pt(P(r, q0))} A${r},${r} 0 0 1 ${pt(P(r, q1))} Q${cx},${cy} ${pt(P(r, p0))}Z`;
      ribbons.push(s('path', { class: 'ribbon', d, fill: `url(#${id})`, 'data-i': i, 'data-j': j }, s('title', { text: `${names[i]} + ${names[j]}: ${m[i][j]} multi-chain tasks` })));
    }

    const ringPath = (b0, b1) => {
      const big = b1 - b0 > Math.PI ? 1 : 0;
      return `M${pt(P(R, b0))} A${R},${R} 0 ${big} 1 ${pt(P(R, b1))} L${pt(P(r + 3, b1))} A${r + 3},${r + 3} 0 ${big} 0 ${pt(P(r + 3, b0))}Z`;
    };
    const arcEls = arcs.map(({ i, a0, a1 }) => {
      const split = a0 + paired[i] * k; // ribbons use [a0, split]; the rest is single-chain only
      const tip = `${names[i]}: ${totals[i]} tasks (${solo[i]} single-chain, ${paired[i]} multi-chain pairings)`;
      return s('g', { class: 'arc', 'data-i': i },
        s('path', { d: ringPath(a0, split), fill: colors[i] }, s('title', { text: tip })),
        s('path', { d: ringPath(split, a1), fill: colors[i], opacity: 0.45 }, s('title', { text: tip })));
    });

    const labels = arcs.map(({ i, mid }) => {
      const [lx, ly] = P(R + 16, mid);
      const c = Math.cos(mid), sn = Math.sin(mid);
      const anchor = c > 0.2 ? 'start' : c < -0.2 ? 'end' : 'middle';
      const y = sn < -0.85 ? ly - 16 : sn > 0.85 ? ly + 4 : ly - 6;
      return s('text', { 'text-anchor': anchor, style: `fill:${colors[i]}` },
        s('tspan', { x: lx.toFixed(1), y: y.toFixed(1), text: names[i], style: 'fill:currentColor' }),
        s('tspan', { x: lx.toFixed(1), dy: 16, text: `(${totals[i]})`, style: 'fill:#667389;font-weight:500;font-size:12px' }));
    });

    const svg = s('svg', { viewBox: '0 0 580 480', role: 'img', 'aria-label': 'Chord diagram of the six target attributes in ROMA Bench' },
      defs, ...ribbons, ...arcEls, ...labels);
    el.append(svg);

    const light = (idx) => {
      el.classList.toggle('hover', idx != null);
      ribbons.forEach((p) => p.classList.toggle('on', idx != null && (+p.dataset.i === idx || +p.dataset.j === idx)));
    };
    arcEls.forEach((p) => {
      const idx = +p.dataset.i;
      p.addEventListener('mouseenter', () => light(idx));
      p.addEventListener('mouseleave', () => light(null));
      p.addEventListener('click', () => light(el.classList.contains('hover') ? null : idx));
    });
  }
  const chordEl = $('#chart-chord');
  if (chordEl) drawChord(chordEl);

  /* ------------------------------------------------------------------ *
   * Results (paper Tabs. 1, 2, 7)
   * ------------------------------------------------------------------ */
  const MODELS = [
    { key: 'gpt', name: 'GPT-5.4', color: '#12a3b8' },
    { key: 'gem', name: 'Gemini 3.5 Flash', color: '#e0527a' },
    { key: 'q25', name: 'Qwen 2.5-Omni', color: '#e9b12c' },
    { key: 'q3', name: 'Qwen 3-Omni', color: '#3fa46a' },
    { key: 'roma', name: 'ROMA-7B', color: null },
  ];
  const GROUPS = ['Single-chain', 'Multi-chain', 'Intent-driven', 'Total'];
  const SETTINGS = [
    {
      tab: 'ROMA Bench',
      note: 'Offline benchmark, 2,100 tasks. Grasp localization failures count as task failures.',
      gpt: [45.1, 38.4, 56.0, 43.5], gem: [54.4, 49.9, 59.4, 53.0], q25: [21.1, 21.2, 26.3, 22.0],
      q3: [24.7, 13.3, 28.8, 19.7], roma: [71.1, 74.1, 73.1, 72.9],
    },
    {
      tab: 'Real robot',
      note: '132 free-form tasks in 8 recreated scenes, answers checked by hand. Every model uses the same physical interface.',
      gpt: [42.6, 29.4, 55.6, 40.2], gem: [50.0, 29.4, 48.1, 41.7], q25: [20.4, 29.4, 18.5, 23.5],
      q3: [33.3, 15.7, 18.5, 23.5], roma: [57.4, 64.7, 63.0, 61.4],
    },
    {
      tab: 'Oracle boxes*',
      note: 'Oracle boxes are offered and matching is relaxed, so localization errors mostly vanish for GPT and Gemini. <b>Not a realistic setting</b>. Frontier models catch up on several single-chain attributes, yet ROMA-7B stays ahead overall.',
      gpt: [68.2, 58.5, 67.5, 63.3], gem: [74.5, 65.1, 67.5, 68.8], q25: [31.3, 32.1, 31.3, 31.7],
      q3: [23.6, 14.8, 31.9, 20.5], roma: [75.0, 76.4, 77.1, 76.0],
    },
  ];
  const resTabs = $('#res-tabs');
  if (resTabs) {
    $('#model-legend').append(...MODELS.map((m) =>
      h('li', { class: m.key === 'roma' ? 'roma' : '', style: `--c:${m.color}` }, h('i', { class: m.key === 'roma' ? 'roma' : '' }), m.name)));
    tabs(resTabs, SETTINGS, (st) => {
      $('#res-note').innerHTML = st.note;
      mount($('#chart-models'), (el) => {
        GROUPS.forEach((g, gi) => {
          el.append(h('div', { class: 'col-group' },
            h('div', { class: 'col-plot' }, ...MODELS.map((m, mi) => {
              const v = st[m.key][gi];
              return h('div', { class: 'bar', title: `${m.name}: ${v}` },
                h('i', { class: m.key === 'roma' ? 'roma' : '', style: `--v:${v};--d:${mi * 0.07}s;--c:${m.color}` },
                  h('b', { class: m.key === 'roma' ? 'roma' : '', text: v.toFixed(1) })));
            })),
            h('div', { class: 'col-name', text: g })));
        });
      });
    });
  }

  // Tab. 1, per-attribute success on ROMA Bench.
  const ATTR = ['Har.', 'Rou.', 'Tex.', 'Ins.', 'Mat.', 'Wei.', 'All'];
  const TABLE1 = [
    ['GPT-5.4', [34.3, 41.0, 38.1, 23.1, 60.1, 38.6, 45.1], [32.0, 37.6, 38.1, 41.3, 42.1, 35.8, 38.4], 56.0, 43.5],
    ['Gemini 3.5 Flash', [45.5, 66.7, 50.0, 28.2, 58.4, 60.8, 54.4], [48.3, 58.8, 46.9, 46.4, 51.8, 52.9, 49.9], 59.4, 53.0],
    ['Qwen 2.5-Omni', [19.2, 12.8, 9.5, 5.1, 37.8, 5.9, 21.1], [20.1, 16.4, 18.8, 23.3, 28.8, 15.0, 21.2], 26.3, 22.0],
    ['Qwen 3-Omni', [4.0, 14.1, 19.0, 6.4, 53.3, 0.0, 24.7], [12.3, 10.3, 16.6, 16.8, 22.0, 0.7, 13.3], 28.8, 19.7],
    ['ROMA-7B', [48.5, 67.9, 81.0, 59.0, 70.8, 91.5, 71.1], [62.5, 75.8, 77.4, 75.1, 70.7, 80.9, 74.1], 73.1, 72.9],
  ];
  const heat = $('#heat');
  if (heat) {
    // flatten to 16 value columns: 7 single, 7 multi, intent, total
    const rows = TABLE1.map(([name, a, b, c, d]) => ({ name, vals: [...a, ...b, c, d] }));
    const best = rows[0].vals.map((_, ci) => Math.max(...rows.map((r) => r.vals[ci])));
    const head1 = h('tr', {},
      h('th', { rowspan: 2, text: 'Model' }),
      h('th', { class: 'grp', colspan: 7, text: 'Single-chain' }), h('th', { class: 'gap' }),
      h('th', { class: 'grp', colspan: 7, text: 'Multi-chain' }), h('th', { class: 'gap' }),
      h('th', { rowspan: 2, text: 'Intent-driven' }), h('th', { rowspan: 2, text: 'Total' }));
    const head2 = h('tr', {}, ...ATTR.map((t) => h('th', { text: t })), h('th', { class: 'gap' }), ...ATTR.map((t) => h('th', { text: t })), h('th', { class: 'gap' }));
    const body = rows.map((r) => {
      const cells = [];
      r.vals.forEach((v, ci) => {
        if (ci === 7) cells.push(h('td', { class: 'gap' }));
        if (ci === 14) cells.push(h('td', { class: 'gap' }));
        const cls = [v === best[ci] ? 'best' : '', (ci === 6 || ci === 13 || ci >= 14) ? 'all' : ''].join(' ').trim();
        cells.push(h('td', { class: cls, style: `--a:${v / 100}`, text: v.toFixed(1) }));
      });
      return h('tr', { class: r.name === 'ROMA-7B' ? 'roma' : '' }, h('th', { scope: 'row', text: r.name }), ...cells);
    });
    heat.append(h('thead', {}, head1, head2), h('tbody', {}, ...body));
  }

  /* ------------------------------------------------------------------ *
   * Analysis (paper Tab. 3, Fig. 23, Tabs. 8 and 9)
   * ------------------------------------------------------------------ */
  const scatter = $('#chart-scatter');
  if (scatter) {
    mount(scatter, (el) => {
      el.classList.add('scatter');
      const W = 520, H = 310, L = 50, Rr = 14, T = 20, B = 48;
      const xmax = 24, ymin = 55, ymax = 80;
      const X = (v) => L + (v / xmax) * (W - L - Rr);
      const Y = (v) => T + (1 - (v - ymin) / (ymax - ymin)) * (H - T - B);
      const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Accuracy versus interactions per scene' });
      const defs = s('defs', {}, s('linearGradient', { id: 'romaGrad', x1: 0, y1: 1, x2: 1, y2: 0 },
        s('stop', { offset: 0, 'stop-color': '#2b7fe0' }), s('stop', { offset: .6, 'stop-color': '#7a63cc' }), s('stop', { offset: 1, 'stop-color': '#f5921e' })));
      svg.append(defs);
      for (let y = 55; y <= 80; y += 5) {
        svg.append(s('line', { class: 'grid', x1: L, x2: W - Rr, y1: Y(y), y2: Y(y) }));
        svg.append(s('text', { x: L - 8, y: Y(y) + 4, 'text-anchor': 'end', text: y }));
      }
      for (let x = 0; x <= 20; x += 5) svg.append(s('text', { x: X(x), y: H - B + 20, 'text-anchor': 'middle', text: x }));
      svg.append(s('line', { class: 'axis', x1: L, x2: W - Rr, y1: H - B, y2: H - B }));
      svg.append(s('text', { x: (L + W - Rr) / 2, y: H - 6, 'text-anchor': 'middle', text: 'further interactions per scene' }));
      svg.append(s('text', { x: 12, y: (T + H - B) / 2, 'text-anchor': 'middle', transform: `rotate(-90 12 ${(T + H - B) / 2})`, text: 'total accuracy (%)' }));
      // exhaustive marker
      const ex = X(22.09);
      svg.append(s('line', { x1: ex, x2: ex, y1: T, y2: H - B, stroke: '#c9c4b6', 'stroke-width': 2, 'stroke-dasharray': '5 5' }));
      svg.append(s('text', { x: ex - 8, y: T + 14, 'text-anchor': 'end', style: 'font-weight:700;fill:#33415a', text: 'Exhaustive' }));
      svg.append(s('text', { x: ex - 8, y: T + 29, 'text-anchor': 'end', text: '22.1, all six on every object' }));
      // bracket for ~4x
      const by = Y(72);
      svg.append(s('path', { d: `M${X(5.4)},${by} H${ex}`, stroke: '#9aa3b5', 'stroke-width': 1.6, fill: 'none', 'marker-end': 'url(#arr)' }));
      svg.append(s('defs', {}, s('marker', { id: 'arr', viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto' },
        s('path', { d: 'M0,0 L10,5 L0,10 z', fill: '#9aa3b5' }))));
      svg.append(s('text', { x: (X(5.4) + ex) / 2, y: by - 8, 'text-anchor': 'middle', style: 'font-weight:700;fill:#33415a', text: '≈ 4× more interactions' }));
      const pts = [
        { n: 'GPT-5.4', x: 1.20, y: 63.3, fill: '#12a3b8' },
        { n: 'Gemini 3.5 Flash', x: 2.77, y: 68.8, fill: '#e0527a' },
        { n: 'ROMA-7B', x: 5.40, y: 76.0, fill: 'url(#romaGrad)', big: true },
      ];
      pts.forEach((p) => {
        svg.append(s('circle', { class: 'pt', cx: X(p.x), cy: Y(p.y), r: p.big ? 12 : 9, fill: p.fill, stroke: '#fff', 'stroke-width': 3 },
          s('title', { text: `${p.n}: ${p.y}% with ${p.x} interactions per scene` })));
        svg.append(s('text', { class: 'pt-label', x: X(p.x) + (p.big ? 18 : 15), y: Y(p.y) + 4, text: `${p.n}  ${p.y}%` }));
      });
      el.append(svg);
    });
  }

  const ACTIONS = ['lift', 'press', 'collide', 'shake', 'rotate', 'squeeze'];
  const ACT_COLORS = ['#7a63cc', '#3fa46a', '#12a3b8', '#f5921e', '#e9b12c', '#2b7fe0'];
  const ACT_DATA = [
    ['GPT-5.4', [39.4, 1.8, 0.3, 14.9, 5.2, 38.4]],
    ['Gemini 3.5 Flash', [40.1, 0.3, 3.5, 19.0, 4.2, 32.9]],
    ['ROMA-7B', [17.6, 10.3, 9.0, 14.4, 13.0, 35.8]],
  ];
  const stack = $('#chart-actions');
  if (stack) {
    $('#actions-legend').append(...ACTIONS.map((a, i) => h('li', { style: `--c:${ACT_COLORS[i]}`, text: a })));
    mount(stack, (el) => ACT_DATA.forEach(([name, vals]) => {
      el.append(h('div', { class: 'stack-row' }, h('span', { text: name }),
        h('div', { class: 'stack-bar' }, ...vals.map((v, i) =>
          h('i', { title: `${ACTIONS[i]}: ${v}%`, style: `--v:${v};--c:${ACT_COLORS[i]};--d:${i * 0.07}s` }, v >= 6 ? h('span', { text: Math.round(v) + '%' }) : null)))));
    }));
  }

  const dumb = $('#chart-modality');
  if (dumb) {
    const ROWS = [
      { label: 'Inside contents', sub: 'without audio', full: 59.0, drop: 24.4, c: 'var(--orange)' },
      { label: 'Roughness', sub: 'without touch', full: 67.9, drop: 25.6, c: 'var(--green)' },
      { label: 'Weight', sub: 'without force', full: 91.5, drop: 46.4, c: 'var(--purple)' },
    ];
    mount(dumb, (el) => ROWS.forEach((r) => {
      el.append(h('div', { class: 'db-row', style: `--c:${r.c}` },
        h('div', { class: 'db-label' }, r.label, ' ', h('small', { text: `· ${r.sub}` })),
        h('div', { class: 'db-track' },
          h('div', { class: 'db-line', style: `--lo:${r.drop};--hi:${r.full}` }),
          h('div', { class: 'db-dot drop', style: `--x:${r.drop}` }, h('em', { text: r.drop.toFixed(1) })),
          h('div', { class: 'db-dot full', style: `--x:${r.full}` }, h('em', { text: r.full.toFixed(1) })))));
    }));
  }

  const abl = $('#chart-ablation');
  if (abl) {
    const ROWS = [
      { label: 'w/o handheld data', value: 2.0, color: 'var(--blue)' },
      { label: 'w/o token initialization', value: 1.1, color: 'var(--orange)' },
      { label: 'w/o dual alignment', value: 1.8, color: 'var(--green)' },
      { label: 'w/o dynamic sampling', value: 1.4, color: 'var(--purple)' },
    ];
    mount(abl, (el) => hbars(el, ROWS, { max: 2.4, fmtv: (v) => '−' + v.toFixed(1) }));
  }

  /* ---------------- real-robot demos: control bar below the video ---------------- */
  $$('.rw-player').forEach((box) => {
    const v = $('video', box), seek = $('.rw-seek', box), time = $('.rw-time', box);
    const t = (x) => `${Math.floor(x / 60)}:${String(Math.floor(x % 60)).padStart(2, '0')}`;
    const draw = () => {
      const d = v.duration || 0;
      seek.value = d ? (v.currentTime / d) * 1000 : 0;
      seek.style.setProperty('--p', `${seek.value / 10}%`);
      time.textContent = `${t(v.currentTime)} / ${t(d)}`;
    };
    const state = () => {
      box.classList.toggle('is-playing', !v.paused && !v.ended);
      box.classList.toggle('is-muted', v.muted || v.volume === 0);
      $('.rw-play', box).setAttribute('aria-label', v.paused ? 'Play' : 'Pause');
    };
    const toggle = () => { if (v.paused || v.ended) v.play().catch(() => {}); else v.pause(); };
    $('.rw-play', box).addEventListener('click', toggle);
    v.addEventListener('click', toggle);
    $('.rw-mute', box).addEventListener('click', () => { v.muted = !v.muted; });
    $('.rw-fs', box).addEventListener('click', () => {
      if (document.fullscreenElement) document.exitFullscreen();
      else (box.requestFullscreen || box.webkitRequestFullscreen).call(box);
    });
    seek.addEventListener('input', () => {
      if (v.duration) v.currentTime = (seek.value / 1000) * v.duration;
      draw();
    });
    ['timeupdate', 'loadedmetadata', 'durationchange'].forEach((ev) => v.addEventListener(ev, draw));
    ['play', 'pause', 'ended', 'volumechange'].forEach((ev) => v.addEventListener(ev, state));
    // only one demo plays at a time
    v.addEventListener('play', () => $$('.rw-player video').forEach((o) => { if (o !== v) o.pause(); }));
    draw(); state();
  });

  /* ------------------------------------------------------------------ *
   * BibTeX copy
   * ------------------------------------------------------------------ */
  const copy = $('#copy-bib');
  if (copy) {
    copy.addEventListener('click', async () => {
      const text = $('#bibtex').textContent;
      try { await navigator.clipboard.writeText(text); }
      catch {
        const r = document.createRange(); r.selectNodeContents($('#bibtex'));
        const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r); document.execCommand('copy'); sel.removeAllRanges();
      }
      copy.textContent = 'Copied';
      setTimeout(() => { copy.textContent = 'Copy'; }, 1600);
    });
  }
})();
