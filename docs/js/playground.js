/* Video-Mirai project page — all interactivity */
(function () {
  const D = window.WEBSITE_DATA;
  if (!D) { console.error("manifest missing"); return; }

  // ============================================================
  // HERO — rotate through clips, fade caption
  // ============================================================
  (function hero() {
    const v = document.getElementById("hero-video");
    const c = document.getElementById("hero-caption");
    if (!v || !D.hero?.length) return;
    let i = 0;
    function show() {
      v.src = D.hero[i].src;
      c.textContent = D.hero[i].caption;
      v.play().catch(() => {});
    }
    v.addEventListener("ended", () => { i = (i + 1) % D.hero.length; show(); });
    // mp4 loops by default; we rotate every 12s instead.
    v.removeAttribute("loop");
    setInterval(() => { i = (i + 1) % D.hero.length; show(); }, 12000);
    show();
  })();

  // ============================================================
  // DEMO 1 — Side-by-side baseline vs. Video-Mirai, time-synced
  // ============================================================
  (function sideBySide() {
    const sel = document.getElementById("gap-prompt");
    const vb  = document.getElementById("vid-base");
    const vo  = document.getElementById("vid-ours");
    if (!sel || !vb || !vo) return;

    D.slider.forEach((p, i) => {
      const o = document.createElement("option");
      o.value = i;
      o.textContent = p.prompt.length > 70 ? p.prompt.slice(0, 67) + "…" : p.prompt;
      sel.appendChild(o);
    });

    let curr = 0;
    function load() {
      const s = D.slider[curr];
      vb.src = `videos/slider/${s.slug}-base.mp4`;
      vo.src = `videos/slider/${s.slug}-ours.mp4`;
      const sync = () => { try { vo.currentTime = vb.currentTime; } catch (_) {} };
      vb.addEventListener("loadedmetadata", sync, { once: true });
      vb.play().catch(() => {});
      vo.play().catch(() => {});
    }
    sel.addEventListener("change", () => { curr = parseInt(sel.value, 10); load(); });
    document.getElementById("gap-prev").addEventListener("click", () => {
      curr = (curr - 1 + D.slider.length) % D.slider.length; sel.value = curr; load();
    });
    document.getElementById("gap-next").addEventListener("click", () => {
      curr = (curr + 1) % D.slider.length; sel.value = curr; load();
    });
    document.getElementById("gap-sync").addEventListener("click", () => {
      vo.currentTime = vb.currentTime; vb.play(); vo.play();
    });

    // Periodic resync — different decoders / different keyframes can drift.
    setInterval(() => {
      if (!vb.paused && Math.abs(vb.currentTime - vo.currentTime) > 0.15) {
        vo.currentTime = vb.currentTime;
      }
    }, 1000);

    load();
  })();

  // ============================================================
  // DEMO 4 — (removed) The Method section now renders the
  //   paper figure directly via <img> in index.html.
  // ============================================================
  (function diagram() {
    const root = document.getElementById("diagram");
    if (!root) return;
    // Build an SVG inline. Coordinates in a 800x500 viewBox.
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 800 500");
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "100%");
    svg.innerHTML = `
      <defs>
        <marker id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#0b0d12"/>
        </marker>
        <marker id="arrowhead-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#FF0000"/>
        </marker>
      </defs>

      <!-- Row 1: Causal DiT rollout -->
      <text x="30" y="60" class="badge">CAUSAL DiT — generates segments left to right</text>
      <g id="g-x1"><rect class="node" x="30"  y="80" rx="10" ry="10" width="120" height="70"/><text class="node-text" x="90"  y="110" text-anchor="middle">X₁ (past)</text><text class="node-text" x="90"  y="128" text-anchor="middle" fill="#6b7280">noise→clean</text></g>
      <g id="g-x2"><rect class="node" x="180" y="80" rx="10" ry="10" width="120" height="70" fill="#dff4fd" stroke="#00B0F0"/><text class="node-text" x="240" y="110" text-anchor="middle">X₂ (current)</text><text class="node-text" x="240" y="128" text-anchor="middle" fill="#00B0F0">h₂ᴸ ← layer L</text></g>
      <g id="g-x3"><rect class="node" x="330" y="80" rx="10" ry="10" width="120" height="70"/><text class="node-text" x="390" y="110" text-anchor="middle">X₃ (future)</text><text class="node-text" x="390" y="128" text-anchor="middle" fill="#6b7280">rolled out</text></g>

      <path id="a-x1-x2" class="arrow arrow-causal" d="M 150 115 L 180 115"/>
      <path id="a-x2-x3" class="arrow arrow-causal" d="M 300 115 L 330 115"/>

      <!-- Row 2: Foresight Encoder -->
      <text x="30" y="225" class="badge train-only">FROZEN FORESIGHT ENCODER (Wan-14B) — reads full rollout, including X₃</text>
      <g id="g-enc" class="train-only">
        <rect class="node" x="30" y="245" rx="12" ry="12" width="420" height="60" fill="#fff5f5" stroke="#FF0000"/>
        <text class="node-text" x="240" y="282" text-anchor="middle" fill="#FF0000">Bidirectional encoder · outputs H₁ᴸ', H₂ᴸ', H₃ᴸ' at matched depth</text>
      </g>
      <path id="a-x1-enc" class="arrow arrow-foresight train-only" d="M 90  165 L 90  245" marker-end="url(#arrowhead-red)"/>
      <path id="a-x2-enc" class="arrow arrow-foresight train-only" d="M 240 165 L 240 245" marker-end="url(#arrowhead-red)"/>
      <path id="a-x3-enc" class="arrow arrow-foresight train-only" d="M 390 165 L 390 245" marker-end="url(#arrowhead-red)"/>

      <!-- Row 3: predictor + loss -->
      <g id="g-pred" class="train-only">
        <rect class="node" x="500" y="80" rx="10" ry="10" width="200" height="70" fill="#f6f4ff" stroke="#7C5CFF"/>
        <text class="node-text" x="600" y="110" text-anchor="middle" fill="#7C5CFF">φ — predictor (3-block DiT)</text>
        <text class="node-text" x="600" y="128" text-anchor="middle" fill="#6b7280">discarded at inference</text>
      </g>

      <path id="a-h2-pred" class="arrow arrow-causal train-only" d="M 300 115 L 500 115"/>

      <g id="g-target" class="train-only">
        <rect class="node" x="500" y="245" rx="10" ry="10" width="200" height="60" fill="#fff5f5" stroke="#FF0000"/>
        <text class="node-text" x="600" y="280" text-anchor="middle" fill="#FF0000">H̄₂ = ½ (H₂ᴸ' + H₃ᴸ')</text>
      </g>
      <path id="a-enc-target" class="arrow arrow-foresight train-only" d="M 450 275 L 500 275" marker-end="url(#arrowhead-red)"/>

      <path id="a-pred-loss"  class="arrow arrow-foresight train-only" d="M 600 150 L 600 245" marker-end="url(#arrowhead-red)"/>
      <text id="loss-label"  class="label-loss train-only" x="610" y="200">ℓᶠ = 1 − cos(·, sg[H̄])</text>

      <!-- Row 4: inference-only label -->
      <g id="g-infer-banner" class="infer-only" style="display:none">
        <rect class="node" x="30" y="370" rx="12" ry="12" width="740" height="80" fill="#0b0d12" stroke="#0b0d12"/>
        <text x="400" y="400" text-anchor="middle" fill="#fafaf7" font-family="Inter" font-size="16" font-weight="600">At inference, the encoder & predictor are discarded.</text>
        <text x="400" y="425" text-anchor="middle" fill="#fafaf7" font-family="JetBrains Mono" font-size="11" fill-opacity="0.7">FLOPs · params · KV-cache — identical to the baseline causal generator.</text>
      </g>
    `;
    root.appendChild(svg);

    const note = document.getElementById("mode-note");

    function setMode(mode) {
      const train = mode === "train";
      root.querySelectorAll(".train-only").forEach(el => {
        el.style.opacity = train ? "1" : "0.08";
      });
      const banner = root.querySelector("#g-infer-banner");
      if (banner) banner.style.display = train ? "none" : "block";
      note.textContent = train
        ? "Future supervises the present (red dashed = stop-grad target)"
        : "Strictly causal · identical FLOPs / KV-cache to the baseline";
      document.querySelectorAll(".mode-btn").forEach(b => {
        const active = b.dataset.mode === mode;
        b.classList.toggle("bg-ink", active);
        b.classList.toggle("text-paper", active);
        b.classList.toggle("border", !active);
        b.classList.toggle("border-line", !active);
      });
    }
    document.querySelectorAll(".mode-btn").forEach(b =>
      b.addEventListener("click", () => setMode(b.dataset.mode))
    );
    setMode("train");
  })();

  // ============================================================
  // DEMO 2 — Foresight readout probe
  // ============================================================
  (function probe() {
    const cur   = document.getElementById("probe-cur");
    const base  = document.getElementById("probe-base");
    const ours  = document.getElementById("probe-ours");
    const fut   = document.getElementById("probe-fut");
    const chips = document.getElementById("probe-prompts");
    const slider = document.getElementById("probe-delta");
    const sliderVal = document.getElementById("probe-delta-val");
    if (!cur || !D.probe) return;

    const T = D.probe.pathTemplate;
    const fill = (img, slug, delta, kind) => {
      const url = T.replace("{slug}", slug).replace("{delta}", delta).replace("{kind}", kind);
      img.dataset.target = url;
      // Optimistic load; on error swap to a shimmer placeholder so the UI stays clean.
      img.onerror = () => {
        img.removeAttribute("src");
        img.parentElement.classList.add("shimmer");
      };
      img.onload = () => img.parentElement.classList.remove("shimmer");
      img.src = url;
    };

    let active = 0;
    const render = () => {
      const slug  = D.probe.prompts[active].slug;
      const delta = parseInt(slider.value, 10);
      sliderVal.textContent = delta;
      fill(cur,  slug, delta, "cur");
      fill(base, slug, delta, "base");
      fill(ours, slug, delta, "ours");
      fill(fut,  slug, delta, "fut");
    };
    D.probe.prompts.forEach((p, i) => {
      const btn = document.createElement("button");
      btn.className = "prompt-chip" + (i === 0 ? " active" : "");
      btn.textContent = p.label;
      btn.addEventListener("click", () => {
        active = i;
        chips.querySelectorAll(".prompt-chip").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        render();
      });
      chips.appendChild(btn);
    });
    slider.addEventListener("input", render);
    render();
  })();

  // ============================================================
  // DEMO 3 — Foresight window sandbox
  // ============================================================
  (function windowDemo() {
    const buttons = document.getElementById("window-buttons");
    const vid = document.getElementById("window-vid");
    const radarHost = document.getElementById("window-radar");
    if (!buttons || !vid) return;

    // Prompt switcher (small) + window switcher (large)
    let activePrompt = 0;
    let activeWindow = "w01";

    // Prompt chips at top
    const promptRow = document.createElement("div");
    promptRow.className = "flex flex-wrap gap-2 mb-3 basis-full";
    D.window.prompts.forEach((p, i) => {
      const b = document.createElement("button");
      b.className = "prompt-chip" + (i === 0 ? " active" : "");
      b.textContent = p.label;
      b.addEventListener("click", () => {
        activePrompt = i;
        promptRow.querySelectorAll(".prompt-chip").forEach(x => x.classList.remove("active"));
        b.classList.add("active");
        refresh();
      });
      promptRow.appendChild(b);
    });
    buttons.appendChild(promptRow);

    // Window buttons
    const winRow = document.createElement("div");
    winRow.className = "grid grid-cols-2 gap-2 basis-full";
    D.window.windows.forEach((w) => {
      const b = document.createElement("button");
      b.className = "rounded-2xl border border-line px-3 py-3 text-left hover:border-ink transition";
      b.innerHTML = `<div class="font-mono text-sm">Δ = ${w.label}</div><div class="text-xs text-muted mt-0.5">${w.desc}</div>`;
      b.addEventListener("click", () => {
        activeWindow = w.key;
        winRow.querySelectorAll("button").forEach(x => {
          x.classList.remove("bg-ink","text-paper","border-ink");
          x.classList.add("border-line");
        });
        b.classList.add("bg-ink","text-paper","border-ink");
        b.classList.remove("border-line");
        refresh();
      });
      if (w.key === activeWindow) {
        b.classList.add("bg-ink","text-paper","border-ink");
        b.classList.remove("border-line");
      }
      winRow.appendChild(b);
    });
    buttons.appendChild(winRow);

    function refresh() {
      const slug = D.window.prompts[activePrompt].slug;
      vid.src = `videos/window/${slug}-${activeWindow}.mp4`;
      vid.play().catch(()=>{});
      // Update radar plot
      const m = D.window.metrics[activeWindow];
      Plotly.react(radarHost, [{
        type: "scatterpolar",
        r: [m.quality, m.semantic, m.total, m.quality],   // close the loop
        theta: ["Quality", "Semantic", "Total", "Quality"],
        fill: "toself",
        line: { color: "#7C5CFF" },
        fillcolor: "rgba(124,92,255,0.18)"
      }], {
        polar: {
          radialaxis: { range: [80, 86], tickfont: { size: 9 } },
          angularaxis: { tickfont: { size: 10 } }
        },
        margin: { l: 20, r: 20, t: 8, b: 8 },
        showlegend: false,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: "Inter, sans-serif", size: 10 },
        autosize: true
      }, { displayModeBar: false, responsive: true });
    }
    refresh();
  })();

  // ============================================================
  // Results table
  // ============================================================
  (function results() {
    const tb = document.getElementById("results-tbody");
    if (!tb) return;
    const fmt = (v) => v.toFixed(2);
    D.results.forEach((row) => {
      const tr = document.createElement("tr");
      if (row.ours) tr.classList.add("ours");
      const isBold = (k) => Array.isArray(row.bold) && row.bold.includes(k);
      tr.innerHTML = `
        <td class="text-left">${row.method.replace(/^  /, "&nbsp;&nbsp;&nbsp;")}</td>
        <td class="text-right ${isBold("q")  ? "bold" : ""}">${fmt(row.q)}</td>
        <td class="text-right ${isBold("s")  ? "bold" : ""}">${fmt(row.s)}</td>
        <td class="text-right ${isBold("t")  ? "bold" : ""}">${fmt(row.t)}</td>
        <td class="text-right ${isBold("sc") ? "bold" : ""}">${fmt(row.sc)}</td>
        <td class="text-right ${isBold("bc") ? "bold" : ""}">${fmt(row.bc)}</td>
        <td class="text-right ${isBold("oc") ? "bold" : ""}">${fmt(row.oc)}</td>
      `;
      tb.appendChild(tr);
    });
  })();

  // ============================================================
  // Gallery — hover to play, swap on click
  // ============================================================
  (function gallery() {
    const renderGrid = (gridId, items) => {
      const grid = document.getElementById(gridId);
      if (!grid || !items) return;
      items.forEach((g) => {
        const card = document.createElement("div");
        card.className = "gallery-card group";
        card.dataset.state = "ours"; // start on ours
        const src = (state) => `videos/gallery/${g.slug}-${state}.mp4`;
        card.innerHTML = `
          <video autoplay muted loop playsinline preload="auto" data-src-ours="${src('ours')}" data-src-base="${src('base')}"></video>
          <div class="label">
            <span class="badge-state">VIDEO-MIRAI</span>
            <button class="toggle px-2 py-1 rounded-full bg-paper/20 hover:bg-paper/40 text-[10px]">⇄</button>
          </div>
          <div class="prompt">${g.prompt}</div>
        `;
        const v = card.querySelector("video");
        v.src = src("ours");
        v.play().catch(()=>{});  // start playing immediately on render
        // Toggle baseline vs ours
        card.querySelector(".toggle").addEventListener("click", (e) => {
          e.stopPropagation();
          const t = card.querySelector(".badge-state");
          if (card.dataset.state === "ours") {
            card.dataset.state = "base";
            v.src = v.dataset.srcBase;
            t.textContent = "BASELINE";
          } else {
            card.dataset.state = "ours";
            v.src = v.dataset.srcOurs;
            t.textContent = "VIDEO-MIRAI";
          }
          v.play().catch(()=>{});
        });
        grid.appendChild(card);
      });
    };
    renderGrid("gallery-grid-5s",  D.gallery5s);
    renderGrid("gallery-grid-30s", D.gallery30s);
  })();
})();
