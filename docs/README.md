# Video-Mirai · Project Page

Static project page for *Video-Mirai: Autoregressive Video Diffusion Models Need Foresight*.

Zero build step — plain HTML, CSS, JS, with Tailwind / KaTeX / Plotly via CDN.

## Run locally

```bash
# from docs/
python3 -m http.server 8765
# open http://localhost:8765
```

## Layout

```
docs/
├── index.html             # all sections (Hero · Gap · Method · Playground · Probe · Window · Results · Gallery · Cite)
├── css/styles.css         # custom theme on top of Tailwind
├── js/playground.js       # side-by-side demo, probe, window sandbox, table, gallery
├── data/manifest.js       # window.WEBSITE_DATA — all content, scores, prompt list, paths
├── assets/                # logo, og-image, and Demo 2 readout images (assets/probe/{slug}-d{0..3}-{cur|base|ours|fut}.jpg)
├── videos/                # curated MP4 assets consumed by manifest/playground.js
│   ├── slider/            # {slug}-base.mp4 / {slug}-ours.mp4 — 30 s pairs
│   ├── window/            # {slug}-{w0|w1|w01|w012}.mp4 — 5 s ablation
│   └── gallery/           # {slug}-base.mp4 / {slug}-ours.mp4
```

Nothing under `../` (the parent repo) is modified or read by JS at runtime — symlinks resolve at serve time.

## Refresh / add videos

Edit the prompt lists in `data/manifest.js`, add matching files under `videos/`,
and keep the slug/path pattern in sync with `js/playground.js`.

## Demo 2 (Foresight Readout Probe)

The probe renders real readout images. To refresh them, drop the 4-image set per
(prompt × Δ) into `assets/probe/` following the pattern
`{slug}-d{0..3}-{cur|base|ours|fut}.jpg` (matching `manifest.js → probe.pathTemplate`).

Rendering pipeline: train a layer-15 MLP readout on frozen Video-Mirai features
(see paper §"Representation probes" / Figure 1), then dump the 4-tile composite
for ~4 prompts × 4 horizons.

## Deploy

Served by GitHub Pages directly out of this `docs/` directory (the standard
GitHub Pages "from /docs" setting). Any static host also works — point it at
`docs/` as the publish dir.

Total payload ~67 MB (mostly the 40 mp4s).
