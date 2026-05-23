// Static manifest consumed by js/playground.js
window.WEBSITE_DATA = {
  hero: [],
  slider: [
    { slug: "bear-tree",     prompt: "A bear climbing a tree." },
    { slug: "shanghai-zoom", prompt: "The bund Shanghai, zoom in." },
    { slug: "burger",        prompt: "A person eating a burger." },
    { slug: "gwen-stacy",    prompt: "Gwen Stacy reading a book, tilt up." }
  ],
  // Foresight Readout Probe — currently uses placeholder shimmer cells until
  // real readout assets are rendered; swap in real PNGs under assets/probe/.
  probe: {
    prompts: [
      { slug: "violet", label: "Fluffy purple-tail animated creature" },
      { slug: "fox",    label: "Cyberpunk fox" },
      { slug: "ging",   label: "Gingerbread figurine" },
      { slug: "walker", label: "Walker" }
    ],
    // For each (slug, delta in 1..3) we have 4 images:
    //   cur, base, ours, fut.
    pathTemplate: "assets/probe/{slug}-d{delta}-{kind}.jpg"
  },
  window: {
    prompts: [
      { slug: "bigfoot",  label: "Bigfoot in snowstorm" },
      { slug: "painting", label: "Van Gogh painting in the room" }
    ],
    windows: [
      { key: "w0",   label: "{0}",      desc: "current only" },
      { key: "w1",   label: "{1}",      desc: "next only" },
      { key: "w01",  label: "{0,1}",    desc: "default — current + next" },
      { key: "w012", label: "{0,1,2}",  desc: "longer horizon" }
    ],
    // Numbers transcribed from Table 1 (foresight_window.tex). Quality / Semantic / Total.
    metrics: {
      w0:   { quality: 85.07, semantic: 81.55, total: 84.36 },
      w1:   { quality: 84.82, semantic: 81.50, total: 84.15 },
      w01:  { quality: 85.38, semantic: 81.59, total: 84.62 },
      w012: { quality: 85.11, semantic: 82.00, total: 84.49 }
    }
  },
  results: [
    { method: "Self-Forcing (chunk-wise)",      q: 84.37, s: 80.87, t: 83.67, sc: 89.83, bc: 92.72, oc: 25.02, ours: false },
    { method: "  + Video-Mirai",                q: 84.82, s: 81.45, t: 84.15, sc: 91.62, bc: 93.77, oc: 25.33, ours: true,  bold: ["q","s","t","sc","bc","oc"] },
    { method: "Causal-Forcing (frame-wise)",    q: 83.16, s: 78.73, t: 82.27, sc: 75.60, bc: 84.41, oc: 23.25, ours: false },
    { method: "  + Video-Mirai",                q: 84.21, s: 79.41, t: 83.25, sc: 76.90, bc: 85.07, oc: 23.66, ours: true,  bold: ["q","s","t","sc","bc","oc"] },
    { method: "Causal-Forcing (chunk-wise)",    q: 84.55, s: 80.92, t: 83.82, sc: 84.93, bc: 90.22, oc: 24.93, ours: false },
    { method: "  + Video-Mirai",                q: 85.38, s: 81.59, t: 84.62, sc: 88.47, bc: 91.94, oc: 25.03, ours: true,  bold: ["q","s","t","sc","bc","oc"] }
  ],
  gallery5s: [
    { slug: "rank118", prompt: "A koala bear playing piano in the forest." },
    { slug: "rank15", prompt: "Skyscraper." },
    { slug: "rank110", prompt: "Aerial panoramic video from a drone of a fantasy land." },
    { slug: "rank25", prompt: "Nursery." },
    { slug: "rank30", prompt: "A person is climbing a rope." },
    { slug: "rank40", prompt: "A giraffe and a bird." }
  ],
  gallery30s: [
    { slug: "m018", prompt: "An astronaut runs on the surface of the moon — low-angle shot, smooth and lightweight movement." },
    { slug: "m002", prompt: "A snowboarder accelerating down a powdery slope, weaving between trees." },
    { slug: "m007", prompt: "A woman in purple overalls and cowboy boots strolls in Mumbai during a winter storm." },
    { slug: "m029", prompt: "An adorable kangaroo in purple overalls and cowboy boots strolls through Mumbai during a colorful festival." },
    { slug: "m006", prompt: "Handheld shot navigating through a bustling market, weaving between stalls." },
    { slug: "m028", prompt: "Rural road in China at night — sky filled with stars, moon hanging high." }
  ]
};
