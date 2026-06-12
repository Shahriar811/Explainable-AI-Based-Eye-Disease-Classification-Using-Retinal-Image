"""
Friendly browser UI for the CSE498 Eye Disease Detection API.

Served at the API root ("/") so anyone opening the public ngrok URL sees a
clean, easy page instead of the developer-oriented Swagger /docs view.
"""

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CSE498 — Eye Disease Detection</title>
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Ccircle cx='32' cy='32' r='28' fill='%23146aff'/%3E%3Ccircle cx='32' cy='32' r='10' fill='white'/%3E%3C/svg%3E" />
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    html, body {
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                   "Helvetica Neue", Arial, sans-serif;
      color: #1a2a44;
      background: linear-gradient(135deg, #eef4ff 0%, #f7fbff 50%, #f3f0ff 100%);
      min-height: 100vh;
    }
    body { padding: 24px 16px 48px; }

    .wrap { max-width: 1080px; margin: 0 auto; }

    header.hero {
      text-align: center;
      padding: 8px 0 28px;
    }
    header.hero .badge {
      display: inline-block;
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #4060a8;
      background: #e6efff;
      border-radius: 999px;
      padding: 6px 14px;
      margin-bottom: 14px;
    }
    header.hero h1 {
      font-size: clamp(28px, 4vw, 40px);
      margin: 0 0 10px;
      letter-spacing: -0.01em;
    }
    header.hero p.tagline {
      margin: 0;
      color: #5a6b8a;
      font-size: 16px;
    }

    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 22px;
      margin-top: 22px;
    }
    @media (max-width: 880px) {
      .grid { grid-template-columns: 1fr; }
    }

    .card {
      background: #ffffff;
      border: 1px solid #e6eaf3;
      border-radius: 18px;
      padding: 22px 22px 24px;
      box-shadow: 0 8px 24px rgba(20, 40, 90, 0.06);
    }

    .step {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 13px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #4060a8;
      margin: 0 0 10px;
    }
    .step .dot {
      width: 22px;
      height: 22px;
      border-radius: 50%;
      background: #146aff;
      color: white;
      font-size: 12px;
      font-weight: 700;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    h2.section-title {
      margin: 0 0 14px;
      font-size: 18px;
      letter-spacing: -0.005em;
    }

    select, button {
      font-family: inherit;
      font-size: 15px;
    }
    select#modelSelect {
      width: 100%;
      padding: 12px 14px;
      border: 1.5px solid #d6dcec;
      border-radius: 12px;
      background: #fbfcff;
      color: #1a2a44;
      appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='8' viewBox='0 0 14 8'%3E%3Cpath fill='%234060a8' d='M7 8 0 0h14z'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 14px center;
      padding-right: 40px;
      cursor: pointer;
    }
    select#modelSelect:focus {
      outline: none;
      border-color: #146aff;
      box-shadow: 0 0 0 4px #d4e3ff;
    }
    .meta {
      margin: 10px 2px 0;
      color: #5a6b8a;
      font-size: 13px;
      min-height: 18px;
    }
    .classes {
      margin-top: 6px;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .chip {
      font-size: 12px;
      background: #eef4ff;
      color: #2a4585;
      border-radius: 999px;
      padding: 4px 10px;
    }

    .upload-block { margin-top: 22px; }
    .drop {
      border: 2px dashed #b7c3df;
      border-radius: 16px;
      padding: 28px 18px;
      text-align: center;
      background: #fafcff;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .drop:hover, .drop.hover {
      border-color: #146aff;
      background: #f0f6ff;
    }
    .drop .icon {
      font-size: 28px;
      color: #146aff;
      margin-bottom: 6px;
    }
    .drop .primary {
      font-weight: 600;
      color: #1a2a44;
    }
    .drop .secondary {
      color: #5a6b8a;
      font-size: 13px;
      margin-top: 4px;
    }
    .drop.has-file {
      border-style: solid;
      border-color: #146aff;
      background: #ffffff;
    }
    .preview-wrap {
      margin-top: 14px;
      display: none;
      text-align: center;
    }
    .preview-wrap.show { display: block; }
    .preview-wrap img {
      max-width: 100%;
      max-height: 260px;
      border-radius: 12px;
      border: 1px solid #e6eaf3;
    }
    .file-name {
      margin-top: 8px;
      font-size: 13px;
      color: #5a6b8a;
      word-break: break-all;
    }

    .xai-toggle {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-top: 18px;
      padding: 12px 14px;
      border: 1px solid #e6eaf3;
      border-radius: 12px;
      background: #fbfcff;
      cursor: pointer;
      user-select: none;
    }
    .xai-toggle:hover { border-color: #c8d4f0; }
    .xai-toggle input[type="checkbox"] {
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }
    .toggle-track {
      flex: 0 0 auto;
      width: 38px;
      height: 22px;
      border-radius: 999px;
      background: #d6dcec;
      position: relative;
      transition: background 0.2s ease;
    }
    .toggle-knob {
      position: absolute;
      top: 2px;
      left: 2px;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: #fff;
      box-shadow: 0 2px 4px rgba(20, 40, 90, 0.2);
      transition: left 0.2s ease;
    }
    .xai-toggle input:checked ~ .toggle-track {
      background: linear-gradient(135deg, #146aff, #4f8bff);
    }
    .xai-toggle input:checked ~ .toggle-track .toggle-knob {
      left: 18px;
    }
    .xai-toggle input:focus-visible ~ .toggle-track {
      box-shadow: 0 0 0 4px #d4e3ff;
    }
    .toggle-text {
      display: flex;
      flex-direction: column;
      gap: 2px;
      font-size: 14px;
      color: #1a2a44;
    }
    .toggle-text small {
      color: #5a6b8a;
      font-size: 12px;
    }

    .actions {
      margin-top: 18px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    button.primary {
      flex: 1 1 auto;
      min-width: 160px;
      padding: 13px 18px;
      border: none;
      border-radius: 12px;
      background: linear-gradient(135deg, #146aff, #4f8bff);
      color: white;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.05s ease, box-shadow 0.15s ease, opacity 0.15s ease;
      box-shadow: 0 6px 18px rgba(20, 106, 255, 0.25);
    }
    button.primary:hover { box-shadow: 0 8px 24px rgba(20, 106, 255, 0.35); }
    button.primary:active { transform: translateY(1px); }
    button.primary:disabled {
      opacity: 0.55;
      cursor: not-allowed;
      box-shadow: none;
    }
    button.ghost {
      padding: 13px 16px;
      border: 1.5px solid #d6dcec;
      border-radius: 12px;
      background: white;
      color: #1a2a44;
      cursor: pointer;
    }
    button.ghost:hover { border-color: #146aff; color: #146aff; }

    .results .empty {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      color: #5a6b8a;
      padding: 36px 8px;
    }
    .results .empty .blob {
      width: 64px;
      height: 64px;
      border-radius: 50%;
      background: radial-gradient(circle at 30% 30%, #b8d2ff, #146aff);
      opacity: 0.85;
      margin-bottom: 12px;
    }

    .pred-summary {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 18px;
    }
    @media (max-width: 480px) {
      .pred-summary { grid-template-columns: 1fr; }
    }
    .stat {
      background: linear-gradient(135deg, #f3f7ff, #ecf2ff);
      border-radius: 14px;
      padding: 14px 16px;
    }
    .stat .lbl {
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #4060a8;
      margin-bottom: 4px;
    }
    .stat .val {
      font-size: 22px;
      font-weight: 700;
      color: #1a2a44;
      line-height: 1.2;
      word-break: break-word;
    }
    .stat.confidence .val { color: #146aff; }

    h3.subtitle {
      font-size: 14px;
      color: #4060a8;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin: 16px 0 10px;
    }
    .bars { display: flex; flex-direction: column; gap: 10px; }
    .bar .row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 14px;
      margin-bottom: 4px;
    }
    .bar .row .name { color: #1a2a44; font-weight: 500; }
    .bar .row .pct { color: #4060a8; font-variant-numeric: tabular-nums; }
    .bar .track {
      width: 100%;
      height: 8px;
      background: #eef2fb;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar .fill {
      height: 100%;
      background: linear-gradient(90deg, #146aff, #4f8bff);
      border-radius: 999px;
      transition: width 0.4s ease;
    }
    .bar.top .fill {
      background: linear-gradient(90deg, #0a55d6, #146aff);
    }

    .alert {
      margin-top: 14px;
      padding: 10px 14px;
      background: #fff3f0;
      color: #a13225;
      border: 1px solid #ffd7cd;
      border-radius: 10px;
      font-size: 14px;
      display: none;
    }
    .alert.show { display: block; }

    .xai-block { margin-top: 22px; }
    .xai-note {
      color: #5a6b8a;
      font-size: 13px;
      margin: 0 0 12px;
      line-height: 1.5;
    }
    .xai-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    @media (max-width: 520px) {
      .xai-grid { grid-template-columns: 1fr; }
    }
    .xai-figure {
      margin: 0;
      border: 1px solid #e6eaf3;
      border-radius: 12px;
      overflow: hidden;
      background: #fafcff;
    }
    .xai-figure img {
      display: block;
      width: 100%;
      height: auto;
      aspect-ratio: 1 / 1;
      object-fit: cover;
    }
    .xai-figure figcaption {
      text-align: center;
      font-size: 12px;
      color: #4060a8;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      padding: 8px 4px;
      border-top: 1px solid #eef2fb;
      background: #fff;
    }

    .spinner {
      width: 22px; height: 22px;
      border: 3px solid #cfe0ff;
      border-top-color: #146aff;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      display: inline-block;
      vertical-align: middle;
      margin-right: 8px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .loading-row {
      display: none;
      align-items: center;
      color: #4060a8;
      font-size: 14px;
      margin-top: 10px;
    }
    .loading-row.show { display: flex; }

    footer {
      text-align: center;
      color: #6b7a99;
      font-size: 12px;
      margin-top: 32px;
    }
    footer a { color: #4060a8; text-decoration: none; }
    footer a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <span class="badge">CSE498 Research</span>
      <h1>Eye Disease Detection</h1>
      <p class="tagline">Pick a model, upload a retinal image, get an instant prediction with an explainable heatmap.</p>
    </header>

    <div class="grid">
      <section class="card">
        <p class="step"><span class="dot">1</span> Choose a model</p>
        <select id="modelSelect" aria-label="Select model">
          <option value="">Loading models…</option>
        </select>
        <div class="meta" id="modelMeta"></div>
        <div class="classes" id="modelClasses"></div>

        <div class="upload-block">
          <p class="step"><span class="dot">2</span> Upload a retinal image</p>
          <div class="drop" id="dropZone" tabindex="0">
            <div class="icon">⬆</div>
            <div class="primary" id="dropPrimary">Drop an image here, or click to browse</div>
            <div class="secondary">JPG, PNG, BMP, WEBP, TIFF · processed locally on the model server</div>
          </div>
          <input type="file" id="fileInput" accept="image/*" hidden />
          <div class="preview-wrap" id="previewWrap">
            <img id="preview" alt="Preview" />
            <div class="file-name" id="fileName"></div>
          </div>
        </div>

        <label class="xai-toggle" for="xaiToggle">
          <input type="checkbox" id="xaiToggle" checked />
          <span class="toggle-track" aria-hidden="true"><span class="toggle-knob"></span></span>
          <span class="toggle-text">
            <strong>Show visual explanation (Grad-CAM++)</strong>
            <small>Highlights which retinal regions influenced the prediction. Adds ~1–2 seconds.</small>
          </span>
        </label>

        <div class="actions">
          <button class="primary" id="predictBtn" disabled>Predict</button>
          <button class="ghost" id="resetBtn" type="button">Reset</button>
        </div>
        <div class="loading-row" id="loadingRow"><span class="spinner"></span><span id="loadingText">Running model…</span></div>
        <div class="alert" id="alertBox"></div>
      </section>

      <section class="card results" id="resultsCard">
        <p class="step"><span class="dot">3</span> Result</p>
        <div class="empty" id="emptyState">
          <div class="blob"></div>
          <div>Your prediction will appear here.</div>
        </div>
        <div id="resultBody" style="display:none;">
          <div class="pred-summary">
            <div class="stat">
              <div class="lbl">Predicted class</div>
              <div class="val" id="predClass">—</div>
            </div>
            <div class="stat confidence">
              <div class="lbl">Confidence</div>
              <div class="val" id="confidence">—</div>
            </div>
          </div>
          <h3 class="subtitle">Top probabilities</h3>
          <div class="bars" id="topK"></div>

          <div id="xaiBlock" class="xai-block" style="display:none;">
            <h3 class="subtitle">Visual explanation (Grad-CAM++)</h3>
            <p class="xai-note">Red regions had the most influence on the model's decision. Yellow / green regions had medium influence; blue / dark regions had little.</p>
            <div class="xai-grid">
              <figure class="xai-figure">
                <img id="xaiOriginal" alt="Original (resized to model input)" />
                <figcaption>Input</figcaption>
              </figure>
              <figure class="xai-figure">
                <img id="xaiOverlay" alt="Grad-CAM++ heatmap overlay" />
                <figcaption>Grad-CAM++ overlay</figcaption>
              </figure>
            </div>
          </div>
        </div>
      </section>
    </div>

    <footer>
      Powered by FastAPI · <a href="/docs">API reference</a>
    </footer>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);
    const state = { models: [], file: null };

    function showAlert(msg) {
      const a = $('alertBox');
      a.textContent = msg;
      a.classList.add('show');
    }
    function clearAlert() {
      $('alertBox').classList.remove('show');
    }

    async function loadModels() {
      const sel = $('modelSelect');
      try {
        const res = await fetch('/models');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        state.models = data;
        sel.innerHTML = '';
        if (!data.length) {
          sel.innerHTML = '<option value="">(no models found on server)</option>';
          $('modelMeta').textContent = 'No model bundles were found in the configured model directory.';
          return;
        }
        data.forEach((m) => {
          const opt = document.createElement('option');
          opt.value = m.id;
          const arch = m.architecture || 'Unknown architecture';
          opt.textContent = arch;
          sel.appendChild(opt);
        });
        sel.addEventListener('change', renderMeta);
        renderMeta();
        updatePredictEnabled();
      } catch (e) {
        sel.innerHTML = '<option value="">(failed to load models)</option>';
        showAlert('Could not load models from server: ' + e.message);
      }
    }

    function renderMeta() {
      const sel = $('modelSelect');
      const id = sel.value;
      const m = state.models.find((x) => x.id === id);
      const meta = $('modelMeta');
      const chips = $('modelClasses');
      chips.innerHTML = '';
      if (!m) {
        meta.textContent = '';
        return;
      }
      const parts = [];
      if (m.num_classes) parts.push(m.num_classes + ' classes');
      if (m.image_size) parts.push('Input ' + m.image_size + '×' + m.image_size + ' px');
      parts.push('File: ' + m.id);
      meta.textContent = parts.join(' · ');
      if (Array.isArray(m.class_names)) {
        m.class_names.forEach((c) => {
          const chip = document.createElement('span');
          chip.className = 'chip';
          chip.textContent = c;
          chips.appendChild(chip);
        });
      }
    }

    function attachUpload() {
      const dz = $('dropZone');
      const fi = $('fileInput');
      const setFile = (f) => {
        if (!f) return;
        if (!f.type.startsWith('image/')) {
          showAlert('Please choose an image file.');
          return;
        }
        clearAlert();
        state.file = f;
        const url = URL.createObjectURL(f);
        $('preview').src = url;
        $('previewWrap').classList.add('show');
        $('fileName').textContent = f.name;
        dz.classList.add('has-file');
        $('dropPrimary').textContent = 'Image ready · click to change';
        $('xaiBlock').style.display = 'none';
        updatePredictEnabled();
      };
      dz.addEventListener('click', () => fi.click());
      dz.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fi.click(); }
      });
      fi.addEventListener('change', (e) => setFile(e.target.files[0]));
      ['dragenter', 'dragover'].forEach((evt) =>
        dz.addEventListener(evt, (e) => {
          e.preventDefault();
          dz.classList.add('hover');
        })
      );
      ['dragleave', 'dragend'].forEach((evt) =>
        dz.addEventListener(evt, (e) => {
          e.preventDefault();
          dz.classList.remove('hover');
        })
      );
      dz.addEventListener('drop', (e) => {
        e.preventDefault();
        dz.classList.remove('hover');
        const f = e.dataTransfer.files && e.dataTransfer.files[0];
        setFile(f);
      });
    }

    function updatePredictEnabled() {
      const ok = !!state.file && !!$('modelSelect').value;
      $('predictBtn').disabled = !ok;
    }

    function resetAll() {
      state.file = null;
      $('fileInput').value = '';
      $('preview').src = '';
      $('previewWrap').classList.remove('show');
      $('fileName').textContent = '';
      $('dropZone').classList.remove('has-file');
      $('dropPrimary').textContent = 'Drop an image here, or click to browse';
      $('resultBody').style.display = 'none';
      $('emptyState').style.display = 'flex';
      $('xaiBlock').style.display = 'none';
      $('xaiOriginal').src = '';
      $('xaiOverlay').src = '';
      clearAlert();
      updatePredictEnabled();
    }

    async function predict() {
      clearAlert();
      const modelId = $('modelSelect').value;
      if (!state.file) { showAlert('Please upload an image first.'); return; }
      if (!modelId) { showAlert('Please select a model.'); return; }

      const wantXai = $('xaiToggle').checked;
      const endpoint = wantXai ? '/predict_xai' : '/predict';

      $('predictBtn').disabled = true;
      $('loadingRow').classList.add('show');
      $('loadingText').textContent = wantXai
        ? 'Running model and computing explanation…'
        : 'Running model…';
      $('resultBody').style.display = 'none';
      $('emptyState').style.display = 'flex';

      const fd = new FormData();
      fd.append('file', state.file);

      try {
        const res = await fetch(
          endpoint + '?model_id=' + encodeURIComponent(modelId) + '&top_k=5',
          { method: 'POST', body: fd }
        );
        if (!res.ok) {
          const txt = await res.text();
          throw new Error('HTTP ' + res.status + ' — ' + txt.slice(0, 200));
        }
        const data = await res.json();
        renderResult(data);
      } catch (e) {
        showAlert('Prediction failed: ' + e.message);
      } finally {
        $('loadingRow').classList.remove('show');
        updatePredictEnabled();
      }
    }

    function renderResult(data) {
      const top = Array.isArray(data.top_k) ? data.top_k : [];
      const predLabel = data.predicted_label || ('Class #' + data.predicted_index);
      let topConf = 0;
      if (top.length && typeof top[0].prob === 'number') {
        topConf = top[0].prob;
      } else if (Array.isArray(data.probabilities)) {
        topConf = data.probabilities[data.predicted_index] || 0;
      }
      $('predClass').textContent = predLabel;
      $('confidence').textContent = (topConf * 100).toFixed(2) + '%';

      const tk = $('topK');
      tk.innerHTML = '';
      top.forEach((row, i) => {
        const pct = (row.prob * 100).toFixed(2);
        const item = document.createElement('div');
        item.className = 'bar' + (i === 0 ? ' top' : '');
        item.innerHTML =
          '<div class="row">'
          + '<span class="name"></span>'
          + '<span class="pct">' + pct + '%</span>'
          + '</div>'
          + '<div class="track"><div class="fill" style="width:' + pct + '%"></div></div>';
        item.querySelector('.name').textContent = row.label || ('Class #' + row.index);
        tk.appendChild(item);
      });

      const xaiBlock = $('xaiBlock');
      if (data.overlay_png_b64 && state.file) {
        $('xaiOriginal').src = $('preview').src;
        $('xaiOverlay').src = 'data:image/png;base64,' + data.overlay_png_b64;
        xaiBlock.style.display = 'block';
      } else {
        xaiBlock.style.display = 'none';
      }

      $('emptyState').style.display = 'none';
      $('resultBody').style.display = 'block';
    }

    document.addEventListener('DOMContentLoaded', () => {
      loadModels();
      attachUpload();
      $('predictBtn').addEventListener('click', predict);
      $('resetBtn').addEventListener('click', resetAll);
      $('modelSelect').addEventListener('change', updatePredictEnabled);
    });
  </script>
</body>
</html>
"""
