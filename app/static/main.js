/**
 * main.js — Music Genre Classifier UI logic
 *
 * Handles:
 *  - Drag & drop / file picker upload
 *  - WaveSurfer.js waveform rendering with play/pause
 *  - POST /predict call to Flask backend
 *  - Animated Chart.js bar chart for genre probabilities
 *  - Animated SVG confidence ring
 *  - Genre badge pop-in animation
 */

/* ── WaveSurfer instance ─────────────────────────────────────────────────── */
let wavesurfer = null;
let genreChart = null;
let currentFile = null;

/* ── DOM refs ─────────────────────────────────────────────────────────────── */
const dropZone     = document.getElementById('drop-zone');
const fileInput    = document.getElementById('file-input');
const fileName     = document.getElementById('file-name');
const uploadCard   = document.getElementById('upload-card');
const waveformCard = document.getElementById('waveform-card');
const resultsCard  = document.getElementById('results-card');
const playBtn      = document.getElementById('play-btn');
const currentTime  = document.getElementById('current-time');
const totalTime    = document.getElementById('total-time');
const modelSelect  = document.getElementById('model-select');
const classifyBtn  = document.getElementById('classify-btn');
const btnText      = document.getElementById('btn-text');
const spinner      = document.getElementById('spinner');
const resetBtn     = document.getElementById('reset-btn');
const genreBadge   = document.getElementById('genre-badge');
const genreNameEl  = document.getElementById('genre-name');
const confPct      = document.getElementById('confidence-pct');
const ringFill     = document.getElementById('ring-fill');

/* ── Inline SVG gradient (needed for ring stroke) ────────────────────────── */
document.querySelector('.confidence-ring').insertAdjacentHTML('afterbegin', `
  <defs>
    <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#8b5cf6"/>
      <stop offset="100%" stop-color="#ec4899"/>
    </linearGradient>
  </defs>
`);

/* ── Utility: format seconds as m:ss ────────────────────────────────────── */
function fmtTime(s) {
  const m = Math.floor(s / 60);
  const sec = String(Math.floor(s % 60)).padStart(2, '0');
  return `${m}:${sec}`;
}

/* ── Show / hide helpers ─────────────────────────────────────────────────── */
function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden'); }

/* ── Drag & drop wiring ──────────────────────────────────────────────────── */
['dragenter', 'dragover'].forEach(evt =>
  dropZone.addEventListener(evt, e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  })
);
['dragleave', 'drop'].forEach(evt =>
  dropZone.addEventListener(evt, e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
  })
);
dropZone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

/* ── Core: load a file into WaveSurfer ───────────────────────────────────── */
function handleFile(file) {
  currentFile = file;
  fileName.textContent = file.name;

  // Tear down previous wavesurfer if any
  if (wavesurfer) {
    wavesurfer.destroy();
    wavesurfer = null;
  }

  // Reset results
  hide(resultsCard);
  resetRing();

  // Show waveform card
  show(waveformCard);
  waveformCard.classList.remove('hidden');

  wavesurfer = WaveSurfer.create({
    container: '#waveform',
    waveColor: 'rgba(139,92,246,0.5)',
    progressColor: '#8b5cf6',
    cursorColor: '#ec4899',
    barWidth: 2,
    barRadius: 2,
    height: 88,
    normalize: true,
    backend: 'WebAudio',
  });

  wavesurfer.loadBlob(file);

  wavesurfer.on('ready', () => {
    totalTime.textContent = fmtTime(wavesurfer.getDuration());
  });

  wavesurfer.on('audioprocess', () => {
    currentTime.textContent = fmtTime(wavesurfer.getCurrentTime());
  });

  wavesurfer.on('finish', () => {
    playBtn.textContent = '▶';
  });

  // Scroll waveform card into view smoothly
  waveformCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ── Play / pause ────────────────────────────────────────────────────────── */
playBtn.addEventListener('click', () => {
  if (!wavesurfer) return;
  wavesurfer.playPause();
  playBtn.textContent = wavesurfer.isPlaying() ? '⏸' : '▶';
});

/* ── Classify ────────────────────────────────────────────────────────────── */
classifyBtn.addEventListener('click', async () => {
  if (!currentFile) return;
  if (!modelSelect.value) {
    alert('Please select a model before classifying.');
    return;
  }

  // Show loading state
  btnText.textContent = 'Classifying…';
  show(spinner);
  classifyBtn.classList.add('loading');
  classifyBtn.disabled = true;

  const formData = new FormData();
  formData.append('audio', currentFile);
  formData.append('model', modelSelect.value);

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) {
      alert(`Error: ${data.error}`);
      return;
    }

    renderResults(data);

  } catch (err) {
    alert('Network error — is the Flask server running?');
    console.error(err);
  } finally {
    btnText.textContent = 'Classify Genre';
    hide(spinner);
    classifyBtn.classList.remove('loading');
    classifyBtn.disabled = false;
  }
});

/* ── Render results ──────────────────────────────────────────────────────── */
function renderResults(data) {
  /* ── Show Results card ── */
  show(resultsCard);
  resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  /* ── Genre badge ── */
  genreNameEl.textContent = data.top_genre;
  genreBadge.classList.remove('revealed');
  void genreBadge.offsetWidth;              // force reflow to restart animation
  genreBadge.classList.add('revealed');

  /* ── Confidence ring ── */
  const pct = Math.round(data.confidence * 100);
  const circumference = 2 * Math.PI * 50;  // r = 50
  const offset = circumference * (1 - data.confidence);
  confPct.textContent = `${pct}%`;
  // Animate: trigger after a tick so CSS transition fires
  requestAnimationFrame(() => {
    ringFill.style.strokeDashoffset = offset;
  });

  /* ── Bar chart ── */
  const probs  = data.probabilities;
  const labels = Object.keys(probs).sort((a, b) => probs[b] - probs[a]);
  const values = labels.map(l => Math.round(probs[l] * 1000) / 10);  // as %

  const ctx = document.getElementById('genre-chart').getContext('2d');
  if (genreChart) genreChart.destroy();

  genreChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Probability (%)',
        data: values,
        backgroundColor: labels.map((_, i) =>
          i === 0
            ? 'rgba(139,92,246,0.85)'
            : 'rgba(139,92,246,0.25)'
        ),
        borderColor: labels.map((_, i) =>
          i === 0 ? '#8b5cf6' : 'rgba(139,92,246,0.4)'
        ),
        borderWidth: 1,
        borderRadius: 6,
      }],
    },
    options: {
      indexAxis: 'y',
      animation: { duration: 900, easing: 'easeOutQuart' },
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.parsed.x.toFixed(1)}%`,
          },
        },
      },
      scales: {
        x: {
          beginAtZero: true,
          max: 100,
          ticks: { color: '#64748b', callback: v => `${v}%` },
          grid: { color: 'rgba(255,255,255,0.05)' },
        },
        y: {
          ticks: { color: '#e2e8f0', font: { size: 12 } },
          grid: { display: false },
        },
      },
    },
  });
}

/* ── Reset ring to empty ─────────────────────────────────────────────────── */
function resetRing() {
  ringFill.style.strokeDashoffset = '314';
  confPct.textContent = '0%';
  genreNameEl.textContent = '—';
  genreBadge.classList.remove('revealed');
  genreBadge.style.opacity = '0';
  genreBadge.style.transform = 'scale(0.7)';
}

/* ── Reset button ────────────────────────────────────────────────────────── */
resetBtn.addEventListener('click', () => {
  currentFile = null;
  fileName.textContent = '';
  fileInput.value = '';
  hide(waveformCard);
  hide(resultsCard);
  if (wavesurfer) { wavesurfer.destroy(); wavesurfer = null; }
  if (genreChart)  { genreChart.destroy(); genreChart = null; }
  resetRing();
  uploadCard.scrollIntoView({ behavior: 'smooth' });
});
