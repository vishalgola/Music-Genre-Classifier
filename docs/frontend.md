# `main.js` & `style.css` — The Animated Frontend

## Code Walkthrough

### `main.js` — Application State & Flow

The JS file manages a simple state machine with three screens: **upload → waveform → results**.

#### File handling

Two input paths lead to `handleFile(file)`:
1. **Drag & drop** — `dropZone.addEventListener('drop', ...)` extracts the file from `e.dataTransfer.files[0]`
2. **File picker** — standard `<input type="file">` fires a `change` event

`handleFile(file)` does three things:
- Stores the file reference in `currentFile` for later use in the fetch call
- Tears down any previous WaveSurfer instance to avoid memory leaks
- Creates a new WaveSurfer instance and calls `wavesurfer.loadBlob(file)` to render the waveform

#### WaveSurfer.js integration

```js
wavesurfer = WaveSurfer.create({
  container: '#waveform',
  waveColor: 'rgba(139,92,246,0.5)',
  progressColor: '#8b5cf6',
  ...
});
wavesurfer.loadBlob(file); // render from in-memory File object — no upload needed
```

WaveSurfer decodes the audio in the browser using the Web Audio API and renders peaks as SVG bars. The `audioprocess` event fires on every animation frame during playback, which we use to update the timestamp display.

#### `classifyBtn` → `/predict` call

```js
const formData = new FormData();
formData.append('audio', currentFile);
const res = await fetch('/predict', { method: 'POST', body: formData });
```

`FormData` is the browser's native way to send file uploads — Flask's `request.files` on the server side reads it automatically.

#### `renderResults(data)`

Receives the JSON response and:
1. Sets genre name text, adds `.revealed` CSS class → triggers `pop-scale` animation
2. Calculates `stroke-dashoffset` for the SVG ring and sets it → CSS transition animates it
3. Sorts genres by probability, creates Chart.js bar chart with `animation.duration = 900ms`

#### `resetBtn`

Destroys WaveSurfer and Chart.js instances (important to release Web Audio nodes and canvas memory), clears all state, and scrolls back to the upload card.

---

### `style.css` — Animations & Design System

#### CSS Variables (Design Tokens)

All colours, spacing, and timing values are defined as `--variable` in `:root`. Changing `--accent: #8b5cf6` to any other colour instantly re-themes the entire UI.

#### `fade-up` + delay classes

```css
.fade-up { opacity: 0; animation: fadeSlideUp 0.65s ease forwards; }
.delay-1 { animation-delay: 0.12s; }
.delay-2 { animation-delay: 0.25s; }
```

All three cards are initially invisible and stagger-animate in on page load. Combined with `forwards`, the opacity stays at 1 after the animation ends.

#### Upload zone pulse (`pulse-border`)

```css
@keyframes pulse-border {
  0%, 100% { box-shadow: 0 0 0 0 var(--accent-glow); }
  50%       { box-shadow: 0 0 0 12px transparent; }
}
```

This creates a "breathing" glow effect around the drop zone. It only runs on hover/drag-over via `animation: pulse-border 1.5s ease-in-out infinite`.

#### Genre badge `pop-scale`

```css
@keyframes pop-scale {
  0%   { transform: scale(0.7); opacity: 0; }
  70%  { transform: scale(1.08); }
  100% { transform: scale(1); opacity: 1; }
}
```

The slight overshoot to 1.08 before settling at 1.0 creates a springy "elastic" feel. We re-trigger this animation by removing the class, forcing a reflow, then re-adding it.

#### SVG Confidence Ring

The ring is an SVG `<circle>` with `stroke-dasharray="314"` (the circumference: `2π × r = 2π × 50 ≈ 314`). Filling percentage is controlled by `stroke-dashoffset`:

```
dashoffset = circumference × (1 - confidence)

confidence = 0.85 → offset = 314 × 0.15 ≈ 47
confidence = 0.0  → offset = 314  (ring completely invisible)
confidence = 1.0  → offset = 0    (ring fully drawn)
```

The CSS `transition: stroke-dashoffset 1.2s` animates the ring filling in smoothly from 0 to the final value.

---

## Theory Behind It

### What is the Web Audio API?

The Web Audio API is a browser-native system for processing and synthesising audio. WaveSurfer.js uses it as its "backend" to:
1. **Decode** the audio file (MP3/WAV/OGG → raw PCM samples)
2. **Analyse** the waveform amplitude at thousands of points
3. **Render** those peaks as SVG bars in the `#waveform` div

This all happens locally in the browser — the audio file is never uploaded until you click "Classify".

### CSS Animations vs. JavaScript animations

CSS animations (`@keyframes`) run on the **compositor thread**, separate from the main JavaScript thread. This means they stay smooth even if JavaScript is busy — for example, while the `fetch` to `/predict` is in flight. JavaScript animations (e.g., using `setInterval`) would stutter under load.

Rule of thumb: use CSS for visual animations, JavaScript for logic-driven state changes.

### Why Chart.js for the bar chart?

Chart.js is a lightweight (~200 KB) canvas-based charting library with built-in animation support. Setting `animation.duration = 900` makes bars grow from zero to their final height over 900ms using the specified easing function. This is much more engaging than a static table of percentages.

### The `void element.offsetWidth` reflow trick

To replay a CSS animation on an element that has already animated, browsers cache the previous state. Reading `offsetWidth` (or any layout property) forces an immediate reflow — the browser recalculates the layout — effectively resetting the animation state so it starts fresh.
