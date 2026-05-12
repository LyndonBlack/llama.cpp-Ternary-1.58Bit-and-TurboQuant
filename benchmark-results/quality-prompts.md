# Quality Testing Prompts

Record of prompts used for qualitative assessment of entropy-guided KV cache performance.

## Walking Dog Animation (2026-05-12)

**Model:** Qwen3.6 35B A3B (Q5_K_M) — Entropy Path B, ratio 2.0, 256K context

**Result:** First attempt: visually rich scene with articulated figures, good anatomy, but animation loop failed to start (one-line bug). Second attempt as a clean re-prompt: ✅ animated correctly on first try — the model resolved the issue without being told about it.

**Comparison:** Meets or exceeds ChatGPT and Gemini output for this popular single-file HTML animation challenge.

**Hardware & performance:**
- CPU: AMD Ryzen 5 3800X (8c/16t)
- RAM: 32 GB DDR4 @ 3600 MHz
- GPU: RTX 3070 Ti 8 GB
- Model: Qwen3.6 35B A3B (Q5_K_M), CPU MoE offload
- KV config: Entropy Path B, ratio 2.0, q8 K + turbo3 V, 256K context
- Thinking/reasoning time: **11.4 seconds**
- Output time: **5 min 23 sec** (10,866 tokens)
- Generation speed: started at ~39 t/s, ended at ~33.6 t/s (averaged)

### Full prompt

```
Act as an expert creative coder and frontend developer. Write a complete, single-file HTML
document (containing HTML, inline CSS, and vanilla JavaScript) to create a high-fidelity, fully
procedural animation of a detailed person walking briskly down a road while walking a dog on
a leash.

Strict Technical & Design Constraints:
1. Single File: All CSS and JavaScript must be contained within the single HTML file.
   Absolutely no external assets, libraries (like Three.js or P5.js), or image files are allowed.
2. High-Fidelity Visuals: Use HTML5 Canvas API or complex SVG paths generated via JS. The
   person and the dog must have detailed, recognizable anatomical proportions (head, torso,
   articulated limbs, tail, etc.), avoiding simplistic stick figures.
3. Dynamic Animation Mechanics: Implement a brisk, energetic walking cycle for both characters.
   Use procedural animation techniques (e.g., trigonometric functions like Math.sin/Math.cos
   for limb articulation) so the movement is fluid and not sluggish.
4. The Leash: A visual leash must smoothly connect the person's hand to the dog's collar,
   updating its physics or curve dynamically as they move.
5. Immersive Environment: Include a beautifully designed, scrolling road or a parallax
   background (like trees, city skyline, or clouds) to create a strong illusion of fast
   forward movement.
6. Fullscreen & Looping: The canvas must be completely responsive, covering the entire screen
   (100vw, 100vh) with no scrollbars. The animation must run in an infinite, seamless loop
   using requestAnimationFrame.
7. Zero Text: There must be absolutely no text rendered on the screen.

Output strictly the raw HTML code inside a single code block. Ensure the code is self-contained,
highly optimized, and runs flawlessly upon opening in any modern browser.
```
