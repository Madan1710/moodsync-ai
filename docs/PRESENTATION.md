# MoodSyncAI — 5-Minute Presentation Outline

This script is structured for the exam format the brief specifies: **5 min
demo + 5 min presentation + 5 min Q&A**. The instructor explicitly said
*"I like more time to see the architecture diagram with data flow and how
you overcome the challenges"* — so the architecture diagram gets the
biggest single time slice.

---

## 🎬 Part 1 — Live demo (5 min)

Open the Streamlit app. Hit each tab in this order — it's the most
compelling narrative:

### 1. Image + Text tab (90 s)
* The brief's exact sentence is pre-filled.
* Upload a face photo that looks sad / worried.
* Hit **Analyse**. Talk through:
  * Bar charts → vision says *sad*, text says *joy*.
  * Top header → amber "⚠️ MISMATCH DETECTED" badge.
  * Generative summary mirrors the brief's example output.
  * Grad-CAM overlay → eyes/brows lit up.
  * Token attention → "really" and "well" are highlighted.

### 2. Video tab (60 s)
* Upload a short clip with shifting expressions.
* Show the per-frame timeline (stacked area chart of all 7 emotions).
* Whisper transcript loops back into the text channel automatically.
* Tri-modal fusion verdict.

### 3. Audio tab (45 s)
* Upload a voice clip.
* Whisper transcribes; HuBERT classifies acoustic emotion.
* Show the audio's distinct contribution to the final fusion vector.

### 4. Webcam tab (30 s)
* Take a quick snap.
* Show the in-browser camera capture working.

### 5. Sidebar toggle (15 s)
* Switch from heuristic to **learned fusion**.
* Re-run. Same result, different reasoning ("Learned fusion p(mismatch)=0.93").

> **Note for the demo:** keep models pre-cached. Run
> `python -m scripts.download_models` *before* the exam.

---

## 📐 Part 2 — Presentation (5 min, architecture-focused)

### Slide 1 — Title (15 s)
* **MoodSyncAI · Multi-Modal Sentiment & Emotion Analyser**
* Your name, DA3 final project.

### Slide 2 — Problem (30 s)
* Single-modality classifiers collapse the **most diagnostic signal —
  modality disagreement** — into a single label.
* The brief's example: face sad + text "the project is going really
  well." A single classifier picks one and discards the other; we keep
  both and explicitly flag the conflict.

### Slide 3 — Architecture diagram (90 s) **[the big one]**
* Show `docs/architecture_diagram.svg` full-screen.
* Walk left to right:
  * Three parallel input branches: image / text / audio.
  * Pre-processing: MediaPipe face crop, BPE tokenizer, 16 kHz resample.
  * Three model backbones (ViT, DistilRoBERTa, HuBERT + Whisper).
  * **Canonical 7-vector alignment layer** — emphasise this is the
    keystone decision.
  * Fusion layer → mismatch flag → flan-T5 generator.
  * Dashed line: Whisper transcript loops back into the text branch.

### Slide 4 — Two design choices that earned marks (60 s)
1. **Label-space alignment.** Vision = FER2013, Text = Ekman; identical
   7 emotions after alias mapping. Fusion is principled.
2. **Two-signal mismatch detection.** JS divergence (continuous) +
   valence-conflict override (discrete). Either alone misses cases.

### Slide 5 — Challenges and solutions (60 s)
| Challenge | What I did |
|-----------|------------|
| ViT has no feature map for Grad-CAM | Reshape transform: 197 tokens → 14×14 spatial after dropping CLS |
| No labelled multimodal data for the learned fusion | Synthetic Dirichlet-sampled dataset with controlled disagreement; 96.3% mismatch acc on val |
| Pretrained heads over-confident | Temperature scaling so KL divergence is meaningful |
| HuBERT only outputs 4 classes | Project to canonical 7-vec; let other modalities fill the gap |
| LLM could silently fail in the demo | Hybrid generator: deterministic template + flan-T5 polish |

### Slide 6 — Results & extended features (45 s)
* All six extended features implemented (webcam, video, audio, attention
  viz, learned fusion, deployment).
* Trained fusion: 96.3% mismatch detection accuracy.
* 38 unit + integration tests · runs in ~3 s.

### Slide 7 — Closing (20 s)
* Repository link, demo URL.
* Thanks; ready for Q&A.

---

## 💬 Part 3 — Q&A prep (5 min)

Likely questions and crisp answers:

| Q | A |
|---|---|
| **Why ViT instead of CNN?** | Brief allows either. ViT-Base (ImageNet-pretrained) outperforms a from-scratch CNN on FER2013 with no training data, and lets me showcase Grad-CAM on a transformer — non-trivial. |
| **Why DistilRoBERTa?** | It's specifically the model whose 7 Ekman labels match the FER2013 vision labels 1-to-1. That alignment is the foundation for principled fusion. |
| **Why not a single end-to-end model?** | (a) No labelled tri-modal dataset of the size needed. (b) Modular pipeline lets us swap any backbone independently. (c) Mismatch detection becomes interpretable — we can point at *which two modalities disagreed and why*. |
| **What is Jensen-Shannon divergence and why use it over KL?** | JS = symmetric average of two KL divergences against the midpoint. Bounded in [0, ln 2], symmetric, well-defined when distributions don't share support. KL is asymmetric and blows up on zero entries. |
| **How was the fusion MLP trained?** | Synthetic dataset of 20k samples sampled from Dirichlet distributions around assigned emotion labels with controlled disagreement rate (50%). Joint loss = KL on fused distribution + 0.5 × BCE on mismatch logit. AdamW, cosine schedule, 30 epochs. |
| **Could the system over-fit to your synthetic data?** | Possibly — that's the main limitation. The 96.3% accuracy is on synthetic-distribution-held-out data. Real-world generalisation needs evaluation on a corpus like CMU-MOSEI. |
| **Why flan-T5 rather than GPT-2 (the lecture choice)?** | Flan-T5 is instruction-tuned, so it follows our two-sentence summary format without fine-tuning. GPT-2 would need supervised fine-tuning on a dataset I don't have. With deterministic beam search, flan-T5 also gives reproducible outputs across demo runs. |
| **What if Grad-CAM disagrees with the prediction?** | That's actually informative — it means the model is keying off a non-face region (background bias) and the prediction is suspect. The UI shows both so the user can spot it. |
| **Production deployment concerns?** | Privacy (face data is biometric), bias (FER2013 is Western/posed), real-time latency (need quantisation + ONNX), and consent flows. Documented in §8 of the report. |

---

## 🎤 Stage tips

* **Keep to time.** Brief explicitly penalises overrunning. Practice with
  a stopwatch.
* **The architecture-diagram slide gets the most time.** That's what the
  examiner asked for.
* **Don't read slides.** Talk *over* the diagram with a pointer.
* **Leave 30 seconds buffer at the end** for "any final questions?" — it
  feels professional.
