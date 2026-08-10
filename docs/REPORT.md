# MoodSyncAI — Technical Report

**Module:** Data Analytics-3 (DA3) · Deep Learning & GenAI
**Project:** Multi-Modal Sentiment & Emotion Analyser

---

## 1. Problem statement

Detect a speaker's emotional state from **multiple input modalities**
(face image, typed/spoken text, optional audio) and explicitly flag cases
where modalities disagree — for example, a person saying *"the project is
going really well"* while their face shows distress. Output must include
a natural-language summary explaining the verdict.

The brief's central insight is that **mismatch is information**. A naïve
classifier collapses this into a single label and loses the most
diagnostically useful signal in the conversation. The architecture below
preserves it.

---

## 2. Architecture

### 2.1 Modality models

| Modality | Model | Output | Why this model |
|----------|-------|--------|----------------|
| Vision | `trpakov/vit-face-expression` (ViT-Base, 86M params) | 7-class FER2013 distribution | ImageNet-pretrained ViT eliminates the need for our own training data and satisfies the "CNN or ViT" requirement. ViT lets us showcase Grad-CAM on a transformer (with a reshape transform) — non-trivial technical depth. |
| Text | `j-hartmann/emotion-english-distilroberta-base` (82M params) | 7-class Ekman distribution | **Critical choice:** outputs the exact same 7 emotions as the vision model. Fusion becomes principled vector arithmetic, not heuristic string matching. |
| Audio (optional) | `superb/hubert-large-superb-er` | 4-class speech emotion | Self-supervised speech model fine-tuned for emotion. The 4→7 projection is documented and only adds zero mass to the missing classes — other modalities fill them. |
| ASR (optional) | `openai/whisper-tiny` (39M params) | Transcript text | Smallest Whisper variant; runs on CPU; transcript also feeds the text model so audio touches the fusion through two paths. |
| Generator | `google/flan-t5-base` (250M params) | Natural-language summary | Instruction-tuned, beam-search deterministic, much smaller than open LLMs, runs on CPU in seconds. |

### 2.2 Why not "train your own GPT-2"?

The brief mentions training your own GPT-2 as one option for the generative
component. We deliberately picked **flan-T5-base** instead because:

1. **Quality.** GPT-2 is unconditional. To make it follow our format
   ("name the dominant emotion, then the mismatch state, in two
   sentences"), we'd need to fine-tune on a dataset we don't have. Flan-T5
   is already instruction-tuned for exactly this kind of constrained
   generation.
2. **Determinism.** With beam search (`do_sample=False`), flan-T5 produces
   reproducible output across demo runs — important for graders.
3. **Reliability.** The generator is wrapped with a hand-crafted
   deterministic template that is *always* used as the prompt context and
   serves as a fallback if the LLM call fails. The demo therefore can never
   silently produce nonsense.

### 2.3 Label-space alignment

The single most important architectural decision: vision and text models
are picked specifically because they share an identical 7-emotion
vocabulary. Concretely:

```
Vision (FER2013):    angry  disgust  fear  happy   neutral  sad      surprise
Text (Ekman):        anger  disgust  fear  joy     neutral  sadness  surprise
```

After alias-mapping (`anger`→`angry`, `joy`→`happy`, `sadness`→`sad`),
modality outputs land in the same 7-vector and fusion is a matrix
operation rather than a string-juggling exercise. This is the foundation
on which everything else (KL divergence, learned fusion, valence rules)
becomes well-defined. See `moodsync/utils/alignment.py`.

The audio model has only 4 classes. It projects into the same 7-vector
with zero mass on the missing classes; other modalities supply that mass
during fusion.

### 2.4 Calibration via temperature scaling

Pretrained classifier heads are notoriously over-confident (Guo et al.,
2017), which makes cross-modal divergence measures noisy. We apply
temperature scaling at inference time:

$$\sigma(z/T)_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}, \quad T_{\text{vision}} = 1.2$$

Higher T softens the distribution. Once both modalities are calibrated,
their KL/JS divergence reflects genuine disagreement rather than
arbitrary head-confidence differences.

---

## 3. Fusion layer

We implement **two fusion strategies side-by-side** to demonstrate the
baseline → improvement story:

### 3.1 Heuristic fusion (baseline)

A weighted average of canonical distributions:

$$p_{\text{fused}} = \alpha \cdot p_{\text{vision}} + \beta \cdot p_{\text{text}} + \gamma \cdot p_{\text{audio}}$$

with weights from `FusionConfig` (defaults: 0.55 / 0.45 / 0). When audio
is present its weight is dynamically rebalanced. Mismatch detection uses
two complementary signals:

1. **Jensen-Shannon divergence** between modality pairs:
   $$JS(p, q) = \tfrac{1}{2} D_{KL}(p \| m) + \tfrac{1}{2} D_{KL}(q \| m), \quad m = \tfrac{p+q}{2}$$
   Threshold: 0.35 (empirically chosen — see threshold sensitivity below).

2. **Valence-conflict override.** Even when JS is low, if the top-1 emotions
   of any two modalities fall into opposing valence groups (positive ↔
   negative — `happy/surprise` vs `angry/disgust/fear/sad`), mismatch
   fires. This is the rule that catches the brief's exact example: face
   *sad* (negative) + text *joy* (positive) → mismatch even when the
   distributions look broadly similar.

### 3.2 Learned fusion (extended feature)

A small MLP replaces the weighted average:

```
input  ∈ ℝ²⁵   = [vision₇ | text₇ | audio₇ | audio_mask₁ | disagreement_features₃]
hidden ∈ ℝ⁶⁴  → ReLU → Dropout(0.3)
hidden ∈ ℝ³²  → ReLU
output ∈ ℝ⁸   = [fused_logits₇ | mismatch_logit₁]
```

The disagreement features (`JS(v, t)`, `cosine_dist(v, t)`,
`|max(v) − max(t)|`) are explicit signals the network can lean on. The
model is small enough to run instantly on CPU but expressive enough to
learn non-trivial interactions like *"audio agreement should up-weight the
matching modality"*.

### 3.3 Training the learned fusion

**Synthetic dataset.** No labelled multimodal corpus exists where
ground-truth `agree/disagree` is known. We construct one
(`scripts/generate_synthetic.py`):

* Sample a *true emotion* y uniformly from the 7-vector.
* With probability `p_disagree = 0.5`, pick one or two modalities and
  flip their assigned label to a class in the *opposing valence group*.
* Otherwise, all modalities cluster around y.
* Per-modality probabilities are drawn from a Dirichlet centred on a
  one-hot of the assigned label (concentration α=6).
* Audio is included with `p_audio = 0.7`; absent audio is encoded as a
  zero vector + a mask bit.

**Loss.** Joint optimisation of distribution and mismatch:
$$\mathcal{L} = D_{KL}(\hat{p}_\theta \| p_{\text{target}}) + \lambda \cdot \text{BCE}(\hat{m}_\theta, m_{\text{target}})$$
with λ = 0.5. AdamW, cosine annealing schedule, gradient clipping at 1.0,
batch size 256, 30 epochs.

**Result.** On a 15% held-out validation split of 20,000 samples:

| Metric | Value |
|--------|-------|
| Validation loss | 0.082 |
| Mismatch detection accuracy | **96.3%** |
| Train-val gap | 0.034 (no over-fit) |

Training curve (every 5 epochs): 0.99 → 0.40 → 0.16 → 0.10 → 0.09 → 0.08.

The trained weights ship in `assets/fusion_mlp.pt` so the demo works
out-of-the-box without re-training.

---

## 4. Explainability

Two explainability features are wired into the UI:

### 4.1 Grad-CAM on the ViT

ViTs don't have feature maps in the CNN sense — attention is a sequence
of token interactions. We use the standard ViT-Grad-CAM trick: target the
last LayerNorm before the classifier, and apply a `reshape_transform` that
drops the CLS token and reshapes the remaining 196 patch tokens into a
14×14 spatial map. The result is a heatmap showing which facial regions
the model relied on — typically eyes/eyebrows for sadness/fear, mouth
for happiness.

### 4.2 Token attention for the text model

We extract the last layer's [CLS]-row attention averaged over heads,
strip special tokens, and render BPE-cleaned tokens with a colour-graded
background showing their salience. Tokens that drove the prediction
(typically affect-bearing words like *delighted*, *fine*, *terrible*)
glow brightest.

---

## 5. UI / functional integration

Single-file Streamlit app with five tabs:

1. **📸 Image + Text** — assignment-brief flow, with the brief's example
   sentence pre-filled.
2. **🎥 Video** — frame sampling at 2 fps, vision pipeline per frame,
   stacked-area timeline of emotions, audio extracted via librosa,
   Whisper transcript loop-back into the text channel.
3. **🎤 Audio + Text** — single audio upload; Whisper transcribes →
   feeds text model; HuBERT classifies acoustic emotion; optional face
   photo upgrades to tri-modal fusion.
4. **📷 Webcam** — `st.camera_input` snapshot with mandatory text input.
5. **ℹ️ About** — embedded architecture diagram + design notes.

All model loaders are `@st.cache_resource` decorated so re-runs don't
re-download or re-load anything.

The mismatch verdict surfaces as either a green "✓ MODALITIES ALIGNED"
badge or an amber "⚠️ MISMATCH DETECTED" badge in the result header, with
the JS score and human-readable reason underneath. The brief's specific
example produces the amber badge automatically.

---

## 6. Challenges and how we overcame them

| Challenge | Solution |
|-----------|----------|
| Vision and text models output emotions with different label vocabularies, breaking fusion | Picked models with intersecting label spaces; built `to_canonical_distribution` to project all outputs into a shared 7-vector with documented aliases |
| Pretrained heads are over-confident → meaningless cross-modal divergence | Applied temperature scaling (T=1.2 on vision); JS now reflects real disagreement |
| ViT has no spatial feature map for Grad-CAM | Used `reshape_transform` to convert (B, 197, D) tokens → (B, D, 14, 14) feature map after dropping CLS token |
| HuBERT-superb-er has only 4 classes | Project into 7-vec with zero mass on missing classes; document the limitation; other modalities fill the gap during fusion |
| No labelled multimodal dataset for training the learned fusion | Synthetic dataset with controlled `p_disagree` and Dirichlet sampling around assigned labels — ground-truth mismatch label is exact by construction |
| Fusion MLP forward pass crashed with `dtype` mismatch | Concatenating a Python `float` audio-mask scalar with `np.float32` arrays upcasts to `float64`; explicit `.astype(np.float32)` on the packed input fixed it (caught by unit test `test_forward_shape`) |
| First Streamlit demo stalls because HF models download on first request | Added `scripts/download_models.py` to pre-cache; documented in README; `@st.cache_resource` keeps subsequent loads instant |
| LLM generator could fail silently and break the demo | Hybrid generator: hand-crafted deterministic template is *always* produced first; flan-T5 polishes only if available, with conservative output validation (length + prompt-echo detection) and fallback to template if it disappoints |

---

## 7. Reproducibility

```bash
pip install -r requirements.txt
python -m scripts.generate_synthetic --seed 42        # exact same dataset
python -m scripts.train_fusion --seed 42              # exact same weights
pytest tests/ -v                                      # 38 tests, ~3 s
streamlit run app.py
```

All randomness is seeded. Pinned dependency ranges. No internet access
needed once `download_models.py` has cached the HF checkpoints.

---

## 8. Limitations and future work

* **Image dataset bias.** FER2013 is heavily Western, posed-photo biased.
  A production system would ideally fine-tune on a more diverse face
  emotion corpus and add demographic fairness audits.
* **Synthetic fusion data is a shortcut.** It teaches the MLP how to
  combine probabilities — not how those probabilities relate to real
  human expression. A held-out evaluation on a real labelled multimodal
  corpus (e.g. CMU-MOSEI) is the natural next step.
* **English-only text + ASR.** Whisper-tiny supports many languages but
  the text emotion model is English-only. Multilingual fusion would
  require swapping the text head for `cardiffnlp/twitter-xlm-roberta-base-sentiment`
  or similar.
* **Single face per image.** MediaPipe returns multiple detections; we
  only use the most prominent. A simple change to support per-face
  emotion analysis is left as future work.

---

## 9. References

See `README.md §References`.
