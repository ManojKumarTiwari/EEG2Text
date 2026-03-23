# EEG-to-Language Decoding via Multimodal Foundation Models
## MTech Project Report

---

**Indian Institute of Technology Jodhpur**
Department of Computer Science and Engineering

---

| | |
|---|---|
| **Student** | [Your Name] |
| **Roll No.** | [Your Roll Number] |
| **Supervisor** | [Supervisor Name] |
| **Programme** | M.Tech. (Computer Science and Engineering) |
| **Academic Year** | 2024–2025 |
| **Submission Date** | March 2026 |

---

*Submitted in partial fulfillment of the requirements for the degree of*
**Master of Technology in Computer Science and Engineering**

---

## Declaration

I hereby declare that the work presented in this report is my own and has been carried out under the supervision of [Supervisor Name] at IIT Jodhpur. The work has not been submitted elsewhere for any other degree or qualification. All sources of information have been duly acknowledged.

**Signature:** ______________________

**Date:** March 2026

---

## Certificate

This is to certify that the report titled **"EEG-to-Language Decoding via Multimodal Foundation Models"** submitted by [Your Name] (Roll No.: [Roll No.]) is a record of bonafide work carried out under my supervision and guidance at the Indian Institute of Technology Jodhpur, in partial fulfillment of the requirements for the degree of Master of Technology in Computer Science and Engineering.

**Supervisor Signature:** ______________________

**Date:** March 2026

---

## Abstract

Brain-Computer Interfaces (BCIs) based on electroencephalography (EEG) have traditionally employed task-specific discriminative classifiers that map neural signals to a fixed set of predefined categories. This project proposes a novel paradigm: **EEG-to-Language decoding**, where raw motor imagery EEG signals are translated into free-form neuroscientific descriptions using a multimodal large language model (LLM) architecture.

The system, inspired by the LLaVA (Large Language and Vision Assistant) framework, combines a **frozen CSBrain EEG foundation model** — a cross-scale spatiotemporal transformer pretrained on large-scale unlabeled EEG data — with **TinyLlama-1.1B-Chat**, a compact 1.1-billion-parameter language model. An **EEGTokenReducer** compresses the spatiotemporal token sequence from 88 to 12 tokens via brain-region pooling, and a trainable **2-layer MLP projection** bridges the 200-dimensional EEG embedding space to TinyLlama's 2048-dimensional input space.

Training employs a two-phase strategy on the BCI Competition IV 2a dataset (4-class motor imagery, 9 subjects): a 5-epoch projection warmup followed by 15-epoch joint fine-tuning with Low-Rank Adaptation (LoRA). The model achieves **31.34% test accuracy** on keyword-extracted class prediction from generated text, exceeding the 25% chance baseline by 6.34 percentage points absolute, with peak validation accuracy of **36.81%**. The generated outputs produce contextually appropriate neuroscientific language including event-related desynchronization (ERD) descriptions, correct lateralization of sensorimotor cortex activity, and mu/beta band references, demonstrating meaningful EEG-to-language alignment with only ~5.7M trainable parameters on consumer-grade 8GB VRAM hardware.

**Keywords:** EEG decoding, Brain-Computer Interface, Large Language Models, Motor Imagery, Foundation Models, LoRA, TinyLlama, CSBrain, Multimodal Learning

---

## Table of Contents

1. Introduction
2. Related Work
3. Problem Statement and Objectives
4. Dataset
5. System Architecture
   - 5.1 CSBrain Foundation Model (Encoder)
   - 5.2 EEGTokenReducer
   - 5.3 EEGProjection MLP
   - 5.4 TinyLlama Decoder with LoRA
   - 5.5 End-to-End Forward Pass
6. Data Pipeline
   - 6.1 Preprocessing
   - 6.2 Label-to-Text Mapping
   - 6.3 LMDB Storage
7. Training Strategy
   - 7.1 Phase 1: Projection Warmup
   - 7.2 Phase 2: Joint Fine-tuning
   - 7.3 Hyperparameters
8. Evaluation
   - 8.1 Keyword Extraction Metric
   - 8.2 Results
   - 8.3 Sample Generated Outputs
9. Discussion
10. Conclusion and Future Work
11. References

---

## 1. Introduction

The electroencephalogram (EEG) captures electrical activity from the brain's cortex at millisecond temporal resolution and has been the primary modality for non-invasive Brain-Computer Interfaces (BCIs). Traditional BCI decoding pipelines learn discriminative mappings from engineered spectral-spatial features to discrete class labels — for example, classifying left-hand versus right-hand motor imagery from mu/beta band power spectral densities over contralateral electrodes.

While these approaches have achieved practical classification accuracy on constrained task sets, they suffer from several fundamental limitations:

1. **Task specificity**: Models trained for one BCI task (e.g., motor imagery) cannot transfer to another (e.g., emotion recognition) without retraining.
2. **Opaque predictions**: A classifier outputs a label index, providing no interpretable description of the neural dynamics it has detected.
3. **Label bottleneck**: Rich, graded neurological information is collapsed into a single categorical output.
4. **Fixed vocabulary**: The model cannot express uncertainty, mixed patterns, or novel neural states outside its training categories.

The recent success of Large Language Models (LLMs) in natural language generation and multimodal vision-language models such as LLaVA, BLIP-2, and InstructBLIP suggests a compelling alternative: **generative EEG decoding**. Instead of classifying EEG into a fixed label set, a generative model can produce open-ended neuroscientific descriptions, express confidence through language, and potentially generalize to tasks outside its training distribution.

This project implements and evaluates an end-to-end **EEG-to-Language** system that:
- Leverages a pretrained EEG foundation model (CSBrain) as a frozen, general-purpose EEG encoder
- Projects EEG representations into the embedding space of a compact LLM (TinyLlama-1.1B)
- Fine-tunes only a small projection MLP and LoRA adapters (~5.7M parameters total)
- Generates natural language descriptions of motor imagery EEG trials
- Evaluates generation quality via keyword-based class extraction

The approach is designed to run on a single consumer GPU with 8GB VRAM, making it accessible for academic research settings.

---

## 2. Related Work

### 2.1 EEG Foundation Models

Traditional EEG decoding relied on hand-crafted features such as Common Spatial Patterns (CSP), Filter Bank CSP (FBCSP), and bandpower features. Deep learning approaches — CNNs (EEGNet [Lawhern et al., 2018]), RNNs, and transformer-based models — improved accuracy but remained task-specific.

The concept of large-scale EEG pretraining emerged with models like:
- **BENDR** (Kostas et al., 2020): Contrastive pretraining on large EEG corpora
- **LaBraM** (Jiang et al., 2024): Large Brain Model with masked EEG modeling
- **CBraMod** (Wang et al., 2024): Criss-Cross Transformer for EEG pretraining
- **CSBrain** (2025, NeurIPS Spotlight): Cross-scale spatiotemporal foundation model with inter-window temporal and inter-region spatial attention

CSBrain, used as the backbone in this work, achieves state-of-the-art results across 16 downstream EEG tasks through its novel multi-scale temporal convolution (kernel sizes 1, 3, 5) and brain-region-aware spatial attention mechanism.

### 2.2 Vision-Language Models as Inspiration

The architecture of this work is directly inspired by **LLaVA** (Liu et al., 2023), which demonstrated that a simple linear projection can effectively bridge a frozen CLIP visual encoder with a Vicuna LLM for visual instruction following. Subsequent work — **BLIP-2** (Li et al., 2023) using a Q-Former, **InstructBLIP**, and **MiniGPT-4** — refined this paradigm.

The key insight transferable to EEG: if a sufficiently powerful pretrained modality encoder exists (here, CSBrain ≈ CLIP for EEG), a compact trainable interface can align it to an LLM without fine-tuning billions of parameters.

### 2.3 EEG-to-Text Decoding

Early EEG-to-text work focused on imagined speech decoding (Wang et al., 2022; Défossez et al., 2023). **EEG-to-Language for BCI** is largely unexplored. The closest prior work includes:
- **DeWave** (Duan et al., 2023): EEG-to-text translation using discrete codex representations
- **BELT** (Chen et al., 2023): EEG-language pretraining with contrastive alignment
- **EEGFormer** (Song et al., 2023): Transformer-based EEG-language alignment

This project contributes a practical, resource-efficient pipeline combining a domain-specific EEG foundation model with modern LLM fine-tuning techniques (4-bit quantization + LoRA).

### 2.4 Parameter-Efficient Fine-tuning

**LoRA** (Hu et al., 2021) injects low-rank decomposition matrices into transformer attention layers, enabling LLM adaptation with a fraction of the parameters (here, 1.1M vs. 1.1B). Combined with **4-bit NF4 quantization** via BitsAndBytes (Dettmers et al., 2022), the full TinyLlama model fits in ~700MB GPU memory.

---

## 3. Problem Statement and Objectives

### 3.1 Problem Statement

Given a 4-second motor imagery EEG recording from the BCI Competition IV 2a dataset (22 channels, 4 classes: Left Hand, Right Hand, Both Feet, Tongue), generate a natural language description of the neural activity that:
1. Correctly identifies the motor imagery class
2. Contains accurate neurophysiological terminology (ERD/ERS, lateralization, band-specific activity)
3. Is coherent and readable as expert neuroscience text

### 3.2 Objectives

1. **Architecture Design**: Implement a modality-bridging architecture connecting CSBrain (EEG encoder) to TinyLlama (text decoder) via a trainable projection module.
2. **Efficient Training**: Train the system on consumer hardware (8GB VRAM) using quantization and LoRA.
3. **Label-Supervised Generation**: Use curated neuroscientific text descriptions as training targets for each motor imagery class.
4. **Evaluation**: Develop a keyword-based accuracy metric that extracts predicted labels from generated free-form text.
5. **Analysis**: Study the two-phase training dynamics and the quality of generated outputs.

---

## 4. Dataset

### 4.1 BCI Competition IV Dataset 2a (BCIC-IV-2a)

The primary dataset is the widely-used BCI Competition IV Dataset 2a (Brunner et al., 2008), which has become the standard benchmark for motor imagery EEG research.

| Property | Value |
|---|---|
| Subjects | 9 healthy subjects |
| Channels | 22 EEG channels (+ 3 EOG, discarded) |
| Classes | Left Hand (0), Right Hand (1), Both Feet (2), Tongue (3) |
| Trials per subject | 288 per session × 2 sessions = 576 |
| Sampling rate | 250 Hz (original) |
| Motor imagery window | 2s–6s post-cue (4 seconds) |
| Format | MATLAB .mat files (BNCI Horizon 2020) |

**Channel Layout relevant to Motor Imagery:**

| Brain Region | Channels |
|---|---|
| Frontal (Region 0) | Fz |
| Parietal (Region 1) | P1, Pz, P2, POz |
| Central / FC (Region 4) | FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4 |

### 4.2 Train/Validation/Test Splits

| Split | Subjects | Samples |
|---|---|---|
| Train | A01–A05 | 2,784 |
| Validation | A06–A07 | 1,152 |
| Test | A08–A09 | 1,152 |

The subject-independent split (training on unseen subjects at test time) makes the task more challenging and evaluates cross-subject generalization.

---

## 5. System Architecture

The proposed architecture consists of four modules arranged in a sequential pipeline:

```
EEG Input (batch, 22, 4, 200)
         ↓
  [1] CSBrain Encoder (frozen)
         ↓ (batch, 22, 4, 200)
  [2] EEGTokenReducer
         ↓ (batch, 12, 200)
  [3] EEGProjection MLP
         ↓ (batch, 12, 2048)
  [4] Concatenate with text embeddings
         prompt_embeds  (batch, 101, 2048)
         eeg_embeds     (batch,  12, 2048)
         target_embeds  (batch,  61, 2048)
         ↓
  [5] TinyLlama-1.1B + LoRA
         ↓
  Generated neuroscience description
```

### 5.1 CSBrain Foundation Model (Encoder)

CSBrain (NeurIPS 2025 Spotlight) is a cross-scale spatiotemporal EEG foundation model pretrained via masked autoencoding on large unlabeled EEG corpora. Its architecture comprises:

| Sub-module | Description |
|---|---|
| `PatchEmbedding` | Conv2D patch projection (200-dim) + learnable positional embeddings + spectral FFT features |
| `TemEmbedEEGLayer` | Cross-scale temporal embedding with parallel convolutions at kernel sizes (1,), (3,), (5,) |
| `BrainEmbedEEGLayer` | Region-aware spatial embedding using a 5-region brain map |
| `CSBrain_TransformerEncoder` | 12-layer transformer with inter-window temporal attention (window_size=5) and inter-region spatial attention |
| `proj_out` | **Replaced with `nn.Identity()`** — raw 200-dim features passed through |

**Input/Output**:
- Input: `(batch, 22, 4, 200)` — 22 channels, 4 temporal patches, 200 time points
- Output: `(batch, 22, 4, 200)` — channel-patch feature vectors in d_model=200 space

All CSBrain parameters are **frozen** throughout training. This prevents catastrophic forgetting of its pretrained EEG representations and significantly reduces VRAM requirements by eliminating encoder gradients.

The brain region mapping for BCIC-IV-2a (22 channels) assigns:
- Region 0 (Frontal): Fz [index 0]
- Region 1 (Parietal): P1, Pz, P2, POz [indices 18–21]
- Region 4 (Central): FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4 [indices 1–17]

Channels are re-sorted by region before encoding: `sorted_indices = [0, 18, 19, 20, 21, 1..17]`.

### 5.2 EEGTokenReducer

The CSBrain encoder outputs 22×4 = 88 tokens, which is prohibitively long for an LLM's context window under memory constraints. The `EEGTokenReducer` compresses this sequence by pooling within brain regions:

**Algorithm**:
1. For each of the 3 occupied brain regions (Frontal: 1ch, Parietal: 4ch, Central: 17ch):
   - Average all channel embeddings within the region → 1 token per temporal patch
2. With 3 regions × 4 temporal patches = **12 EEG tokens total**

**Input**: `(batch, 22, 4, 200)` — from CSBrain
**Output**: `(batch, 12, 200)` — 12 region-temporal tokens

This compression is essential for fitting within TinyLlama's effective context window on 8GB VRAM while preserving the spatiotemporal structure of the EEG.

### 5.3 EEGProjection MLP

A 2-layer multilayer perceptron maps from EEG feature space (d=200) to TinyLlama's embedding dimension (d=2048):

```
Linear(200 → 2048) → GELU → Dropout(0.1) → Linear(2048 → 2048)
```

**Input**: `(batch, 12, 200)`
**Output**: `(batch, 12, 2048)`

Parameters: 200×2048 + 2048 + 2048×2048 + 2048 = **4,608,000** (trainable in both phases).

This module is the critical alignment bridge between the EEG encoder and the LLM. It is analogous to the linear projection in LLaVA or the Q-Former in BLIP-2.

### 5.4 TinyLlama-1.1B-Chat Decoder with LoRA

**Base model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

TinyLlama is a compact open-source language model with 1.1 billion parameters trained on 3 trillion tokens. It uses the Llama-2 architecture (grouped-query attention, RoPE, SwiGLU) with a hidden dimension of 2048, making it compatible with the EEGProjection output.

**Memory optimization**:
- **4-bit NF4 quantization** via BitsAndBytes `load_in_4bit=True` + double quantization
- Reduces model footprint from ~4.4GB to ~700MB VRAM
- Compute dtype: float16

**LoRA Configuration**:

| Parameter | Value |
|---|---|
| Rank (r) | 8 |
| Alpha (α) | 16 |
| Dropout | 0.05 |
| Target modules | `q_proj`, `v_proj` (all attention layers) |
| Trainable parameters | **1,126,400** (~0.10% of 1.1B) |

LoRA decomposes weight updates as ΔW = BA where B ∈ ℝ^(d×r) and A ∈ ℝ^(r×k), with rank r=8, enabling adaptation at a fraction of the full fine-tuning cost.

**Chat template** (TinyLlama Zephyr/ChatML format):
```
<|system|>
You are an expert EEG analyst specializing in brain-computer interfaces and motor
imagery decoding. Given EEG signal embeddings, provide a scientific description of
the brain activity.</s>
<|user|>
[EEG_TOKENS]
Analyze this EEG recording and describe the motor imagery task the subject is
performing, including the relevant brain regions and frequency bands.</s>
<|assistant|>
[Target neuroscience description]
```

### 5.5 End-to-End Forward Pass

During training, the forward pass constructs a single sequence:

```
[prompt_embeds (101 tokens)] ++ [eeg_embeds (12 tokens)] ++ [target_embeds (61 tokens)]
       ↑ text context                ↑ EEG signal                ↑ supervised output
```

**Loss masking**: Only the `target_embeds` region contributes to cross-entropy loss. Prompt and EEG token positions receive label `−100` (ignored).

**Attention mask**: All 174 tokens attend to each other (full causal attention with padding mask).

### 5.6 Model Parameter Summary

| Module | Parameters | Trainable |
|---|---|---|
| CSBrain Encoder | ~8,000,000 | No (frozen) |
| EEGProjection MLP | 4,608,000 | Yes (both phases) |
| TinyLlama base (4-bit) | ~1,100,000,000 | No |
| LoRA adapters | 1,126,400 | Yes (phase 2 only) |
| **Total trainable** | **~5,734,400** | |

---

## 6. Data Pipeline

### 6.1 EEG Preprocessing

Raw BCIC-IV-2a `.mat` files undergo the following preprocessing steps:

1. **Channel selection**: Extract 22 EEG channels (discard 3 EOG channels)
2. **Zero-mean normalization**: Subtract the per-sample mean
3. **Bandpass filtering**: 5th-order Butterworth filter, passband 0.3–50 Hz
4. **Motor imagery window extraction**: Crop 2s–6s post-cue (the active imagery period)
5. **Resampling**: Downsample from 250 Hz to 200 Hz → 800 samples per trial
6. **Reshaping**: `(22, 800)` → `(22, 4, 200)` — 4 temporal patches of 200 samples each
7. **Amplitude normalization**: Divide by 100 (approximate μV scale normalization)

### 6.2 Label-to-Text Mapping

A critical design choice is the conversion of 4 integer class labels into rich neuroscientific text targets. Each class has 3 paraphrase variants to provide training augmentation:

**Class 0 — Left Hand**
> "The EEG recording shows event-related desynchronization (ERD) predominantly over the right sensorimotor cortex, particularly at electrode C4. There is contralateral mu rhythm (8–12 Hz) suppression and beta band (13–30 Hz) desynchronization, consistent with left-hand motor imagery. The left hemisphere shows relatively preserved or enhanced activity."

**Class 1 — Right Hand**
> "This EEG pattern demonstrates strong ERD over the left sensorimotor cortex, especially at electrode C3. The mu and beta rhythms are suppressed contralaterally, indicating right-hand motor imagery. Right-lateralized sensorimotor activity with clear hemispheric asymmetry is present."

**Class 2 — Both Feet**
> "The EEG shows bilateral ERD over the central midline regions, particularly at electrode Cz and supplementary motor area. Both lower limbs' representations are active with symmetric desynchronization in mu and beta bands over bilateral central electrodes, consistent with foot/lower limb motor imagery."

**Class 3 — Tongue**
> "This recording exhibits ERD in the lateral portions of the sensorimotor cortex, associated with the tongue and orofacial motor representation. The tongue homunculus activates bilateral lateral motor areas with mu and beta band suppression, distinct from upper or lower limb patterns."

During training, a random paraphrase is selected per sample. During evaluation, paraphrase index 0 is always used.

### 6.3 LMDB Storage

Preprocessed EEG trials are stored in LMDB (Lightning Memory-Mapped Database) for efficient random access during training:

- Key format: `f"{subject}_{trial_idx}"`
- Value: pickled dict `{eeg: numpy array, label: int}`
- Enables multi-worker DataLoader without per-worker file handle conflicts

---

## 7. Training Strategy

### 7.1 Phase 1: Projection Warmup (5 Epochs)

**Motivation**: Before introducing LoRA updates, it is beneficial to first align the EEG token space to TinyLlama's expected input distribution. Random projection weights would inject high-magnitude noise into the LLM's attention layers, potentially destabilizing LoRA training.

| Setting | Value |
|---|---|
| Trainable modules | `EEGProjection` only |
| Learning rate | 5×10⁻⁴ (= base LR × 2.5) |
| LoRA parameters | Frozen |

### 7.2 Phase 2: Joint Fine-tuning (15 Epochs)

With the projection initialized to a reasonable EEG→LLM mapping, LoRA adapters are unfrozen and both projection and LoRA are jointly optimized.

| Setting | Value |
|---|---|
| Trainable modules | `EEGProjection` + LoRA adapters |
| Learning rate | 2×10⁻⁴ |

### 7.3 Common Training Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| Physical batch size | 4 | GPU memory constraint (8GB) |
| Gradient accumulation steps | 8 | Effective batch size = 32 |
| Optimizer | AdamW | Standard for transformer fine-tuning |
| Weight decay | 0.01 | Mild regularization |
| LR schedule | Cosine annealing | Smooth LR decay to eta_min=1e-6 |
| Gradient clipping | max_norm=1.0 | Prevent gradient explosion |
| Mixed precision | torch.amp.autocast (float16) | Speed + memory |
| Max target token length | 128 | Bounds generation length |
| Max new tokens (eval) | 32 | Keywords appear early; avoids OOM |
| Total epochs | 20 | 5 warmup + 15 joint |

---

## 8. Evaluation

### 8.1 Keyword Extraction Metric

Since the model generates open-ended text rather than a classification label, a keyword-matching scheme extracts the predicted class from the generated string:

| Class | Matching Keywords |
|---|---|
| Left Hand | "left hand", "left-lateralized", "right sensorimotor", "right central", "contralateral" |
| Right Hand | "right hand", "right-lateralized", "left sensorimotor", "left central", "contralateral" |
| Both Feet | "feet", "foot", "bilateral", "midline", "cz", "vertex", "lower limb", "supplementary motor" |
| Tongue | "tongue", "orofacial", "face", "lateral portions" |

The predicted class is the one whose keyword set has the highest match count in the lowercased generated text. Ties are broken by class index (first match wins).

**Limitation**: This metric conflates generation quality with classification accuracy. A perfectly accurate description of Left Hand imagery will score 100% for that sample; a fluent but incorrect description scores 0%.

### 8.2 Results

**Phase 1 — Projection Warmup Training Dynamics**

| Epoch | Training Loss | Val Accuracy |
|---|---|---|
| 1 | 0.7926 | 27.69% |
| 2 | 0.2141 | 27.43% |
| 3 | 0.0989 | 27.00% |
| 4 | 0.0639 | 26.91% |
| 5 | 0.0500 | **28.30%** ← best warmup |

**Phase 2 — Joint Fine-tuning Training Dynamics**

| Epoch | Training Loss | Val Accuracy |
|---|---|---|
| 1 | 0.0566 | 25.61% |
| 2 | 0.0519 | 26.30% |
| 3 | 0.0490 | 27.00% |
| 4 | 0.0480 | 29.50% |
| 5 | 0.0470 | 31.10% |
| 6 | 0.0456 | **36.81%** ← best overall |
| 7–14 | 0.044–0.038 | 26–33% |
| 15 | 0.0361 | 25.61% |

**Test Set Evaluation (best checkpoint at joint epoch 6)**

| Metric | Value |
|---|---|
| Correct predictions | 361 / 1152 |
| **Test Accuracy** | **31.34%** |
| Chance baseline (4-class) | 25.00% |
| Improvement over chance | +6.34 pp |

### 8.3 Sample Generated Outputs

The following examples illustrate the model's generation quality on held-out test samples:

**Example 1 — Incorrect Prediction**
```
True Label:      Left Hand
Predicted Label: Both Feet (keyword "bilateral" matched)
Generated Text:  "The EEG recording shows bilateral central desynchronization in the
                  mu band, particularly over midline electrodes. The sensorimotor
                  cortex shows broad suppression..."
```

**Example 2 — Incorrect Prediction**
```
True Label:      Right Hand
Predicted Label: Tongue (keyword "lateral" matched)
Generated Text:  "This recording exhibits lateral sensorimotor activity with
                  orofacial cortex activation patterns..."
```

**Example 3 — Correct Prediction**
```
True Label:      Right Hand
Predicted Label: Right Hand ✓
Generated Text:  "The EEG pattern demonstrates strong ERD over the left sensorimotor
                  cortex, especially at electrode C3. Mu and beta rhythms are
                  suppressed contralaterally, consistent with right-hand motor imagery."
```

The correct example demonstrates that the model can produce accurate, lateralized descriptions referencing the appropriate electrode (C3) and frequency bands (mu, beta), indicating meaningful EEG-to-language alignment.

---

## 9. Discussion

### 9.1 Architecture Choices

**Frozen CSBrain encoder**: Analogous to LLaVA's frozen CLIP encoder, this design avoids catastrophic forgetting and eliminates encoder gradient computation, saving ~6GB VRAM. The pretrained representations encode rich spatiotemporal EEG structure that would be difficult to learn from the limited BCIC-IV-2a training set.

**EEGTokenReducer**: Reducing from 88 to 12 tokens (7.3× compression) is critical for memory. Alternative designs (e.g., a learned Q-Former as in BLIP-2) may preserve more information but require additional parameters and training.

**4-bit quantization + LoRA**: This combination (QLoRA; Dettmers et al., 2023) enables fine-tuning a 1.1B-parameter model on 8GB VRAM with only 0.10% trainable parameters — a significant practical achievement for resource-constrained academic settings.

### 9.2 Training Dynamics

The two-phase training curve reveals an interesting pattern:
- **Phase 1**: Rapid loss decrease (0.79 → 0.05) but modest accuracy gains (27–28%), suggesting the projection learns to produce in-distribution LLM inputs but not yet semantically meaningful ones
- **Phase 2**: Loss continues decreasing (0.057 → 0.036) but accuracy peaks early (epoch 6: 36.81%) then degrades (epoch 15: 25.61%)

The accuracy degradation in late Phase 2 training is consistent with **overfitting to specific text patterns** while losing the ability to generate generalizable keyword-rich descriptions. Techniques like early stopping (applied here via checkpoint saving), stronger LoRA dropout, or smaller rank may mitigate this.

### 9.3 Limitations

1. **Keyword metric sensitivity**: "Contralateral" is a keyword for both Left and Right Hand, creating ambiguity. More precise keyword sets or a BLEU/ROUGE evaluation would be informative.
2. **Cross-subject generalization**: The test subjects (A08–A09) are completely unseen during training. With only 9 subjects total, the test set is limited.
3. **Text target diversity**: Only 3 paraphrases per class limits text diversity. LLM-generated augmentation could improve this.
4. **EEGTokenReducer design**: Simple mean pooling within regions may lose discriminative information. Attention-based pooling could be more expressive.
5. **Short generation at eval (max_new_tokens=32)**: While keywords appear early, some relevant keywords may be truncated in longer descriptions.

### 9.4 Comparison to Discriminative Baseline

Standard discriminative models on BCIC-IV-2a achieve 60–75% accuracy (with subject-specific training) or 40–55% (cross-subject). The proposed generative model at 31.34% is below these baselines, which is expected given:
- Generative decoding is a strictly harder task than classification
- The model must produce the correct text rather than merely selecting from 4 options
- Cross-subject evaluation is harder than subject-specific
- Only 5.7M parameters are trained vs. full model fine-tuning in discriminative approaches

The result is better understood as a proof-of-concept for generative EEG decoding rather than a competitive classification system.

---

## 10. Conclusion and Future Work

### 10.1 Conclusion

This project demonstrates the feasibility of **generative EEG-to-language decoding** using a modality-bridging architecture that combines a frozen EEG foundation model (CSBrain) with a quantized language model (TinyLlama-1.1B) via a compact trainable projection. The system:

- Runs entirely on consumer-grade 8GB VRAM hardware
- Requires only ~5.7M trainable parameters (projection MLP + LoRA adapters)
- Achieves 31.34% test accuracy on 4-class motor imagery decoding via keyword extraction from generated text — exceeding the 25% chance baseline by 6.34 percentage points
- Generates contextually appropriate neuroscientific language with correct lateralization, electrode references, and frequency band terminology

The two-phase training strategy (projection warmup → joint LoRA fine-tuning) proves effective at stabilizing training and reaching peak performance at joint epoch 6.

### 10.2 Future Work

**1. Improved EEG Tokenization**
- Learned Q-Former (BLIP-2 style) for adaptive EEG-to-language alignment
- Attention-based pooling in EEGTokenReducer instead of mean pooling
- Multi-scale temporal tokens preserving different temporal resolutions

**2. Richer Training Targets**
- LLM-generated augmentation: Use GPT-4 to create hundreds of paraphrases per class
- Chain-of-thought targets: "First, I observe... Therefore, the class is..."
- Subject-specific descriptions incorporating prior EEG knowledge

**3. Larger Language Models**
- Llama-3-8B or Mistral-7B with improved quantization (GPTQ, AWQ)
- Domain-adapted LLMs pretrained on neuroscience literature

**4. Better Evaluation**
- BLEU/ROUGE metrics for text quality
- Human evaluation by EEG experts
- Classification accuracy using a separate text classifier on generated outputs

**5. Multi-task Generalization**
- Extend to emotion recognition, sleep staging, and seizure detection tasks using the same architecture
- Instruction-tuned BCI assistant capable of answering questions about EEG recordings

**6. Interpretability**
- Attention visualization to understand which EEG tokens the LLM attends to when generating specific keywords
- Probing classifiers on projected EEG embeddings

---

## 11. References

1. Brunner, C., Leeb, R., Müller-Putz, G., Schlögl, A., & Pfurtscheller, G. (2008). BCI Competition 2008 – Graz data set A. *Institute for Knowledge Discovery (Laboratory of Brain-Computer Interfaces), Graz University of Technology.*

2. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. *Advances in Neural Information Processing Systems (NeurIPS) 36.*

3. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. *International Conference on Learning Representations (ICLR) 2022.*

4. Jiang, W., Zhao, L., & Lu, B. (2024). Large brain model for learning generic representations with tremendous EEG data in BCI. *International Conference on Learning Representations (ICLR) 2024.*

5. Kostas, D., Aroca-Ouellette, S., & Bhatt, M. (2020). BENDR: Using transformers and a contrastive self-supervised learning task to learn from physiological signals. *Frontiers in Human Neuroscience, 15.*

6. Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering, 15*(5).

7. Li, J., Li, D., Savarese, S., & Hoi, S. (2023). BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. *International Conference on Machine Learning (ICML) 2023.*

8. Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual instruction tuning. *Advances in Neural Information Processing Systems (NeurIPS) 36.*

9. Pfurtscheller, G., & Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. *Clinical Neurophysiology, 110*(11), 1842–1857.

10. Wang, J., Qin, Y., Wang, J., Li, J., & Wang, M. (2024). CBraMod: A criss-cross brain foundation model for EEG decoding. *arXiv preprint.*

11. **CSBrain Authors** (2025). CSBrain: Cross-scale spatiotemporal brain foundation model for EEG decoding. *Advances in Neural Information Processing Systems (NeurIPS) 2025 — Spotlight.*

12. Zhang, P., Chen, X., Wang, Y., et al. (2023). TinyLlama: An open-source small language model. *arXiv:2401.02385.*

---

## Appendix A: File Structure

```
CSBrain/
├── eeg_llm_notebook.ipynb        # Main project notebook (this work)
├── EEG_LLM_Architecture.md       # Architecture documentation
├── models/
│   ├── CSBrain.py                # Foundation model architecture
│   ├── CSBrain_transformer.py    # Supporting modules
│   └── CSBrain_transformerlayer.py # Custom transformer layers
├── data/
│   └── BCICIV2a/
│       ├── raw/                  # Downloaded .mat files
│       └── processed_lmdb/       # Preprocessed LMDB databases
├── pth/
│   └── CSBrain.pth               # Pretrained foundation model weights
├── pth_downtasks/
│   └── eeg_llm_bcic/
│       ├── best_projection.pth   # Best EEGProjection weights
│       └── best_lora/            # Best LoRA adapter weights (HF PEFT)
└── sh/
    └── finetune_CSBrain_BCIC.sh  # Fine-tuning script for BCIC
```

## Appendix B: Environment Setup

**Hardware**: Single GPU with ≥8GB VRAM (tested on NVIDIA GPU)
**Software**:
- Python 3.8+
- PyTorch ≥ 2.0 with CUDA
- Transformers ≥ 4.36 (HuggingFace)
- PEFT ≥ 0.7 (LoRA support)
- BitsAndBytes ≥ 0.41 (4-bit quantization)
- scipy, numpy, lmdb, tqdm

**Installation** (following CBraMod environment):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft bitsandbytes accelerate
pip install scipy numpy lmdb tqdm einops umap-learn scikit-learn
```

## Appendix C: Reproducibility

To reproduce the results:
1. Download BCIC-IV-2a data (auto-downloaded by the notebook from BNCI Horizon 2020)
2. Download CSBrain pretrained weights from Google Drive (link in README) → `pth/CSBrain.pth`
3. Run all cells in `eeg_llm_notebook.ipynb` sequentially
4. Training takes approximately 2–4 hours for 20 epochs on a modern GPU

**Random seed**: All experiments use `torch.manual_seed(42)`, `numpy.random.seed(42)`, and `random.seed(42)` for reproducibility.

---

*End of Report*

---

**Indian Institute of Technology Jodhpur**
NH 62, Nagaur Road, Karwar, Jodhpur, Rajasthan — 342 030
