# SupertonicTTS Voice Style Extractor

This repo will guide you to extract voice style embeddings from any WAV file for SupertonicTTS,
without the official (unreleased) style encoder.

Put in a 3~10 second voice sample, and get a voice style JSON that makes SupertonicTTS speak in that voice.

**Requirements:** NVIDIA GPU with 4GB+ VRAM, CUDA support.

## Responsible Use

**This is research code released for academic purposes only.** Voice cloning technology can be misused for serious harm. By using this repository, you agree to the following.

- **Obtain explicit consent** from any speaker whose voice you intend to clone. Cloning a real person's voice without permission may be illegal in your jurisdiction.
- **Do not use this tool** for non-consensual voice impersonation, voice phishing (vishing), fraud, harassment, defamation, generation of misleading political or commercial content, or circumvention of voice-based authentication.
- **Do not target identifiable individuals** (including public figures) without their explicit permission.
- **Disclose synthetic audio** as AI-generated when distributing it, ideally with a watermark or provenance metadata (e.g., C2PA).

The author disclaims all liability for misuse. Evaluation in the accompanying paper uses Common Voice (CC0 license) and does not target identifiable individuals. If you observe misuse of this code, please open an issue.

### How it works:
```
              ┌───────────────────────────────────────────┐
              │           TTS Pipeline (PyTorch)          │
┌───────────┐ │ ┌─────────┐  ┌───────────┐  ┌───────────┐ │ ┌─────────┐
│   style   │→│ │  Text   │→ │  Vector   │→ │  Vocoder  │ │→│ gen WAV │
│  vector   │ │ │ Encoder │  │ Estimator │  │           │ │ └────┬────┘
└─────┬─────┘ │ └─────────┘  └───────────┘  └───────────┘ │      │
      │       └───────────────────────────────────────────┘      │
      │                                                          │
      │                      ┌────────────┐                      │
      │                      │   WavLM    │◄─────────────────────┘
      │                      │  Layer 3   │
      │                      │  (speaker  │◄── target WAV
      │                      │ identity)  │
      │                      └─────┬──────┘
      │                            │
      │      gradient              │ loss
      └────────────────────────────┘
      "update style to be more similar"
```

1. Auto-selects the closest preset style (F1~F5, M1~M5) via WavLM Layer 3 distance
2. Synthesizes WAV via TTS, compares with target WAV using WavLM Layer 3 features
3. Updates style vector via gradient descent until convergence (early stop at 0.24)

### Convergence Guide:
Same-speaker baseline loss is **0.15~0.24**. Optimization stops automatically when this threshold is reached.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download SupertonicTTS models
Download `onnx/` and `voice_styles/` folders from [Supertone/supertonic-2](https://huggingface.co/Supertone/supertonic-2) and place them in the project root.

### 3. Prepare your voice sample
Place your WAV file (3~10 seconds, single speaker) in the `wavs/` folder. Any sample rate is fine (auto-resampled to 44.1kHz).

### 4. Create config
Create `configs/ljs.json`:
```json
{
    "name": "ljs",
    "target_wav": "wavs/ljs.wav",
    "reference_style": "auto",
    "seed": 42,
    "lr": 2e-4,
    "num_steps": 3000,
    "total_step": 5,
    "speed": 1.05,
    "save_every": 100
}
```

| Parameter | Description |
|-----------|-------------|
| `name` | Name for saving checkpoints and output files |
| `target_wav` | Path to the voice WAV file to clone |
| `reference_style` | `"auto"` (auto-select closest) or `"voice_styles/F1.json"` (manual) |
| `seed` | Random seed for reproducibility |
| `lr` | Learning rate. 2e-4 recommended. Too high breaks pronunciation, too low is slow |
| `num_steps` | Max optimization steps (early stopping may stop sooner) |
| `save_every` | Checkpoint save interval |

### 5. Run optimization
```bash
python optimize_style.py ljs
```
Training auto-resumes from the latest checkpoint if interrupted. Early stops when loss reaches 0.24 (same-speaker level).

### 6. Use the extracted style
See `main.py` for inference example.

## How long does it take?
1. Model loading & conversion (~30 seconds)
2. Auto style selection (~1 minute, 10 styles compared)
3. Optimization (~5-6 minutes, avg 503 steps on RTX 3090)

## Performance
Evaluated on 44 speakers × 5 utterances = 220 samples:

| | SIM (WavLM) ↑ | SIM (ECAPA) ↑ | SIM (ResNet) ↑ | WER ↓ |
|---|---|---|---|---|
| Nearest preset (no opt.) | 0.758 | 0.129 | 0.112 | 4.80% |
| **Proposed method** | **0.867** | **0.452** | **0.446** | **2.70%** |

## File Structure
```
configs/                  # Training configs
├── ljs.json
└── ...

wavs/                     # Reference WAV files
├── ljs.wav
└── ...

onnx/                     # SupertonicTTS ONNX models
├── duration_predictor.onnx
├── text_encoder.onnx
├── vector_estimator.onnx
├── vocoder.onnx
├── tts.json
└── unicode_indexer.json

voice_styles/             # Voice style JSONs
├── M1~M5.json, F1~F5.json  (presets)
└── ljs.json                 (extracted)

logs/                     # Checkpoints
└── ljs/
    ├── train_config.json
    ├── ljs_00000100.json
    └── ...

results/                  # Test outputs
└── ljs_optimized.wav
```

## Models Used

| Model | Role |
|-------|------|
| duration_predictor | Duration prediction (SupertonicTTS) |
| text_encoder | Text encoding (SupertonicTTS) |
| vector_estimator | Flow matching denoising (SupertonicTTS) |
| vocoder | Latent to WAV (SupertonicTTS) |
| WavLM-Large | Perceptual loss, Layer 3 (microsoft/wavlm-large) |

## Technical Details

### ONNX to PyTorch Conversion
ONNX models are converted to PyTorch for gradient backpropagation:
- onnxslim for model cleanup
- Forced opset 17 (onnx2torch compatibility)
- Clip node empty input fix

### WavLM Layer 3 Feature Matching
Based on probing analysis by [Chiu et al. (2025)](https://arxiv.org/abs/2501.05310), WavLM Layer 3 best encodes speaker identity. We compare time-averaged feature statistics (mean, std) between generated and target audio. Time-axis averaging reduces content dependency.

### Style Space
- `style_ttl` [1, 50, 256] = 12,800 parameters (timbre, optimized)
- `style_dp` [1, 8, 16] = 128 parameters (rhythm/duration, frozen)

Male and female voices occupy separate regions in style space. Starting from the closest preset style is critical.

## Citation

If you use this work, please cite:

```bibtex
@misc{kim2026supertonicembed,
  author       = {Gyeongmin Kim},
  title        = {Extracting Voice Styles from Frozen TTS Models via Gradient-Based Inverse Optimization},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19646514},
  url          = {https://doi.org/10.5281/zenodo.19646514}
}
```

Preprint available on Zenodo: https://doi.org/10.5281/zenodo.19646514

## Looking for help?
If you have any questions, please feel free to open an issue.
