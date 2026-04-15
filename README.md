# SupertonicTTS Voice Style Extractor

This repo will guide you to extract voice style embeddings from any WAV file for SupertonicTTS,
without the official (unreleased) style encoder.

Put in a 3~10 second voice sample, and get a voice style JSON that makes SupertonicTTS speak in that voice.

**Requirements:** NVIDIA GPU with 4GB+ VRAM, CUDA support.

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
3. Optimization (~2-5 minutes, avg 494 steps on RTX 3090)

## Performance
Evaluated on 20 speakers × 5 utterances = 100 samples:

| | SIM ↑ | WER ↓ |
|---|---|---|
| Preset styles (no cloning) | — | 1.72% |
| **Proposed method** | **0.874** | **1.30%** |

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
Based on probing analysis by [Chen et al. (2025)](https://arxiv.org/abs/2501.05310), WavLM Layer 3 best encodes speaker identity (100% accuracy). We compare time-averaged feature statistics (mean, std) between generated and target audio. Time-axis averaging removes content dependency.

### Style Space
- `style_ttl` [1, 50, 256] = 12,800 parameters (timbre, optimized)
- `style_dp` [1, 8, 16] = 128 parameters (rhythm/duration, frozen)

Male and female voices occupy separate regions in style space. Starting from the closest preset style is critical.

## Looking for help?
If you have any questions, please feel free to open an issue.
