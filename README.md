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
      │                      │   HuBERT   │◄─────────────────────┘
      │                      │  (6 layer  │
      │                      │  feature   │◄── target WAV
      │                      │ matching)  │
      │                      └─────┬──────┘
      │                            │
      │      gradient              │ loss
      └────────────────────────────┘
      "update style to be more similar"
```

1. Auto-selects the closest preset style (F1~F5, M1~M5) as starting point
2. Synthesizes WAV via TTS, compares with target WAV using HuBERT features
3. Updates style vector via gradient descent, repeats thousands of times

### Convergence Guide:
Best loss reaching **2.0~2.4** is considered same-voice level. Lower is better.

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
    "save_every": 500
}
```

| Parameter | Description |
|-----------|-------------|
| `name` | Name for saving checkpoints and output files |
| `target_wav` | Path to the voice WAV file to clone |
| `reference_style` | `"auto"` (auto-select closest) or `"voice_styles/F1.json"` (manual) |
| `seed` | Random seed for reproducibility |
| `lr` | Learning rate. 2e-4 recommended. Too high breaks pronunciation, too low is slow |
| `num_steps` | Number of optimization steps |
| `save_every` | Checkpoint save interval |

### 5. Run optimization
```bash
python optimize_style.py ljs
```
Training auto-resumes from the latest checkpoint if interrupted.

### 6. Use the extracted style
See `main.py` for inference example.

## How long does it take?
1. Model loading & conversion (~30 seconds)
2. Auto style selection (~1 minute, 10 styles compared)
3. Optimization (~10-30 minutes for 3000 steps on RTX 3090)

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
    ├── ljs_00000500.json
    ├── ljs_00001000.json
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
| HuBERT-Large | Multi-layer perceptual loss (facebook/hubert-large-ls960-ft) |

## Technical Details

### ONNX to PyTorch Conversion
ONNX models are converted to PyTorch for gradient backpropagation:
- onnxslim for model cleanup
- Forced opset 17 (onnx2torch compatibility)
- Clip node empty input fix

### HuBERT Multi-Layer Feature Matching
Compares features from 6 HuBERT layers (1, 3, 5, 7, 9, 11):
- **Low layers (1~3)**: Timbre, tone
- **Mid layers (5~7)**: Pronunciation, prosody
- **High layers (9~11)**: Speaker identity
- **Gram matrix**: Style correlations (Neural Style Transfer for audio)

Time-axis averaging removes content dependency, only voice characteristics are compared.

### Style Space
- `style_ttl` [1, 50, 256] = 12,800 parameters (timbre, optimized)
- `style_dp` [1, 8, 16] = 128 parameters (rhythm/duration, frozen)

Male and female voices occupy separate regions in style space. Starting from the closest preset style is critical.

## Looking for help?
If you have any questions, please feel free to open an issue.
