"""
Extract voice style JSON from a WAV file for SupertonicTTS.

Core approach:
  - Convert ONNX TTS models to PyTorch (enables gradient backpropagation)
  - Optimize style vectors via HuBERT multi-layer feature matching
  - Multiple text rotation prevents overfitting to specific phonemes

Usage:
    python optimize_style.py ljs         # uses configs/ljs.json
    python optimize_style.py zhongli     # uses configs/zhongli.json
"""

import json
import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa
import soundfile as sf
import onnxslim
import onnx
from onnx import shape_inference
import onnx2torch
from onnx2torch import convert

from helper import load_text_to_speech, load_voice_style

# SSL certificate workaround
os.environ.pop('SSL_CERT_FILE', None)
os.environ.pop('CURL_CA_BUNDLE', None)
os.environ.pop('REQUESTS_CA_BUNDLE', None)
import httpx
_orig_client = httpx.Client
class _NoVerifyClient(_orig_client):
    def __init__(self, *args, **kwargs):
        kwargs['verify'] = False
        super().__init__(*args, **kwargs)
httpx.Client = _NoVerifyClient

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== ONNX to PyTorch conversion =====

def _patch_onnx2torch():
    """Bypass onnx2torch's safe_shape_inference which writes temp files."""
    def patched(m):
        if isinstance(m, str):
            m = onnx.load(m)
        try:
            return shape_inference.infer_shapes(m)
        except:
            return m
    onnx2torch.converter.safe_shape_inference = patched

def _fix_clip(model):
    """Remove empty Clip inputs that cause onnx2torch conversion errors."""
    for node in model.graph.node:
        if node.op_type == 'Clip':
            inputs = list(node.input)
            while inputs and inputs[-1] == '':
                inputs.pop()
            del node.input[:]
            node.input.extend(inputs)
    return model

def load_pt_model(name, onnx_dir="onnx"):
    """Load ONNX model, slim it, fix opset, and convert to PyTorch."""
    slimmed = onnxslim.slim(os.path.join(onnx_dir, name))
    for opset in slimmed.opset_import:
        if opset.domain == '' or opset.domain == 'ai.onnx':
            opset.version = 17
    _fix_clip(slimmed)
    m = convert(slimmed)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(DEVICE)

# ===== HuBERT perceptual loss =====

def load_hubert():
    """Load pretrained HuBERT-Large for multi-layer feature extraction."""
    from transformers import HubertModel
    model = HubertModel.from_pretrained('facebook/hubert-large-ls960-ft').to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def hubert_feature_loss(hubert, gen_wav, target_features, layers=[1,3,5,7,9,11]):
    """
    Compare HuBERT features across multiple layers.

    For each layer, compares:
      - Time-averaged mean (captures overall spectral characteristics)
      - Time-averaged std (captures energy dynamics)
      - Gram matrix (captures feature correlations = "style")

    Time-averaging removes content dependency.
    """
    if gen_wav.ndim == 1:
        gen_wav = gen_wav.unsqueeze(0)
    gen_wav_16k = torchaudio.functional.resample(gen_wav, 44100, 16000)
    gen_out = hubert(gen_wav_16k, output_hidden_states=True)

    loss = 0.0
    for l in layers:
        gen_feat = gen_out.hidden_states[l]
        tgt_mean, tgt_std, tgt_gram = target_features[l]

        gen_mean = gen_feat.mean(dim=1)
        gen_std = gen_feat.std(dim=1)
        loss += F.mse_loss(gen_mean, tgt_mean)
        loss += F.mse_loss(gen_std, tgt_std)

        gen_gram = torch.bmm(gen_feat.transpose(1, 2), gen_feat) / gen_feat.shape[1]
        loss += F.mse_loss(gen_gram, tgt_gram) * 0.1

    return loss / len(layers)

def extract_hubert_targets(hubert, target_wav, layers=[1,3,5,7,9,11]):
    """Pre-compute HuBERT feature statistics from target WAV."""
    if target_wav.ndim == 1:
        target_wav = target_wav.unsqueeze(0)
    wav_16k = torchaudio.functional.resample(target_wav, 44100, 16000)

    with torch.no_grad():
        out = hubert(wav_16k, output_hidden_states=True)

    features = {}
    for l in layers:
        feat = out.hidden_states[l]
        mean = feat.mean(dim=1)
        std = feat.std(dim=1)
        gram = torch.bmm(feat.transpose(1, 2), feat) / feat.shape[1]
        features[l] = (mean, std, gram)
    return features

# ===== TTS forward pass =====

def tts_forward(text_ids, text_mask, style_ttl, style_dp,
                dp_model, te_model, ve_model, voc_model,
                total_step, speed, noisy_latent, latent_mask):
    """Differentiable TTS forward pass through all 4 models."""
    dur = dp_model(text_ids, style_dp, text_mask)
    dur = dur / speed
    text_emb = te_model(text_ids, style_ttl, text_mask)
    xt = noisy_latent * latent_mask
    total_step_t = torch.tensor([total_step], dtype=torch.float32).to(DEVICE)
    for step in range(total_step):
        current_step_t = torch.tensor([step], dtype=torch.float32).to(DEVICE)
        xt = ve_model(xt, text_emb, style_ttl, latent_mask, text_mask, current_step_t, total_step_t)
    wav = voc_model(xt)
    return wav, dur

# ===== Save style JSON =====

def save_style(path, style_ttl, style_dp, source_file=None):
    """Save style vectors in SupertonicTTS-compatible JSON format."""
    from datetime import datetime
    style_json = {
        "style_ttl": {
            "data": style_ttl.cpu().numpy().tolist(),
            "dims": [1, 50, 256],
            "type": "float32"
        },
        "style_dp": {
            "data": style_dp.cpu().numpy().tolist(),
            "dims": [1, 8, 16],
            "type": "float32"
        },
        "metadata": {
            "source_file": source_file or "unknown",
            "source_sample_rate": 44100,
            "target_sample_rate": 44100,
            "extracted_at": datetime.now().isoformat()
        }
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(style_json, f)

# ===== Main =====

def main():
    _patch_onnx2torch()

    # Load config
    arg = sys.argv[1] if len(sys.argv) > 1 else "ljs"
    if os.path.exists(arg):
        config_path = arg
    elif os.path.exists(f"configs/{arg}.json"):
        config_path = f"configs/{arg}.json"
    elif os.path.exists(f"configs/{arg}"):
        config_path = f"configs/{arg}"
    else:
        print(f"Config not found: {arg}")
        sys.exit(1)

    print(f"Loading config: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    name = cfg["name"]
    target_wav_path = cfg["target_wav"]
    reference_style = cfg.get("reference_style")
    seed = cfg.get("seed", 42)
    lr = cfg.get("lr", 2e-4)
    num_steps = cfg.get("num_steps", 3000)
    total_step = cfg.get("total_step", 5)
    speed = cfg.get("speed", 1.05)
    save_every = cfg.get("save_every", 500)

    # Texts for multi-text rotation (content-independent, prevents overfitting)
    opt_texts = [
        "The sun sets behind the mountains, painting the sky in shades of pink and orange.",
        "I took a walk in the park this morning, and the sound of birds was so pleasant.",
        "Would you like to sit down and have a cup of tea together?",
        "The wind carries the scent of flowers from the garden across the valley.",
        "Every great journey begins with a single step forward into the unknown.",
    ]
    opt_lang = "en"

    # Paths
    output_json = f"voice_styles/{name}.json"
    log_dir = f"logs/{name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Save config to log dir
    with open(os.path.join(log_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)

    # Find latest checkpoint for auto-resume
    def find_latest_checkpoint():
        pattern = os.path.join(log_dir, f"{name}_*.json")
        files = [f for f in glob.glob(pattern) if "train_config" not in f]
        if not files:
            return None, 0
        latest = max(files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split("_")[-1]))
        step_num = int(os.path.splitext(os.path.basename(latest))[0].split("_")[-1])
        return latest, step_num

    latest_ckpt, start_step = find_latest_checkpoint()

    print(f"Using device: {DEVICE}")
    print(f"Name: {name}")

    # ===== 1. Load target WAV and extract HuBERT features =====
    print(f"\nLoading target WAV: {target_wav_path}")
    target_wav, _ = librosa.load(target_wav_path, sr=44100)
    target_wav_t = torch.tensor(target_wav, dtype=torch.float32).to(DEVICE)
    print(f"  Duration: {len(target_wav)/44100:.2f}s")

    print("\nLoading HuBERT...")
    hubert = load_hubert()
    print("  HuBERT loaded.")

    hubert_layers = [1, 3, 5, 7, 9, 11]
    print("Extracting target features...")
    target_hubert_feats = extract_hubert_targets(hubert, target_wav_t, hubert_layers)
    print("  Done.")

    # ===== 2. Load TTS models (ONNX -> PyTorch) =====
    print("\nConverting ONNX models to PyTorch...")
    dp_model = load_pt_model("duration_predictor.onnx")
    te_model = load_pt_model("text_encoder.onnx")
    ve_model = load_pt_model("vector_estimator.onnx")
    voc_model = load_pt_model("vocoder.onnx")
    print("  All models converted.")

    # ===== 3. Preprocess texts =====
    tts = load_text_to_speech("onnx")
    text_inputs = []
    for text in opt_texts:
        ids_np, mask_np = tts.text_processor(text, opt_lang)
        text_inputs.append((
            torch.tensor(ids_np, dtype=torch.long).to(DEVICE),
            torch.tensor(mask_np, dtype=torch.float32).to(DEVICE)
        ))

    # ===== 4. Generate fixed noisy latent (seed-controlled) =====
    torch.manual_seed(seed)
    np.random.seed(seed)
    tmp_style = load_voice_style("voice_styles/M1.json")
    tmp_dp = torch.tensor(tmp_style.dp, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        init_dur = dp_model(text_inputs[0][0], tmp_dp, text_inputs[0][1]) / speed
    dur_val = init_dur.item()
    wav_len = int(dur_val * 44100)
    chunk_size = tts.base_chunk_size * tts.chunk_compress_factor
    latent_len = int(np.ceil(wav_len / chunk_size))
    latent_dim = tts.ldim * tts.chunk_compress_factor
    noisy_latent_fixed = torch.tensor(np.random.randn(1, latent_dim, latent_len).astype(np.float32)).to(DEVICE)
    latent_mask = torch.ones(1, 1, latent_len, dtype=torch.float32).to(DEVICE)
    del tmp_style, tmp_dp

    # ===== 5. Initialize style vectors =====
    if latest_ckpt:
        # Auto-resume from latest checkpoint
        print(f"\nResuming from: {latest_ckpt} (step {start_step})")
        ref_style = load_voice_style(latest_ckpt)
        style_ttl = torch.tensor(ref_style.ttl, dtype=torch.float32).to(DEVICE).clone().requires_grad_(True)
        style_dp = torch.tensor(ref_style.dp, dtype=torch.float32).to(DEVICE).clone()
    elif reference_style == "auto":
        # Auto-select closest preset style via HuBERT distance
        print("\nFinding closest style to target WAV...")
        all_style_paths = sorted(glob.glob("voice_styles/[FM]*.json"))
        best_dist = float('inf')
        best_path = None
        for sp in all_style_paths:
            s = load_voice_style(sp)
            s_ttl = torch.tensor(s.ttl, dtype=torch.float32).to(DEVICE)
            s_dp = torch.tensor(s.dp, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                test_wav, _ = tts_forward(
                    text_inputs[0][0], text_inputs[0][1], s_ttl, s_dp,
                    dp_model, te_model, ve_model, voc_model,
                    total_step, speed, noisy_latent_fixed, latent_mask,
                )
                dist = hubert_feature_loss(hubert, test_wav.squeeze(), target_hubert_feats, hubert_layers).item()
            print(f"  {os.path.basename(sp)}: {dist:.4f}")
            if dist < best_dist:
                best_dist = dist
                best_path = sp
        print(f"  >> Best: {os.path.basename(best_path)} (dist={best_dist:.4f})")
        ref_style = load_voice_style(best_path)
        style_ttl = torch.tensor(ref_style.ttl, dtype=torch.float32).to(DEVICE).clone().requires_grad_(True)
        style_dp = torch.tensor(ref_style.dp, dtype=torch.float32).to(DEVICE).clone()
    elif reference_style:
        # Use specified reference style
        print(f"\nInitializing style from: {reference_style}")
        ref_style = load_voice_style(reference_style)
        style_ttl = torch.tensor(ref_style.ttl, dtype=torch.float32).to(DEVICE).clone().requires_grad_(True)
        style_dp = torch.tensor(ref_style.dp, dtype=torch.float32).to(DEVICE).clone()
    else:
        # Random initialization (not recommended)
        print("\nInitializing style randomly")
        style_ttl = (torch.randn(1, 50, 256) * 0.1).to(DEVICE).requires_grad_(True)
        style_dp = torch.tensor(load_voice_style("voice_styles/M1.json").dp, dtype=torch.float32).to(DEVICE).clone()

    print(f"  style_ttl: {style_ttl.shape}, style_dp: {style_dp.shape} (dp frozen)")

    # ===== 6. Optimization (style_ttl only, style_dp frozen) =====
    optimizer = torch.optim.Adam([style_ttl], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=200, factor=0.5, min_lr=lr * 0.01
    )

    end_step = num_steps
    if start_step >= end_step:
        print(f"\nAlready reached target step ({end_step}). Nothing to do.")
        return

    print(f"\nStarting optimization (step {start_step+1} -> {end_step})...")

    best_loss = float('inf')
    best_ttl = None
    best_dp = style_dp.detach().clone()

    for step in range(start_step, end_step):
        optimizer.zero_grad()

        # Rotate through texts each step
        text_idx = step % len(text_inputs)
        text_ids, text_mask = text_inputs[text_idx]

        # Forward pass
        wav_out, _ = tts_forward(
            text_ids, text_mask, style_ttl, style_dp,
            dp_model, te_model, ve_model, voc_model,
            total_step, speed, noisy_latent_fixed, latent_mask,
        )
        gen_wav = wav_out.squeeze()

        # Compute loss
        loss = hubert_feature_loss(hubert, gen_wav, target_hubert_feats, hubert_layers)

        # Backward + update
        loss.backward()
        torch.nn.utils.clip_grad_norm_([style_ttl], max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_ttl = style_ttl.detach().clone()

        # Log
        if (step + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Step {step+1}/{end_step} | Loss: {loss.item():.9f} | LR: {current_lr:.9f} | Best: {best_loss:.9f}")

        # Save checkpoint
        if (step + 1) % save_every == 0:
            ckpt_path = f"{log_dir}/{name}_{step+1:08d}.json"
            save_style(ckpt_path, best_ttl, best_dp, target_wav_path)
            print(f"  >> Checkpoint saved: {ckpt_path}")

    # ===== 7. Save final result =====
    print(f"\nSaving best style to: {output_json}")
    save_style(output_json, best_ttl, best_dp, target_wav_path)
    print(f"  Done! Best loss: {best_loss:.9f}")

    # ===== 8. Generate test audio =====
    print("\nGenerating test audio with optimized style...")
    with torch.no_grad():
        wav_test, _ = tts_forward(
            text_inputs[0][0], text_inputs[0][1],
            best_ttl.to(DEVICE), best_dp.to(DEVICE),
            dp_model, te_model, ve_model, voc_model,
            total_step, speed, noisy_latent_fixed, latent_mask,
        )
    wav_np = wav_test.squeeze().cpu().numpy()
    result_path = f"results/{name}_optimized.wav"
    sf.write(result_path, wav_np, 44100)
    print(f"  Saved: {result_path}")


if __name__ == "__main__":
    main()
