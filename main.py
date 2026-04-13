import os
import numpy as np
import torch
import soundfile as sf
from helper import load_text_to_speech, load_voice_style

# ===== Settings =====

SEED = 42
VOICE_STYLE = "voice_styles/ljs.json"
TEXT = "You get in life what you have the courage to ask for."
LANG = "en"

# ===== Run =====

np.random.seed(SEED)
torch.manual_seed(SEED)

tts = load_text_to_speech("onnx")
style = load_voice_style(VOICE_STYLE)
wav, duration = tts(TEXT, LANG, style)
os.makedirs("results", exist_ok=True)
sf.write("results/ljs_optimized.wav", wav, tts.sample_rate)
print(f"Saved: results/ljs_optimized.wav")
