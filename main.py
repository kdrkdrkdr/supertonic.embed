import os
import numpy as np
import torch
import soundfile as sf
from helper import load_text_to_speech, load_voice_style

NAME = "hutao"
LANG = "en"
VOICE_STYLE = f"logs/{NAME}/{NAME}_final.json"
TEXT = "어렸을 때부터 하늘을 올려다보는 것을 좋아했어요. 길을 걷다 보면 문득 익숙한 냄새가 기억을 떠올리게 하죠."

tts = load_text_to_speech("onnx")
style = load_voice_style(VOICE_STYLE)
wav, duration = tts(TEXT, LANG, style)
sf.write(f"{NAME}.wav", wav, tts.sample_rate)