import json
import os
import re
from unicodedata import normalize

import numpy as np
import onnxruntime as ort

AVAILABLE_LANGS = ["en", "ko", "es", "pt", "fr"]


class UnicodeProcessor:
    def __init__(self, unicode_indexer_path: str):
        with open(unicode_indexer_path, "r") as f:
            self.indexer = json.load(f)

    def _preprocess_text(self, text: str, lang: str) -> str:
        text = normalize("NFKD", text)

        emoji_pattern = re.compile(
            "[\U0001f600-\U0001f64f"
            "\U0001f300-\U0001f5ff"
            "\U0001f680-\U0001f6ff"
            "\U0001f700-\U0001f77f"
            "\U0001f780-\U0001f7ff"
            "\U0001f800-\U0001f8ff"
            "\U0001f900-\U0001f9ff"
            "\U0001fa00-\U0001fa6f"
            "\U0001fa70-\U0001faff"
            "\u2600-\u26ff"
            "\u2700-\u27bf"
            "\U0001f1e6-\U0001f1ff]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)

        replacements = {
            "–": "-", "‑": "-", "—": "-", "_": " ",
            "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
            "´": "'", "`": "'",
            "[": " ", "]": " ", "|": " ", "/": " ", "#": " ", "→": " ", "←": " ",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)

        text = re.sub(r"[♥☆♡©\\]", "", text)

        expr_replacements = {"@": " at ", "e.g.,": "for example, ", "i.e.,": "that is, "}
        for k, v in expr_replacements.items():
            text = text.replace(k, v)

        text = re.sub(r" ,", ",", text)
        text = re.sub(r" \.", ".", text)
        text = re.sub(r" !", "!", text)
        text = re.sub(r" \?", "?", text)
        text = re.sub(r" ;", ";", text)
        text = re.sub(r" :", ":", text)
        text = re.sub(r" '", "'", text)

        while '""' in text:
            text = text.replace('""', '"')
        while "''" in text:
            text = text.replace("''", "'")
        while "``" in text:
            text = text.replace("``", "`")

        text = re.sub(r"\s+", " ", text).strip()

        if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
            text += "."

        if lang not in AVAILABLE_LANGS:
            raise ValueError(f"Invalid language: {lang}")
        text = f"<{lang}>" + text + f"</{lang}>"
        return text

    def __call__(self, text: str, lang: str) -> tuple[np.ndarray, np.ndarray]:
        text = self._preprocess_text(text, lang)
        text_ids_length = np.array([len(text)], dtype=np.int64)
        unicode_vals = np.array([ord(c) for c in text], dtype=np.uint16)
        text_ids = np.array(
            [[self.indexer[val] for val in unicode_vals]], dtype=np.int64
        )
        text_mask = length_to_mask(text_ids_length)
        return text_ids, text_mask


class Style:
    def __init__(self, style_ttl: np.ndarray, style_dp: np.ndarray):
        self.ttl = style_ttl
        self.dp = style_dp


class TextToSpeech:
    def __init__(self, cfgs, text_processor, dp_ort, text_enc_ort, vector_est_ort, vocoder_ort):
        self.cfgs = cfgs
        self.text_processor = text_processor
        self.dp_ort = dp_ort
        self.text_enc_ort = text_enc_ort
        self.vector_est_ort = vector_est_ort
        self.vocoder_ort = vocoder_ort
        self.sample_rate = cfgs["ae"]["sample_rate"]
        self.base_chunk_size = cfgs["ae"]["base_chunk_size"]
        self.chunk_compress_factor = cfgs["ttl"]["chunk_compress_factor"]
        self.ldim = cfgs["ttl"]["latent_dim"]

    def _sample_noisy_latent(self, duration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        wav_len_max = duration.max() * self.sample_rate
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = ((wav_len_max + chunk_size - 1) / chunk_size).astype(np.int32)
        latent_dim = self.ldim * self.chunk_compress_factor
        noisy_latent = np.random.randn(1, latent_dim, latent_len).astype(np.float32)
        latent_mask = get_latent_mask(wav_lengths, self.base_chunk_size, self.chunk_compress_factor)
        noisy_latent = noisy_latent * latent_mask
        return noisy_latent, latent_mask

    def _infer_chunk(self, text: str, lang: str, style: Style, total_step: int, speed: float) -> tuple[np.ndarray, np.ndarray]:
        text_ids, text_mask = self.text_processor(text, lang)
        dur, *_ = self.dp_ort.run(None, {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask})
        dur = dur / speed
        text_emb, *_ = self.text_enc_ort.run(None, {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask})
        xt, latent_mask = self._sample_noisy_latent(dur)
        total_step_np = np.array([total_step], dtype=np.float32)
        for step in range(total_step):
            current_step = np.array([step], dtype=np.float32)
            xt, *_ = self.vector_est_ort.run(None, {
                "noisy_latent": xt, "text_emb": text_emb, "style_ttl": style.ttl,
                "text_mask": text_mask, "latent_mask": latent_mask,
                "current_step": current_step, "total_step": total_step_np,
            })
        wav, *_ = self.vocoder_ort.run(None, {"latent": xt})
        return wav, dur

    def __call__(self, text: str, lang: str, style: Style, total_step: int = 5, speed: float = 1.05, silence_duration: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
        max_len = 120 if lang == "ko" else 300
        chunks = chunk_text(text, max_len=max_len)
        wav_cat = None
        dur_cat = None
        for chunk in chunks:
            wav, dur = self._infer_chunk(chunk, lang, style, total_step, speed)
            if wav_cat is None:
                wav_cat = wav
                dur_cat = dur
            else:
                silence = np.zeros((1, int(silence_duration * self.sample_rate)), dtype=np.float32)
                wav_cat = np.concatenate([wav_cat, silence, wav], axis=1)
                dur_cat += dur + silence_duration
        total_samples = int(self.sample_rate * dur_cat.item())
        return wav_cat[0, :total_samples], self.sample_rate


def length_to_mask(lengths: np.ndarray, max_len=None) -> np.ndarray:
    max_len = max_len or lengths.max()
    ids = np.arange(0, max_len)
    mask = (ids < np.expand_dims(lengths, axis=1)).astype(np.float32)
    return mask.reshape(-1, 1, max_len)


def get_latent_mask(wav_lengths: np.ndarray, base_chunk_size: int, chunk_compress_factor: int) -> np.ndarray:
    latent_size = base_chunk_size * chunk_compress_factor
    latent_lengths = (wav_lengths + latent_size - 1) // latent_size
    return length_to_mask(latent_lengths)


def load_text_to_speech(onnx_dir: str) -> TextToSpeech:
    opts = ort.SessionOptions()
    providers = ["CPUExecutionProvider"]
    load = lambda name: ort.InferenceSession(os.path.join(onnx_dir, name), sess_options=opts, providers=providers)
    dp = load("duration_predictor.onnx")
    text_enc = load("text_encoder.onnx")
    vector_est = load("vector_estimator.onnx")
    vocoder = load("vocoder.onnx")
    with open(os.path.join(onnx_dir, "tts.json"), "r") as f:
        cfgs = json.load(f)
    text_processor = UnicodeProcessor(os.path.join(onnx_dir, "unicode_indexer.json"))
    return TextToSpeech(cfgs, text_processor, dp, text_enc, vector_est, vocoder)


def load_voice_style(path: str) -> Style:
    with open(path, "r") as f:
        data = json.load(f)
    ttl_dims = data["style_ttl"]["dims"]
    dp_dims = data["style_dp"]["dims"]
    ttl = np.array(data["style_ttl"]["data"], dtype=np.float32).reshape(1, ttl_dims[1], ttl_dims[2])
    dp = np.array(data["style_dp"]["data"], dtype=np.float32).reshape(1, dp_dims[1], dp_dims[2])
    return Style(ttl, dp)


def chunk_text(text: str, max_len: int = 300) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    chunks = []
    for paragraph in paragraphs:
        pattern = r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+"
        sentences = re.split(pattern, paragraph)
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_len:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
    return chunks
