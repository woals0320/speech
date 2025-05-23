# pip install streamlit transformers torchaudio pyaudio scipy
# pip install streamlit==1.31.0

import streamlit as st
import torch
import pyaudio
import wave
import tempfile
import numpy as np
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import scipy.io.wavfile as wavfile


# -------------------------------
# 모델 로드
# -------------------------------
@st.cache_resource
def load_models():
    # 음성 인식용 Wav2Vec2
    processor = Wav2Vec2Processor.from_pretrained("./my-finetuned-wav2vec2", local_files_only=True)
    wav2vec_model = Wav2Vec2ForCTC.from_pretrained("./my-finetuned-wav2vec2", local_files_only=True)

    # 문장 교정용 KoGrammar
    grammar_tokenizer = AutoTokenizer.from_pretrained("./KoBART", local_files_only=True)
    grammar_model = AutoModelForSeq2SeqLM.from_pretrained("./KoBART", local_files_only=True)

    return processor, wav2vec_model, grammar_tokenizer, grammar_model

processor, wav2vec_model, grammar_tokenizer, grammar_model = load_models()


# -------------------------------
# PyAudio로 녹음
# -------------------------------
def record_audio(filename="recorded.wav", duration=5, fs=16000):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1

    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels,
                    rate=fs, input=True,
                    frames_per_buffer=chunk)
    st.info("🎙️ 녹음 중입니다. 말을 하세요...")
    frames = []

    for _ in range(0, int(fs / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    st.success("녹음 완료!")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename


# -------------------------------
# Wav2Vec2 음성 → 텍스트
# -------------------------------
def transcribe_audio(filename):
    sr, data = wavfile.read(filename)
    if data.ndim > 1:
        data = data[:, 0]  # 모노 처리

    input_values = processor(data.astype(np.float32), return_tensors="pt", sampling_rate=sr).input_values

    with torch.no_grad():
        logits = wav2vec_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription


# -------------------------------
# 문장 교정
# -------------------------------
def correct_sentence(text):
    inputs = grammar_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = grammar_model.generate(inputs.input_ids, max_length=128)
    corrected = grammar_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎙️ 한국어 음성 인식 + 문장 교정 (Wav2Vec2 + KoGrammar)")

st.markdown("음성을 입력하면 텍스트로 변환하고, 문법을 자동 교정해드립니다.")

duration = st.slider("녹음 시간 (초)", 3, 10, 5)

if st.button("🎤 녹음 시작"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        wav_path = tmpfile.name
    filename = record_audio(filename=wav_path, duration=duration)

    st.audio(filename, format='audio/wav')

    # 1. 음성 인식
    raw_text = transcribe_audio(filename)
    st.subheader("📝 인식된 문장:")
    st.write(raw_text)

    # 2. 문장 교정
    corrected_text = correct_sentence(raw_text)
    st.subheader("✅ 교정된 문장:")
    st.success(corrected_text)