#!/usr/bin/env python3
"""Lightweight test harness that runs generate_from_file with a mock engine
to validate chunking, streaming and stitching behavior with Vietnamese input.

This does NOT call the real HiggsAudio model; it simulates generation by
producing synthetic audio sized to match the estimate_speech_seconds_from_text
so we can validate the pipeline end-to-end quickly.
"""
import os
import numpy as np
import soundfile as sf
import importlib.util
import sys

# Load the generate_long_audio module by path so this test can run without
# requiring repository packaging/import setup.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
gen_path = os.path.join(repo_root, 'scripts', 'generate_long_audio.py')
spec = importlib.util.spec_from_file_location('generate_long_audio', gen_path)
gen = importlib.util.module_from_spec(spec)
sys.modules['generate_long_audio'] = gen
spec.loader.exec_module(gen)

from generate_long_audio import generate_from_file, estimate_speech_seconds_from_text
from boson_multimodal.data_types import Message, ChatMLSample


class MockResponse:
    def __init__(self, audio: np.ndarray, sampling_rate: int):
        self.audio = audio
        self.sampling_rate = sampling_rate


class MockEngine:
    def __init__(self, *args, **kwargs):
        self.sr = 24000

    def generate(self, chat_ml_sample: ChatMLSample, **kwargs):
        # Inspect last user message to size the synthetic audio
        user_msgs = [m for m in chat_ml_sample.messages if getattr(m, 'role', '') == 'user']
        text = user_msgs[-1].content if user_msgs else ''
        est_sec = estimate_speech_seconds_from_text(text)
        # produce audio slightly longer than estimate to pass duration checks
        dur = max(0.5, est_sec * 1.05)
        samples = int(dur * self.sr)
        # simple sine wave
        t = np.linspace(0, dur, samples, endpoint=False)
        audio = 0.05 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        return MockResponse(audio, self.sr)


def mock_factory(mpath, tpath, device):
    return MockEngine()


if __name__ == '__main__':
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    input_text = os.path.join(repo_root, 'PhuongHang.txt')
    output_wav = os.path.join(repo_root, 'output', 'test_vi_mock.wav')
    ref_audio = os.path.join(repo_root, 'input', 'ref.mp3')

    os.makedirs(os.path.dirname(output_wav), exist_ok=True)

    print('Running mock Vietnamese flow test...')
    generate_from_file('dummy_model', 'dummy_tokenizer', input_text, output_wav,
                       device='cpu', system_prompt='Generate Vietnamese speech.',
                       reference_audio=ref_audio, reference_text='Đoạn mẫu giọng đọc.',
                       engine_factory=mock_factory, stream=True, tmp_base=os.path.join(repo_root, 'scripts', 'tmp_chunks_test'))

    if os.path.exists(output_wav):
        info = sf.info(output_wav)
        print(f'Test output written: {output_wav} ({info.frames} frames @ {info.samplerate} Hz)')
    else:
        raise SystemExit('Test failed: output file not created')
