#!/usr/bin/env python3
"""
Small helper to convert a long text file into generated audio using the
HiggsAudioServeEngine. It implements chunked generation and stitches audio with a
short crossfade to reduce audible seams.

Usage:
    python scripts/generate_long_audio.py /path/to/text.txt /path/to/output.wav [--reference-audio ref.wav] [--reference-text "Text matching the reference audio"]
"""

import sys
import os
import argparse
import json
import numpy as np
import soundfile as sf
import torchaudio
from typing import List, Dict, Optional

# Ensure the custom node package is importable.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CUSTOM_NODE_PATH = os.path.join(REPO_ROOT, 'custom_nodes', 'ComfyUI-HiggsAudio_Wrapper')
if CUSTOM_NODE_PATH not in sys.path:
    sys.path.insert(0, CUSTOM_NODE_PATH)

# Try to import the Boson engine types for convenience, but allow the module to be
# imported in test contexts where the real package is not present. The real
# engine is imported lazily inside `generate_from_file` when needed.
HAS_BOSON = True
try:
    from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
except Exception:
    HAS_BOSON = False

# Conservative defaults. Tune these for your Mac Studio.
CHUNK_CHAR_SIZE = 2000            # reduced chunk size for better voice consistency
CHUNK_MAX_TOKENS = 2048          # reduced token budget to maintain voice quality
CROSSFADE_MS = 120               # increased crossfade for smoother transitions
SAMPLE_RATE = 24000              # default sample rate; HiggsAudio model card says 24 kHz
MIN_CHUNK_CHARS = 500            # minimum chunk size to avoid tiny fragments
TRIM_START_SEC = 0.5             # trim this many seconds from the very start of final output to remove artifacts


def split_text_into_chunks(text: str, max_chars: int) -> List[str]:
    """Split text into chunks at paragraph or sentence boundaries when possible.
    This is a heuristic: we prefer to split at double-newline (paragraphs),
    otherwise at sentence-ending punctuation, and finally just cut at max_chars.
    Maintains minimum chunk size to avoid voice inconsistency.
    """
    import re
    
    # First, split into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    
    def clean_chunk(s: str) -> str:
        """Clean up a chunk to ensure it starts and ends cleanly."""
        s = s.strip()
        # Ensure chunk ends with proper punctuation
        if not s[-1] in '.!?':
            s = s + '.'
        return s
    
    # Process each paragraph
    current_chunk = []
    current_len = 0
    
    for p_idx, p in enumerate(paragraphs):
        # Split paragraph into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p) if s.strip()]
        
        for s_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence[-1] in '.!?':
                sentence = sentence + '.'
            
            # If adding this sentence would exceed max_chars
            if current_len + len(sentence) + 1 > max_chars and current_len >= MIN_CHUNK_CHARS:
                # Store current chunk and start new one
                chunks.append(clean_chunk(' '.join(current_chunk)))
                current_chunk = []
                current_len = 0
            
            # Handle very long sentences
            if len(sentence) > max_chars:
                # First save any accumulated content if we have enough
                if current_chunk and current_len >= MIN_CHUNK_CHARS:
                    chunks.append(clean_chunk(' '.join(current_chunk)))
                    current_chunk = []
                    current_len = 0
                
                # Split long sentence at reasonable points (commas, conjunctions)
                splits = re.split(r'(?<=[,;])\s+|(?<=\band\b)\s+|(?<=\bor\b)\s+|(?<=\bbut\b)\s+', sentence)
                temp_chunk = []
                temp_len = 0
                
                for split in splits:
                    if temp_len + len(split) + 1 <= max_chars:
                        temp_chunk.append(split)
                        temp_len += len(split) + 1
                    else:
                        if temp_chunk:
                            chunks.append(clean_chunk(' '.join(temp_chunk)))
                        temp_chunk = [split]
                        temp_len = len(split)
                
                if temp_chunk:
                    chunks.append(clean_chunk(' '.join(temp_chunk)))
            else:
                current_chunk.append(sentence)
                current_len += len(sentence) + 1
            
            # If this is the last sentence of a paragraph and we have content
            if s_idx == len(sentences) - 1 and p_idx < len(paragraphs) - 1:
                if current_len >= MIN_CHUNK_CHARS:
                    chunks.append(clean_chunk(' '.join(current_chunk)))
                    current_chunk = []
                    current_len = 0
    
    # Don't forget any remaining content
    if current_chunk:
        chunks.append(clean_chunk(' '.join(current_chunk)))
    
    return chunks


def crossfade_concat(a: np.ndarray, b: np.ndarray, sr: int, crossfade_ms: int) -> np.ndarray:
    overlap = int(sr * crossfade_ms / 1000.0)
    if overlap <= 0:
        return np.concatenate([a, b], axis=0)
    if overlap > len(a) or overlap > len(b):
        # fallback: no crossfade
        return np.concatenate([a, b], axis=0)
    # linear crossfade
    fade_out = np.linspace(1.0, 0.0, overlap)
    fade_in = np.linspace(0.0, 1.0, overlap)
    out = np.copy(a[:-overlap])
    middle = a[-overlap:] * fade_out + b[:overlap] * fade_in
    out = np.concatenate([out, middle, b[overlap:]], axis=0)
    return out


def estimate_speech_seconds_from_text(text: str) -> float:
    """Estimate spoken seconds for the given text using a conservative speech rate.

    Assumes ~150 words per minute (~2.5 words/sec). This is only a heuristic used
    to detect obviously-short generated audio and trigger a retry with higher
    token budget / less-restrictive stopping rules.
    """
    words = len([w for w in text.split() if w.strip()])
    if words == 0:
        return 0.0
    return words / 2.5


def load_reference_audio(audio_path: str) -> Dict:
    """Load reference audio file and convert to ComfyUI format."""
    import torch
    data, sample_rate = sf.read(audio_path, dtype='float32')
    # Convert to tensor and reshape to [batch, channel, time]
    if len(data.shape) == 1:
        waveform = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    else:
        waveform = torch.from_numpy(data.T).unsqueeze(0)  # Transpose for channel-first
    return {
        "waveform": waveform.float(),
        "sample_rate": sample_rate
    }


def add_voice_primer(text: str) -> str:
    """Add a short primer to help establish the voice at the start.

    The primer is localized when Vietnamese is detected so the model is not
    biased by an English prefix which can cause short/blank outputs for non-
    English inputs.
    """
    # simple Vietnamese detection: look for common Vietnamese diacritics/words
    def looks_like_vietnamese(s: str) -> bool:
        vi_chars = set('àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ')
        # quick scan first 200 chars
        sample = s[:200].lower()
        for ch in sample:
            if ch in vi_chars:
                return True
        # fallback: check for some common Vietnamese words
        for w in ('và', 'nhưng', 'không', 'anh', 'chị', 'tôi', 'của'):
            if f' {w} ' in f' {sample} ':
                return True
        return False

    if looks_like_vietnamese(text):
        primer = "Bắt đầu đọc. "
    else:
        primer = "Reading begins. "
    return primer + text


def generate_from_file(model_path: str, tokenizer_path: str, input_file: str, out_file: str,
                       device: str = 'cpu', system_prompt: str = None,
                       reference_audio: Optional[str] = None,
                       reference_text: Optional[str] = None,
                       *, engine_factory=None, stream: bool = False, tmp_base: str = None):
    # load text
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Add voice primer
    text = add_voice_primer(text)

    chunks = split_text_into_chunks(text, CHUNK_CHAR_SIZE)
    print(f"Split input into {len(chunks)} chunk(s) (approx {CHUNK_CHAR_SIZE} chars per chunk)")

    # prepare engine (allow injection for unit tests)
    if engine_factory is None:
        # lazy import to allow tests to inject a mock factory
        try:
            from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        except Exception as e:
            raise ImportError("Failed to import HiggsAudioServeEngine from the local custom node."
                              " Make sure you run this script from the repo root and the submodule is present.")

        def _default_factory(mpath, tpath, device):
            return HiggsAudioServeEngine(mpath, tpath, device=device)

        engine_factory = _default_factory

    engine = engine_factory(model_path, tokenizer_path, device)

    # Load reference audio if provided
    reference = None
    if reference_audio:
        reference = load_reference_audio(reference_audio)
        print(f"Loaded reference audio: {reference_audio}")

    # prepare streaming/temp paths if requested
    if tmp_base is None:
        tmp_base = os.path.join(REPO_ROOT, 'scripts', 'tmp_chunks')
    os.makedirs(tmp_base, exist_ok=True)

    # job id: deterministic-ish from input filename + timestamp
    import hashlib, time
    job_id = hashlib.sha1((os.path.abspath(input_file) + str(os.path.getmtime(input_file))).encode()).hexdigest()[:12]
    job_dir = os.path.join(tmp_base, job_id)
    os.makedirs(job_dir, exist_ok=True)
    manifest_path = os.path.join(job_dir, 'manifest.json')

    # build or load manifest
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as mf:
            manifest = json.load(mf)
        print(f"Resuming job {job_id}: found manifest with {len(manifest.get('chunks', []))} chunks")
        # allow resumed run to pick up chunks list from manifest
        if 'chunks' in manifest and len(manifest['chunks']) > 0:
            # trust manifest chunk list
            disk_chunks = manifest['chunks']
        else:
            disk_chunks = None
    else:
        manifest = {
            'job_id': job_id,
            'input': os.path.abspath(input_file),
            'output': os.path.abspath(out_file),
            'chunks': [],
            'params': {
                'chunk_char_size': CHUNK_CHAR_SIZE,
                'chunk_max_tokens': CHUNK_MAX_TOKENS,
                'crossfade_ms': CROSSFADE_MS,
                'sample_rate': SAMPLE_RATE,
            }
        }
        disk_chunks = None

    sr = SAMPLE_RATE

    # if we don't have chunk list from manifest, create it
    if disk_chunks is None:
        for idx, chunk in enumerate(chunks):
            chunk_file = os.path.join(job_dir, f'chunk_{idx:04d}.wav')
            manifest['chunks'].append({'idx': idx, 'chars': len(chunk), 'file': chunk_file, 'done': False})
        with open(manifest_path, 'w', encoding='utf-8') as mf:
            json.dump(manifest, mf, indent=2)
    else:
        # if manifest defines chunks, we still need chunks variable mapped to content for generation
        # rebuild chunks from input text using same splitter
        # (we assume the original splitting params are unchanged)
        # reuse local chunks variable
        pass

    # Main generation loop: generate per-chunk and either accumulate or stream to disk
    for meta in manifest['chunks']:
        idx = meta['idx']
        if meta.get('done', False):
            print(f"Skipping already-generated chunk {idx}")
            continue

        chunk = chunks[idx]
        print(f"Generating chunk {idx+1}/{len(manifest['chunks'])} (chars={len(chunk)})")

        # Build messages: keep a short system prompt and then the user text chunk
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append(Message(role='system', content=system_prompt))

        # Add reference audio for voice cloning if available
        if reference:
            if reference_text:
                messages.append(Message(role='system', content=reference_text))
            else:
                messages.append(Message(role='system', content='Reference audio for voice cloning.'))

            # Convert audio to base64 for message
            import base64
            import io
            buffer = io.BytesIO()
            sf.write(buffer, reference['waveform'][0, 0].numpy(), reference['sample_rate'], format='WAV')
            buffer.seek(0)
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

            # Add assistant message with audio content
            audio_content = AudioContent(raw_audio=audio_base64, audio_url='')
            messages.append(Message(role='assistant', content=[audio_content]))

        messages.append(Message(role='user', content=chunk))

        # Call generate with retry logic to avoid truncated / skipped chunks.
        attempts = 0
        max_attempts = 3
        tokens = CHUNK_MAX_TOKENS
        a = None
        expected_sec = estimate_speech_seconds_from_text(chunk)
        while attempts < max_attempts:
            attempts += 1
            stop_strings = ["<|end_of_text|>", "<|eot_id|>"] if attempts == 1 else None
            print(f"Attempt {attempts} for chunk {idx}: tokens={tokens}, stop_strings={'yes' if stop_strings else 'no'}")
            resp = engine.generate(
                chat_ml_sample=ChatMLSample(messages=messages),
                max_new_tokens=tokens,
                temperature=0.2,  # Reduced temperature for more stable output
                top_p=0.85,       # More restrictive sampling for consistency
                top_k=40,         # Slightly reduced for more focused sampling
                stop_strings=stop_strings
            )

            if not hasattr(resp, 'audio') or resp.audio is None:
                print(f"Chunk {idx} returned no audio on attempt {attempts}")
                if attempts >= max_attempts:
                    raise RuntimeError(f"Chunk {idx} returned no audio")
                tokens = tokens * 2
                continue

            # ensure numpy float32 1D array
            a = resp.audio.astype(np.float32)
            sr = getattr(resp, 'sampling_rate', SAMPLE_RATE)

            # Validate duration vs expected; if obviously too short, retry with larger token budget
            duration = float(len(a)) / float(sr) if sr > 0 else 0.0
            if expected_sec > 0 and duration < expected_sec * 0.6 and attempts < max_attempts:
                print(f"Chunk {idx} audio too short ({duration:.2f}s vs expected {expected_sec:.2f}s). Retrying with higher token budget.")
                tokens = tokens * 2
                continue

            # accept output
            break

        # streaming: write chunk to disk and update manifest
        chunk_file = meta['file']
        sf.write(chunk_file, a, sr)
        meta['done'] = True
        with open(manifest_path, 'w', encoding='utf-8') as mf:
            json.dump(manifest, mf, indent=2)
        print(f"Wrote chunk {idx} to {chunk_file}")

    # stitching phase: read chunks from disk sequentially and write final output incrementally
    print("Stitching chunks from disk...")
    # open output file for streaming write
    with sf.SoundFile(out_file, mode='w', samplerate=sr, channels=1, subtype='PCM_16') as out_f:
        prev_tail = None
        for meta in manifest['chunks']:
            chunk_file = meta['file']
            if not os.path.exists(chunk_file):
                raise RuntimeError(f"Missing chunk file during stitch: {chunk_file}")
            data, _ = sf.read(chunk_file, dtype='float32')
            data = np.asarray(data, dtype=np.float32)
            tail_len = int(sr * CROSSFADE_MS / 1000.0)
            if prev_tail is None:
                # write first chunk *without* its trailing overlap (we keep it in memory)
                trim_samples = int(sr * TRIM_START_SEC) if TRIM_START_SEC > 0 else 0
                # Avoid over-trimming
                if trim_samples >= len(data):
                    # nothing to write; set prev_tail to empty and continue
                    prev_tail = np.zeros(0, dtype=np.float32)
                    continue

                if tail_len > 0 and len(data) > (tail_len + trim_samples):
                    out_f.write(data[trim_samples:-tail_len])
                    prev_tail = data[-tail_len:]
                else:
                    out_f.write(data[trim_samples:])
                    prev_tail = data
            else:
                # perform crossfade between prev_tail and head of current data
                head = data[:tail_len] if len(data) >= tail_len else data
                if tail_len > 0 and len(head) == tail_len:
                    fade_out = np.linspace(1.0, 0.0, tail_len)
                    fade_in = np.linspace(0.0, 1.0, tail_len)
                    middle = prev_tail * fade_out + head * fade_in
                    # Seek backwards by tail_len frames and overwrite the trailing overlap
                    try:
                        # move write pointer to overwrite previous tail
                        out_f.seek(max(0, out_f.frames - tail_len))
                        out_f.write(middle)
                        out_f.write(data[tail_len:])
                    except Exception:
                        # if seeking is not supported, fall back to append (may duplicate overlap)
                        out_f.write(middle)
                        out_f.write(data[tail_len:])
                else:
                    # fallback: just write current data
                    out_f.write(data)
                # update prev_tail to current chunk's trailing overlap
                prev_tail = data[-tail_len:] if tail_len > 0 and len(data) >= tail_len else data
    print(f"Saved stitched output to {out_file}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('input', help='Input text file')
    p.add_argument('output', help='Output WAV path')
    p.add_argument('--model', default='bosonai/higgs-audio-v2-generation-3B-base')
    p.add_argument('--tokenizer', default='bosonai/higgs-audio-v2-tokenizer')
    p.add_argument('--device', default='cpu')
    p.add_argument('--system-prompt', default='Generate audio following instruction.')
    p.add_argument('--reference-audio', help='Path to reference audio file for voice cloning')
    p.add_argument('--reference-text', help='Text corresponding to the reference audio')
    args = p.parse_args()

    generate_from_file(args.model, args.tokenizer, args.input, args.output,
                      device=args.device, system_prompt=args.system_prompt,
                      reference_audio=args.reference_audio,
                      reference_text=args.reference_text)
