import os, json, uuid, re, time
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

# ========= CONFIG CHUNG =========
# True: lưu voice & history lên Google Drive (giữ được giữa các lần reset Colab)
# False: lưu local trong runtime Colab (reset là mất)
USE_GOOGLE_DRIVE = True

# Phát hiện đang chạy trong Colab hay không
try:
    import google.colab  # type: ignore
    IN_COLAB = True
except ImportError:
    IN_COLAL = False
    IN_COLAB = False  # phòng khi typo, nhưng giữ IN_COLAB để dùng dưới

BASE_DIR = ""

if USE_GOOGLE_DRIVE and IN_COLAB:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/neutts_vi_studio"
else:
    if IN_COLAB:
        BASE_DIR = "/content/tts_app_data"
    else:
        HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
        BASE_DIR = os.path.join(HERE, "tts_app_data")

os.makedirs(BASE_DIR, exist_ok=True)

VOICES_DIR = os.path.join(BASE_DIR, "voices")
HISTORY_DIR = os.path.join(BASE_DIR, "history_audio")
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

VOICES_JSON = os.path.join(BASE_DIR, "voices.json")
HISTORY_JSON = os.path.join(BASE_DIR, "history.json")

# ========= IMPORT LIB =========
import numpy as np
import librosa
import soundfile as sf
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from neucodec import NeuCodec
from phonemizer.backend import EspeakBackend
from vinorm import TTSnorm
import gradio as gr
from resemblyzer import VoiceEncoder, preprocess_wav

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Encoder để đo độ giống giọng (speaker similarity)
speaker_encoder = VoiceEncoder()

# ========= MODEL & CODEC =========
MODEL_ID = "dinhthuan/neutts-air-vi"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    trust_remote_code=True,
).to(device)
model.eval()

codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(device)
codec.eval()

# ========= PHONEMIZER & NORMALIZER =========
phonemizer = EspeakBackend(language='vi', preserve_punctuation=True, with_stress=True)

def vn_norm(text: str) -> str:
    return TTSnorm(text, punc=False, unknown=True, lower=False, rule=False)

# ========= PRESETS =========
PRESETS = {
    "Neutral": {
        "deepen": 0,
        "stronger": 0,
        "nasal": 0,
        "speed": 1.0,
        "pitch": 0.0,
        "volume": 1.0,
        "spacious_echo": False,
        "auditorium_echo": False,
        "lofi_tel": False,
        "robotic": False,
    },
    "Podcast": {
        "deepen": 15,
        "stronger": 10,
        "nasal": -10,
        "speed": 1.0,
        "pitch": -1.0,
        "volume": 1.2,
        "spacious_echo": True,
        "auditorium_echo": False,
        "lofi_tel": False,
        "robotic": False,
    },
    "Audiobook": {
        "deepen": 10,
        "stronger": 5,
        "nasal": -5,
        "speed": 0.95,
        "pitch": -0.5,
        "volume": 1.0,
        "spacious_echo": False,
        "auditorium_echo": True,
        "lofi_tel": False,
        "robotic": False,
    },
    "Call Center": {
        "deepen": -5,
        "stronger": 15,
        "nasal": 10,
        "speed": 1.05,
        "pitch": 1.0,
        "volume": 1.1,
        "spacious_echo": False,
        "auditorium_echo": False,
        "lofi_tel": True,
        "robotic": False,
    },
}

PRESETS_PATH = os.path.join(BASE_DIR, "presets.json")

# ========= DATA CLASSES & DB =========
@dataclass
class VoiceEntry:
    voice_id: str
    name: str
    language: str
    ref_audio_path: str
    ref_text: str
    ref_phones: str
    ref_codes: List[int]
    created_at: float

@dataclass
class HistoryEntry:
    history_id: str
    voice_id: str
    voice_name: str
    text: str
    language: str
    audio_path: str
    created_at: float
    modifiers: Dict[str, Any]

def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

_voices_db: Dict[str, VoiceEntry] = {}
_history_db: Dict[str, HistoryEntry] = {}

def reload_dbs():
    global _voices_db, _history_db
    v_raw = load_json(VOICES_JSON, [])
    h_raw = load_json(HISTORY_JSON, [])
    _voices_db = {v["voice_id"]: VoiceEntry(**v) for v in v_raw}
    _history_db = {h["history_id"]: HistoryEntry(**h) for h in h_raw}

def persist_dbs():
    save_json(VOICES_JSON, [asdict(v) for v in _voices_db.values()])
    save_json(HISTORY_JSON, [asdict(h) for h in _history_db.values()])

def load_presets_from_disk():
    global PRESETS
    data = load_json(PRESETS_PATH, None)
    if isinstance(data, dict) and data:
        PRESETS = data
        print(f"Loaded presets from {PRESETS_PATH}")
    else:
        print("Using default PRESETS (no presets.json found or invalid).")

def save_presets_to_disk():
    save_json(PRESETS_PATH, PRESETS)
    print(f"Saved presets to {PRESETS_PATH}")

reload_dbs()
load_presets_from_disk()

# ========= TEXT → PHONES =========
def text_to_phones(text: str, language: str = "vi") -> str:
    if language in ["vi", "auto", "vietnamese"]:
        txt_norm = vn_norm(text)
        phones = phonemizer.phonemize([txt_norm])[0]
        return phones
    else:
        phones = phonemizer.phonemize([text])[0]
        return phones

# ========= AUDIO UTILS =========
def load_audio_mono16k(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    wav, sr = librosa.load(path, sr=target_sr, mono=True)
    return wav, target_sr

def compute_similarity(ref_path: str, demo_path: str) -> float:
    """
    Tính cosine similarity giữa giọng reference và demo.
    Trả về [0.0, 1.0], -1 nếu lỗi.
    """
    try:
        ref_wav = preprocess_wav(ref_path)
        demo_wav = preprocess_wav(demo_path)
        ref_emb = speaker_encoder.embed_utterance(ref_wav)
        demo_emb = speaker_encoder.embed_utterance(demo_wav)
        sim = float(np.dot(ref_emb, demo_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(demo_emb)))
        sim01 = (sim + 1.0) / 2.0
        return max(0.0, min(1.0, sim01))
    except Exception as e:
        print("Similarity error:", e)
        return -1.0

def encode_ref_audio_to_codes(audio_path: str) -> List[int]:
    wav, sr = load_audio_mono16k(audio_path, 16000)
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        ref_codes = codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0).cpu().numpy().tolist()
    return ref_codes

# ========= TTS CORE =========
SPEECH_END_TOKEN = "<|SPEECH_GENERATION_END|>"

def build_chat_prompt(ref_phones: str, target_phones: str, ref_codes: List[int]) -> str:
    codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
    combined_phones = (ref_phones + " " + target_phones).strip()
    chat = (
        "user: Convert the text to speech:"
        f"<|TEXT_PROMPT_START|>{combined_phones}<|TEXT_PROMPT_END|>\n"
        f"assistant:<|SPEECH_GENERATION_START|>{codes_str}"
    )
    return chat

def extract_speech_codes_from_output(output_text: str) -> List[int]:
    pattern = r"<\|speech_(\d+)\|>"
    codes = [int(m) for m in re.findall(pattern, output_text)]
    return codes

def decode_codes_to_audio(codes: List[int], sr: int = 24000) -> np.ndarray:
    codes_tensor = torch.tensor(codes, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        audio = codec.decode_code(codes_tensor, sample_rate=sr)[0, 0].cpu().numpy()
    return audio

def generate_tts_raw(
    text: str,
    voice: VoiceEntry,
    language: str = "vi",
    max_new_tokens: int = 2048,
) -> np.ndarray:
    target_phones = text_to_phones(text, language=language)
    chat = build_chat_prompt(voice.ref_phones, target_phones, voice.ref_codes)

    input_ids = tokenizer.encode(chat, return_tensors="pt").to(device)
    speech_end_id = tokenizer.convert_tokens_to_ids(SPEECH_END_TOKEN)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=50,
            eos_token_id=speech_end_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    codes = extract_speech_codes_from_output(output_text)
    audio = decode_codes_to_audio(codes, sr=24000)
    return audio

# ========= AUDIO FX =========
def apply_speed(audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
    if abs(speed - 1.0) < 1e-3:
        return audio
    return librosa.effects.time_stretch(audio, rate=speed)

def apply_pitch(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    if abs(semitones) < 1e-3:
        return audio
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)

def apply_volume(audio: np.ndarray, gain: float) -> np.ndarray:
    return audio * gain

def apply_deepen_stronger_nasal(audio: np.ndarray, sr: int, deepen: int, stronger: int, nasal: int) -> np.ndarray:
    semitones = (deepen / 50.0) * 6.0
    audio = apply_pitch(audio, sr, semitones)
    if abs(stronger) > 1e-3:
        factor = 1.0 + (stronger / 100.0)
        audio = np.tanh(audio * factor)
    if abs(nasal) > 1e-3:
        S = librosa.stft(audio)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=S.shape[0]*2-2)
        boost = 1.0 + (nasal / 100.0) * 0.8
        band = (freqs >= 1000) & (freqs <= 3000)
        S[band, :] *= boost
        audio = librosa.istft(S)
    return audio

def apply_echo(audio: np.ndarray, sr: int, delay_s: float = 0.25, decay: float = 0.4) -> np.ndarray:
    delay_samples = int(delay_s * sr)
    echo = np.zeros_like(audio)
    if delay_samples < len(audio):
        echo[delay_samples:] = audio[:-delay_samples] * decay
    return audio + echo

def apply_reverb_like(audio: np.ndarray, sr: int, decay: float = 0.5, repeats: int = 4, base_delay: float = 0.06):
    out = audio.copy()
    for i in range(1, repeats+1):
        out += apply_echo(audio, sr, delay_s=base_delay*i, decay=decay/(i+1))
    return out

def apply_telephone(audio: np.ndarray, sr: int) -> np.ndarray:
    S = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1.0/sr)
    mask = (freqs >= 300) & (freqs <= 3400)
    S[~mask] = 0
    audio_bp = np.fft.irfft(S)
    return audio_bp

def apply_robotic(audio: np.ndarray, sr: int) -> np.ndarray:
    t = np.arange(len(audio)) / sr
    mod = np.sign(np.sin(2 * np.pi * 50 * t))
    return audio * mod

def apply_fx(
    audio: np.ndarray,
    sr: int,
    spacious: bool,
    auditorium: bool,
    lofi_tel: bool,
    robotic_fx: bool,
) -> np.ndarray:
    if spacious:
        audio = apply_reverb_like(audio, sr, decay=0.6, repeats=3, base_delay=0.08)
    if auditorium:
        audio = apply_reverb_like(audio, sr, decay=0.8, repeats=5, base_delay=0.12)
    if lofi_tel:
        audio = apply_telephone(audio, sr)
    if robotic_fx:
        audio = apply_robotic(audio, sr)
    return audio

def process_audio_with_modifiers(
    audio: np.ndarray,
    sr: int,
    deepen: int,
    stronger: int,
    nasal: int,
    speed: float,
    pitch_semitones: float,
    volume: float,
    spacious: bool,
    auditorium: bool,
    lofi_tel: bool,
    robotic_fx: bool,
) -> np.ndarray:
    audio = apply_deepen_stronger_nasal(audio, sr, deepen, stronger, nasal)
    audio = apply_speed(audio, sr, speed)
    audio = apply_pitch(audio, sr, pitch_semitones)
    audio = apply_volume(audio, volume)
    audio = apply_fx(audio, sr, spacious, auditorium, lofi_tel, robotic_fx)
    peak = np.max(np.abs(audio)) + 1e-8
    if peak > 1.0:
        audio = audio / peak
    return audio

# ========= PAUSE TAGS <#0.5#> =========
PAUSE_PATTERN = re.compile(r"<#(\d+(\.\d+)?)#>")

def synthesize_with_pauses(
    text: str,
    voice: VoiceEntry,
    language: str,
    modifiers: Dict[str, Any],
    sr: int = 24000,
) -> np.ndarray:
    parts = PAUSE_PATTERN.split(text)
    final_audio = []
    for i in range(0, len(parts), 3):
        seg_text = parts[i].strip()
        if seg_text:
            raw = generate_tts_raw(seg_text, voice, language=language)
            proc = process_audio_with_modifiers(
                raw, sr,
                deepen=modifiers["deepen"],
                stronger=modifiers["stronger"],
                nasal=modifiers["nasal"],
                speed=modifiers["speed"],
                pitch_semitones=modifiers["pitch"],
                volume=modifiers["volume"],
                spacious=modifiers["spacious_echo"],
                auditorium=modifiers["auditorium_echo"],
                lofi_tel=modifiers["lofi_tel"],
                robotic_fx=modifiers["robotic"],
            )
            final_audio.append(proc)
        if i+1 < len(parts):
            pause_s = float(parts[i+1])
            n_samples = int(pause_s * sr)
            final_audio.append(np.zeros(n_samples, dtype=np.float32))
    if not final_audio:
        return np.zeros(1, dtype=np.float32)
    audio_concat = np.concatenate(final_audio)
    peak = np.max(np.abs(audio_concat)) + 1e-8
    if peak > 1.0:
        audio_concat = audio_concat / peak
    return audio_concat

# ========= VOICE CLONE & HISTORY =========
def create_voice_preview(
    audio_path: str,
    language: str,
    voice_name: str,
    preview_text: str,
) -> Tuple[str, str, float]:
    """
    Trả về (temp_voice_id, preview_audio_path, similarity_score)
    similarity_score: 0.0–1.0, -1 nếu lỗi.
    """
    if not preview_text.strip():
        preview_text = "Xin chào, đây là giọng nói vừa được clone."
    ref_codes = encode_ref_audio_to_codes(audio_path)
    ref_phones = text_to_phones(preview_text, language=language)
    voice_id = f"temp-{uuid.uuid4().hex[:8]}"
    tmp_voice = VoiceEntry(
        voice_id=voice_id,
        name=voice_name or f"Voice {voice_id}",
        language=language,
        ref_audio_path=audio_path,
        ref_text=preview_text,
        ref_phones=ref_phones,
        ref_codes=ref_codes,
        created_at=time.time(),
    )
    audio_demo = generate_tts_raw(preview_text, tmp_voice, language=language)
    preview_path = os.path.join(VOICES_DIR, f"{voice_id}_preview.wav")
    sf.write(preview_path, audio_demo, 24000)

    _voices_db[voice_id] = tmp_voice

    sim = compute_similarity(audio_path, preview_path)
    return voice_id, preview_path, sim

def finalize_voice(temp_voice_id: str, final_name: str) -> Tuple[bool, str]:
    if temp_voice_id not in _voices_db:
        return False, "Không tìm thấy voice tạm để lưu."
    v = _voices_db[temp_voice_id]
    v.name = final_name or v.name
    new_id = uuid.uuid4().hex
    v.voice_id = new_id
    old_preview = os.path.join(VOICES_DIR, f"{temp_voice_id}_preview.wav")
    new_preview = os.path.join(VOICES_DIR, f"{new_id}_preview.wav")
    if os.path.exists(old_preview):
        os.rename(old_preview, new_preview)
    _voices_db.pop(temp_voice_id)
    _voices_db[new_id] = v
    persist_dbs()
    return True, new_id

def list_voices_for_ui() -> List[Tuple[str, str]]:
    return [(v.name, v.voice_id) for v in _voices_db.values()]

def delete_voice(voice_id: str) -> str:
    v = _voices_db.pop(voice_id, None)
    if v:
        prev = os.path.join(VOICES_DIR, f"{voice_id}_preview.wav")
        if os.path.exists(prev):
            os.remove(prev)
        persist_dbs()
        return f"Đã xoá voice '{v.name}'."
    return "Voice không tồn tại."

def save_history(
    voice: VoiceEntry,
    text: str,
    language: str,
    audio: np.ndarray,
    modifiers: Dict[str, Any],
) -> str:
    history_id = uuid.uuid4().hex
    audio_path = os.path.join(HISTORY_DIR, f"{history_id}.wav")
    sf.write(audio_path, audio, 24000)
    entry = HistoryEntry(
        history_id=history_id,
        voice_id=voice.voice_id,
        voice_name=voice.name,
        text=text,
        language=language,
        audio_path=audio_path,
        created_at=time.time(),
        modifiers=modifiers,
    )
    _history_db[history_id] = entry
    persist_dbs()
    return audio_path

def list_history_for_ui() -> List[Tuple[str, str]]:
    items = []
    for h in sorted(_history_db.values(), key=lambda x: x.created_at, reverse=True):
        short_text = (h.text[:30] + "...") if len(h.text) > 30 else h.text
        label = time.strftime("%Y-%m-%d %H:%M", time.localtime(h.created_at)) + f" – {h.voice_name} – {short_text}"
        items.append((label, h.history_id))
    return items

def get_history_audio(history_id: str) -> Tuple[str, str]:
    h = _history_db.get(history_id)
    if not h:
        return "", ""
    return h.audio_path, h.text

# ========= GRADIO UI =========
custom_css = """
body { background-color: #f9fafb; }
.gradio-container { max-width: 1200px !important; margin: 0 auto; }
h1, h2, h3, h4 { color: #0f172a; }
label, p { color: #111827; }
button { border-radius: 9999px !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=custom_css) as demo:
    gr.Markdown(
        """
        # 🔊 NeuTTS Vietnamese Voice Studio  
        Clone giọng & Text-to-Speech tiếng Việt (NeuTTS-Air + NeuCodec)

        > **Lưu ý**: Chỉ clone giọng khi đã có sự đồng ý của chủ giọng. Không sử dụng để giả mạo hay lừa đảo.
        """
    )

    voices_state = gr.State(value=None)
    temp_voice_id_state = gr.State(value="")

    def refresh_voices_state():
        reload_dbs()
        return list_voices_for_ui()

    # ----- TAB: Voice Clone -----
    with gr.Tab("🎙️ Voice Clone"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1️⃣ Tải file audio mẫu (≥ 10s)")
                in_audio = gr.Audio(
                    label="Reference Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                    interactive=True,
                )
                gr.Markdown("Sau khi upload, dùng player bên trên để nghe lại kiểm tra.")

                gr.Markdown("### 2️⃣ Thiết lập thông tin voice")
                lang_clone = gr.Dropdown(
                    choices=["vietnamese", "auto"],
                    value="vietnamese",
                    label="Language",
                )
                voice_name_input = gr.Textbox(
                    label="Tên voice",
                    placeholder="VD: Giọng Nam Bắc trầm",
                )
                preview_text_input = gr.Textbox(
                    label="Text dùng để tạo audio demo",
                    placeholder="Xin chào, đây là giọng nói vừa được clone.",
                    lines=2,
                )
                btn_gen_preview = gr.Button("Tạo demo voice 🎧", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 3️⃣ Nghe thử, xem độ giống & lưu voice")
                preview_audio = gr.Audio(
                    label="Audio demo của voice clone",
                    type="filepath",
                )
                similarity_text = gr.Markdown("")
                save_voice_name = gr.Textbox(
                    label="Tên voice sau khi lưu (có thể chỉnh lại)",
                )
                btn_save_voice = gr.Button("Lưu voice vào Voice Library 💾")
                clone_status = gr.Markdown("")

        def on_gen_preview(audio_path, lang, vname, ptext):
            if not audio_path:
                return "", "", "⚠️ Vui lòng upload file audio ≥ 10s.", "", gr.update()
            voice_lang = "vi"
            temp_id, demo_path, sim = create_voice_preview(audio_path, voice_lang, vname, ptext)
            v = _voices_db[temp_id]
            if sim >= 0:
                sim_msg = f"🔍 Similarity (ước lượng giống giọng): **{sim*100:.1f}%**"
            else:
                sim_msg = "❔ Không tính được similarity (có thể do lỗi audio)."
            return (
                temp_id,
                demo_path,
                f"✅ Đã tạo demo cho voice tạm: **{v.name}**. Nghe thử, nếu ổn hãy bấm *Lưu voice*.",
                sim_msg,
                v.name,
            )

        temp_voice_id_state, preview_audio, clone_status, similarity_text, save_voice_name = btn_gen_preview.click(
            fn=on_gen_preview,
            inputs=[in_audio, lang_clone, voice_name_input, preview_text_input],
            outputs=[temp_voice_id_state, preview_audio, clone_status, similarity_text, save_voice_name],
        )

        def on_save_voice(temp_id, final_name):
            ok, msg_or_id = finalize_voice(temp_id, final_name)
            if not ok:
                return msg_or_id, refresh_voices_state()
            return f"✅ Đã lưu voice với ID `{msg_or_id}`.", refresh_voices_state()

        clone_status, voices_state = btn_save_voice.click(
            fn=on_save_voice,
            inputs=[temp_voice_id_state, save_voice_name],
            outputs=[clone_status, voices_state],
        )

    # ----- TAB: Text to Speech -----
    with gr.Tab("🗣️ Text to Speech"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Chọn voice & nghe demo")
                voice_dropdown = gr.Dropdown(
                    label="Voice",
                    choices=[],
                    interactive=True,
                )
                btn_refresh_voices = gr.Button("🔄 Refresh danh sách voice")
                voice_demo_audio = gr.Audio(label="Voice preview", type="filepath")
                btn_play_voice_demo = gr.Button("Nghe demo voice")

                gr.Markdown("### Emotion & Ngôn ngữ")
                emotion = gr.Dropdown(
                    label="Emotion (hiện chưa áp trực tiếp, dùng để ghi chú)",
                    choices=[
                        "auto", "neutral", "happy", "sad", "angry",
                        "fearful", "disgusted", "surprised", "fluent"
                    ],
                    value="auto",
                )
                lang_tts = gr.Dropdown(
                    label="Language",
                    choices=["auto", "vietnamese"],
                    value="vietnamese",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Voice Preset")
                preset_dropdown = gr.Dropdown(
                    label="Chọn preset",
                    choices=list(PRESETS.keys()),
                    value=list(PRESETS.keys())[0] if PRESETS else None,
                )
                btn_apply_preset = gr.Button("Áp dụng preset 🎚️")

                # Export / Import presets
                with gr.Row():
                    preset_export_btn = gr.Button("Export presets ⬇️")
                    preset_import_btn = gr.Button("Import presets ⬆️")
                preset_export_file = gr.File(label="Tải presets.json", interactive=False)
                preset_import_file = gr.File(label="Chọn file presets.json để import", file_types=[".json"])
                preset_status = gr.Markdown("")

                gr.Markdown("### Voice Modifier")
                deepen = gr.Slider(-50, 50, value=0, step=1, label="Deepen (âm trầm ↔ sáng)")
                stronger = gr.Slider(-50, 50, value=0, step=1, label="Stronger (mạnh mẽ)")
                nasal = gr.Slider(-50, 50, value=0, step=1, label="Nasal (mũi)")

                gr.Markdown("### Special FX")
                spacious = gr.Checkbox(label="Spacious Echo")
                auditorium = gr.Checkbox(label="Auditorium Echo")
                lofi_tel = gr.Checkbox(label="Lofi Telephone")
                robotic = gr.Checkbox(label="Robotic")

                gr.Markdown("### Speed / Pitch / Volume")
                speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
                pitch = gr.Slider(-12, 12, value=0, step=0.5, label="Pitch (semitones)")
                volume = gr.Slider(0.01, 10.0, value=1.0, step=0.01, label="Volume gain")

        gr.Markdown("### Nhập text để đọc")
        tts_text = gr.Textbox(
            lines=4,
            placeholder="Nhập nội dung cần đọc. Dùng cú pháp: Xin chào<#0.5#>hôm nay trời đẹp.",
            label="Text",
        )
        tts_status = gr.Markdown("")
        tts_output_audio = gr.Audio(
            label="Kết quả audio",
            type="filepath",
        )
        btn_tts = gr.Button("Tạo audio từ text 🚀", variant="primary")

        def on_apply_preset(preset_name):
            cfg = PRESETS.get(preset_name, PRESETS["Neutral"])
            return (
                cfg["deepen"],
                cfg["stronger"],
                cfg["nasal"],
                cfg["spacious_echo"],
                cfg["auditorium_echo"],
                cfg["lofi_tel"],
                cfg["robotic"],
                cfg["speed"],
                cfg["pitch"],
                cfg["volume"],
            )

        btn_apply_preset.click(
            fn=on_apply_preset,
            inputs=[preset_dropdown],
            outputs=[
                deepen, stronger, nasal,
                spacious, auditorium, lofi_tel, robotic,
                speed, pitch, volume,
            ],
        )

        def on_export_presets():
            save_presets_to_disk()
            return PRESETS_PATH

        preset_export_btn.click(
            fn=on_export_presets,
            inputs=[],
            outputs=[preset_export_file],
        )

        def on_import_presets(file_obj):
            global PRESETS
            if file_obj is None:
                return gr.update(choices=list(PRESETS.keys()), value=preset_dropdown.value), "⚠️ Vui lòng chọn file JSON."
            try:
                with open(file_obj.name, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    return gr.update(choices=list(PRESETS.keys()), value=preset_dropdown.value), "⚠️ File JSON không đúng định dạng (phải là object)."

                PRESETS = data
                save_presets_to_disk()

                choices = list(PRESETS.keys())
                default_choice = choices[0] if choices else None
                return gr.update(choices=choices, value=default_choice), "✅ Import presets thành công."
            except Exception as e:
                return gr.update(choices=list(PRESETS.keys()), value=preset_dropdown.value), f"❌ Lỗi khi import: {e}"

        preset_dropdown, preset_status = preset_import_btn.click(
            fn=on_import_presets,
            inputs=[preset_import_file],
            outputs=[preset_dropdown, preset_status],
        )

        def on_refresh_voices():
            return gr.update(choices=refresh_voices_state())

        btn_refresh_voices.click(
            fn=on_refresh_voices,
            inputs=[],
            outputs=[voice_dropdown],
        )

        def on_play_voice_demo(voice_id):
            if not voice_id or voice_id not in _voices_db:
                return "", "⚠️ Chưa chọn voice."
            preview = os.path.join(VOICES_DIR, f"{voice_id}_preview.wav")
            if not os.path.exists(preview):
                return "", "⚠️ Voice này chưa có file demo."
            return preview, "✅ Đang phát demo voice."

        voice_demo_audio, tts_status = btn_play_voice_demo.click(
            fn=on_play_voice_demo,
            inputs=[voice_dropdown],
            outputs=[voice_demo_audio, tts_status],
        )

        def on_tts(
            voice_id, text, emotion_val, language_ui,
            deepen_v, stronger_v, nasal_v,
            spacious_v, auditorium_v, lofi_tel_v, robotic_v,
            speed_v, pitch_v, volume_v,
        ):
            if not voice_id or voice_id not in _voices_db:
                return "", "⚠️ Vui lòng chọn voice."
            if not text.strip():
                return "", "⚠️ Vui lòng nhập text."

            voice = _voices_db[voice_id]
            lang_internal = "vi"  # tập trung tiếng Việt

            modifiers = {
                "emotion": emotion_val,
                "deepen": deepen_v,
                "stronger": stronger_v,
                "nasal": nasal_v,
                "spacious_echo": spacious_v,
                "auditorium_echo": auditorium_v,
                "lofi_tel": lofi_tel_v,
                "robotic": robotic_v,
                "speed": speed_v,
                "pitch": pitch_v,
                "volume": volume_v,
            }

            audio = synthesize_with_pauses(
                text=text,
                voice=voice,
                language=lang_internal,
                modifiers=modifiers,
            )
            out_path = save_history(voice, text, lang_internal, audio, modifiers)
            return out_path, "✅ Tạo audio thành công. Bạn có thể nghe và tải về."

        tts_output_audio, tts_status = btn_tts.click(
            fn=on_tts,
            inputs=[
                voice_dropdown, tts_text, emotion, lang_tts,
                deepen, stronger, nasal,
                spacious, auditorium, lofi_tel, robotic,
                speed, pitch, volume,
            ],
            outputs=[tts_output_audio, tts_status],
        )

    # ----- TAB: Voice Library -----
    with gr.Tab("📚 Voice Library"):
        lib_table = gr.Dataframe(
            headers=["ID", "Tên voice", "Language", "Created at"],
            label="Danh sách voice",
            interactive=False,
        )
        lib_selected = gr.Textbox(label="ID voice được chọn")
        btn_lib_refresh = gr.Button("🔄 Refresh")
        btn_lib_delete = gr.Button("🗑️ Xoá voice")
        lib_status = gr.Markdown("")

        def lib_load():
            reload_dbs()
            rows = []
            for v in _voices_db.values():
                rows.append([
                    v.voice_id,
                    v.name,
                    v.language,
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(v.created_at)),
                ])
            return rows

        btn_lib_refresh.click(fn=lambda: lib_load(), inputs=[], outputs=[lib_table])

        def on_lib_delete(vid):
            msg = delete_voice(vid.strip())
            return msg, lib_load(), refresh_voices_state()

        lib_status, lib_table, voices_state = btn_lib_delete.click(
            fn=on_lib_delete,
            inputs=[lib_selected],
            outputs=[lib_status, lib_table, voices_state],
        )

    # ----- TAB: History -----
    with gr.Tab("🕒 History"):
        history_dropdown = gr.Dropdown(
            label="Chọn bản ghi audio",
            choices=[],
        )
        btn_hist_refresh = gr.Button("🔄 Refresh History")
        hist_audio = gr.Audio(label="Audio đã tạo", type="filepath")
        hist_text = gr.Textbox(label="Text gốc", lines=4)

        def hist_refresh():
            reload_dbs()
            return gr.update(choices=list_history_for_ui())

        btn_hist_refresh.click(fn=hist_refresh, inputs=[], outputs=[history_dropdown])

        def hist_load_item(history_id):
            if not history_id:
                return "", ""
            audio_path, text = get_history_audio(history_id)
            return audio_path, text

        hist_audio, hist_text = history_dropdown.change(
            fn=hist_load_item,
            inputs=[history_dropdown],
            outputs=[hist_audio, hist_text],
        )

    # Khởi tạo dropdown voice khi app start
    demo.load(
        fn=lambda: gr.update(choices=refresh_voices_state()),
        inputs=[],
        outputs=[voice_dropdown],
    )

if __name__ == "__main__":
    # share=True để có link public từ Colab
    demo.launch(share=True)
