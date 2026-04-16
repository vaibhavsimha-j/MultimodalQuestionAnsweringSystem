import streamlit as st
import torch
import numpy as np
import os
import easyocr
import cv2
import librosa
import csv
import assemblyai as aai
import tensorflow_hub as hub
from PIL import Image
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    VideoMAEImageProcessor, TimesformerForVideoClassification
)
from ultralytics import YOLO
from groq import Groq
from moviepy.editor import VideoFileClip
import tempfile

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal QA System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DARK THEME CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── global background ── */
  .stApp { background-color: #18142b; color: #e0e0f0; }

  /* ── sidebar ── */
  section[data-testid="stSidebar"] {
      background-color: #111128 !important;
      border-right: 1px solid #2a2a4a;
  }
  section[data-testid="stSidebar"] * { color: #d0d0e8 !important; }

  /* ── sidebar text inputs ── */
  section[data-testid="stSidebar"] .stTextInput > div > div > input {
      background-color: #1c1c38 !important;
      border: 1px solid #3a3a60 !important;
      border-radius: 6px !important;
      color: #e0e0f0 !important;
  }

  /* ── info box in sidebar ── */
  section[data-testid="stSidebar"] .stAlert {
      background-color: #181830 !important;
      border: 1px solid #3a3a60 !important;
      border-radius: 8px !important;
  }

  /* ── main title ── */
  h1 { color: #ffffff !important; font-size: 2.4rem !important; font-weight: 700 !important; }

  /* ── section headers ── */
  h2, h3 { color: #c0c0e0 !important; font-weight: 600 !important; }

  /* ── panel cards ── */
  .panel-card {
      background-color: #151530;
      border: 1px solid #2a2a50;
      border-radius: 12px;
      padding: 20px 24px;
  }

  /* ── file uploader ── */
  .stFileUploader > div {
      background-color: #151530 !important;
      border: 1px dashed #3a3a60 !important;
      border-radius: 10px !important;
  }
  .stFileUploader label { color: #c0c0e0 !important; }

  /* ── text input (query box) ── */
  .stTextInput > div > div > input {
      background-color: #1c1c38 !important;
      border: 1px solid #3a3a60 !important;
      border-radius: 8px !important;
      color: #e0e0f0 !important;
  }
  .stTextInput label { color: #b0b0d0 !important; }

  /* ── Query button ── */
  .stButton > button {
      background-color: #2a2a50 !important;
      color: #e0e0f0 !important;
      border: 1px solid #4a4a80 !important;
      border-radius: 8px !important;
      width: 100%;
      font-size: 1rem !important;
      padding: 0.5rem 1rem !important;
      transition: background-color 0.2s;
  }
  .stButton > button:hover {
      background-color: #3a3a70 !important;
      border-color: #6a6ab0 !important;
  }

  /* ── preview container ── */
  .preview-box {
      background-color: #0a0a1e;
      border: 1px solid #2a2a50;
      border-radius: 12px;
      min-height: 380px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
  }

  /* ── answer box ── */
  .answer-box {
      background-color: #151530;
      border: 1px solid #3a3a60;
      border-radius: 10px;
      padding: 16px 20px;
      color: #e0e0f0;
      margin-top: 12px;
  }

[data-testid="stFileUploader"] {
    max-width: 220px;
}
[data-testid="stFileUploader"] section {
    padding: 1rem !important;
    min-height: auto !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

[data-testid="stFileUploader"] small {
    display: none !important;
}

[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzoneInstructions"] {
    display: none !important;
}

[data-testid="stFileUploader"] button {
    width: 180px !important;
}

  /* ── divider ── */
  hr { border-color: #2a2a4a !important; }

  /* ── spinner ── */
  .stSpinner > div { border-top-color: #7070c0 !important; }

  /* ── hide default Streamlit header/footer ── */
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── DEVICE ─────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"


# ── FEATURE EXTRACTOR (logic unchanged) ───────────────────────────────────────
class FeatureExtractor:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        self.yolo_model = YOLO('yolov8n.pt')
        self.video_processor = VideoMAEImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.video_model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400").to(device)

        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        class_map_path = self.yamnet_model.class_map_path().numpy().decode('utf-8')
        self.class_names = []
        with open(class_map_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.class_names.append(row['display_name'])

        self.transcriber = aai.Transcriber()
        self.ocr_reader = easyocr.Reader(['en'], gpu=(device == "cuda"))

    def get_audio_transcript(self, file_path):
        target_path = file_path
        temp_audio = "temp_audio_speech.mp3"
        if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            try:
                video = VideoFileClip(file_path)
                if video.audio:
                    video.audio.write_audiofile(temp_audio, verbose=False, logger=None)
                    target_path = temp_audio
                video.close()
            except:
                return "Audio Extraction Failed"

        try:
            config = aai.TranscriptionConfig(speech_models=["universal-3-pro", "universal-2"])
            transcript = self.transcriber.transcribe(target_path, config=config)
            if os.path.exists(temp_audio) and target_path == temp_audio:
                os.remove(temp_audio)
            return transcript.text if transcript.text else "No speech detected."
        except Exception as e:
            return f"Transcription Failed: {e}"

    def get_sound_profile(self, file_path):
        audio_path = file_path
        temp_wav = "temp_sound_analysis.wav"
        if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            try:
                video = VideoFileClip(file_path)
                video.audio.write_audiofile(temp_wav, codec='pcm_s16le', verbose=False, logger=None)
                audio_path = temp_wav
                video.close()
            except:
                return "Sound extraction failed"

        try:
            waveform, _ = librosa.load(audio_path, sr=16000)
            scores, _, _ = self.yamnet_model(waveform)
            avg_scores = np.mean(scores, axis=0)
            top_indices = np.argsort(avg_scores)[-3:][::-1]
            results = [self.class_names[i] for i in top_indices]
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            return ", ".join(results)
        except:
            return "Sound analysis failed"

    def get_clip_embeddings(self, image, text_list):
        inputs = self.clip_processor(text=text_list, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        return outputs.logits_per_image

    def get_blip_scout(self, image):
        inputs = self.blip_processor(image, return_tensors="pt").to(device)
        out = self.blip_model.generate(**inputs)
        return self.blip_processor.decode(out[0], skip_special_tokens=True)

    def get_yolo_detections(self, image):
        results = self.yolo_model(image, verbose=False, conf=0.5)
        detections = [self.yolo_model.names[int(c)] for c in results[0].boxes.cls.cpu().numpy()]
        if not detections:
            return "No specific objects"
        return ", ".join(set(detections))

    def get_timesformer_action(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return "Invalid Video"

        indices = np.linspace(0, total_frames - 1, 8).astype(int)
        count = 0
        success = True
        while success:
            success, frame = cap.read()
            if success and count in indices:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            count += 1
            if len(frames) == 8:
                break
        cap.release()

        while len(frames) < 8:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

        inputs = self.video_processor(list(frames), return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.video_model(**inputs)

        return self.video_model.config.id2label[outputs.logits.argmax().item()]

    def get_ocr_text(self, image_path):
        try:
            results = self.ocr_reader.readtext(image_path, detail=0)
            return ", ".join(results) if results else "No text detected"
        except Exception as e:
            return f"OCR Failed: {e}"


# ── REASONING ENGINE (logic unchanged) ────────────────────────────────────────
class ReasoningEngine:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def final_answer(self, query, data):
        prompt = f"""
        **Query:** {query}
        **Audio Evidence:**
        - Speech Transcript: "{data.get('transcript', 'N/A')}"
        - Sound Profile: "{data.get('sound_tags', 'N/A')}"
        **Visual Evidence:**
        - Scene: {data.get('scout', 'N/A')}
        - Action: {data.get('action', 'N/A')}
        - Objects: {data.get('yolo_objects', 'N/A')}
        - Classification (CLIP): {data.get('clip_match', 'N/A')}
        - Extracted Text (OCR): {data.get('ocr_text', 'N/A')}

        **Instructions:** Answer by combining audio and visual evidence. Be direct and concise.
        """
        resp = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        return resp.choices[0].message.content


# ── CACHED MODEL LOADER ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models(groq_key: str, assembly_key: str):
    os.environ["GROQ_API_KEY"] = groq_key
    aai.settings.api_key = assembly_key
    extractor = FeatureExtractor()
    brain = ReasoningEngine()
    return extractor, brain


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔑 API Configuration")
    st.markdown("Enter your API keys to start the Multimodal QA System.")
    st.markdown("")

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Your Groq API key from console.groq.com",
    )
    assembly_key = st.text_input(
        "AssemblyAI API Key",
        type="password",
        placeholder="xxxxxxxxxxxxxxxx",
        help="Your AssemblyAI API key from assemblyai.com",
    )

    st.markdown("")
    st.info("🔒 Your API Keys are SAFE with us!")

    keys_ready = bool(groq_key and assembly_key)


# ── MAIN LAYOUT ────────────────────────────────────────────────────────────────
st.markdown("# Multimodal QA System")
st.markdown(
    "<p style='color:#8080b0; margin-top:-10px; margin-bottom:20px;'>"
    "Answers queries regarding your <b>AUDIO</b>, <b>VIDEO</b> &amp; <b>IMAGE</b>"
    "</p>",
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 1], gap="large")

# ── LEFT COLUMN ────────────────────────────────────────────────────────────────
with left_col:
    st.markdown("Upload Media")
    
    uploaded_file = st.file_uploader(
        "Upload Image, Video or Audio",
        type=["mp4", "avi", "mov", "mkv", "mp3", "wav", "m4a", "flac", "jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    st.caption("Supported Formats: mp4, avi, mov, mkv, mp3, wav, m4a, flac, jpg, jpeg, png, webp")
    
    st.markdown("Ask Anything 💡!")
    user_query = st.text_input("",
        placeholder="Enter your Query",
        label_visibility="collapsed"
    )

    st.markdown("")
    query_clicked = st.button("Run", disabled=not keys_ready or uploaded_file is None or not user_query)

    if not keys_ready:
        st.caption("⚠️ Enter both the API keys in the sidebar to enable querying.")

# ── RIGHT COLUMN ───────────────────────────────────────────────────────────────
with right_col:
    st.markdown(
        """
        <div style="margin-top:-40px;">
            <h3 style="color:#c0c0e0; margin-bottom:20px;">Preview</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    if uploaded_file is not None:
        file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
        is_video = file_ext in ("mp4", "avi", "mov", "mkv")
        is_audio = file_ext in ("mp3", "wav", "m4a", "flac")
        is_image = file_ext in ("jpg", "jpeg", "png", "webp")
        
        preview_left, preview_center, preview_right = st.columns([0.1, 0.8, 0.1])
        
        if is_video:
            st.video(uploaded_file)
        elif is_image:
            st.image(uploaded_file, width= 450)
        elif is_audio:
            st.audio(uploaded_file)
    else:
        st.markdown(
            "<div style='background:#0a0a1e; border:1px solid #2a2a50; border-radius:12px;"
            " min-height:340px; display:flex; align-items:center; justify-content:center;"
            " color:#3a3a60; font-size:1rem;'>Upload a file to see a preview</div>",
            unsafe_allow_html=True,
        )

# ── PIPELINE EXECUTION ─────────────────────────────────────────────────────────
if query_clicked and uploaded_file is not None and user_query and keys_ready:

    # Save uploaded file to a temp path so file-path-based models can read it
    suffix = "." + uploaded_file.name.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    file_ext = suffix.lstrip(".")
    is_video = file_ext in ("mp4", "avi", "mov", "mkv")
    is_audio = file_ext in ("mp3", "wav", "m4a", "flac")

    st.markdown("---")
    st.markdown("### Answer")

    with st.spinner("Loading models (first run may take a few minutes)..."):
        extractor, brain = load_models(groq_key, assembly_key)

    analysis_data = {}

    with st.status("Running multimodal analysis...", expanded=True) as status:

        if is_video or is_audio:
            st.write("Transcribing speech...")
            analysis_data['transcript'] = extractor.get_audio_transcript(file_path)
            st.write("Classifying environment sounds...")
            analysis_data['sound_tags'] = extractor.get_sound_profile(file_path)

        if is_video:
            st.write("Analyzing video action...")
            analysis_data['action'] = extractor.get_timesformer_action(file_path)
            cap = cv2.VideoCapture(file_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 2))
            _, frame = cap.read()
            raw_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        elif not is_audio:
            raw_img = Image.open(file_path).convert("RGB")
        else:
            raw_img = None

        if raw_img:
            st.write("Generating caption...")
            analysis_data['scout'] = extractor.get_blip_scout(raw_img)
            st.write("Detecting objects...")
            analysis_data['yolo_objects'] = extractor.get_yolo_detections(raw_img)

            st.write("Reading text (OCR)...")
            temp_ocr_path = "temp_frame_ocr.jpg"
            raw_img.save(temp_ocr_path)
            analysis_data['ocr_text'] = extractor.get_ocr_text(temp_ocr_path)
            if os.path.exists(temp_ocr_path):
                os.remove(temp_ocr_path)

            st.write("Running CLIP classification...")
            prompt = f"Query: {user_query}\nContext: {analysis_data['scout']}\nGenerate 15 visual candidates (comma list)."
            resp = brain.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile"
            )
            candidates = [c.strip() for c in resp.choices[0].message.content.split(',')]
            logits = extractor.get_clip_embeddings(raw_img, candidates)
            analysis_data['clip_match'] = candidates[logits.argmax().item()]

        st.write("Synthesizing final answer...")
        answer = brain.final_answer(user_query, analysis_data)
        status.update(label="Analysis complete!", state="complete", expanded=False)

    st.markdown(
        f"<div class='answer-box'>{answer}</div>",
        unsafe_allow_html=True,
    )

    # Clean up temp file
    if os.path.exists(file_path):
        os.remove(file_path)
