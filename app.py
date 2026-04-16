import streamlit as st
import torch
import numpy as np
import os
import easyocr
import cv2
import librosa
import csv
import assemblyai as aai
import tensorflow as tf
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

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Multimodal MQA Agent", page_icon="🧠", layout="wide")

# --- 2. SIDEBAR FOR API KEYS (SECURITY) ---
with st.sidebar:
    st.title("🔑 API Configuration")
    st.markdown("Enter your keys to enable Multimodal Analysis.")
    
    input_groq_key = st.text_input("Groq API Key", type="password", help="Get it from console.groq.com")
    input_aai_key = st.text_input("AssemblyAI API Key", type="password", help="Get it from assemblyai.com")
    
    if input_groq_key:
        os.environ["GROQ_API_KEY"] = input_groq_key
    if input_aai_key:
        aai.settings.api_key = input_aai_key

    st.divider()
    st.info("💡 Pro-tip: This app uses YOLOv8 for objects, TimeSformer for actions, and YAMNet for background sounds.")

# --- 3. CACHED MODEL LOADING ---
@st.cache_resource
def load_heavy_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Vision Models
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    yolo_model = YOLO('yolov8n.pt')
    video_processor = VideoMAEImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    video_model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400").to(device)
    
    # Sound Models
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
    class_names = []
    with open(class_map_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.append(row['display_name'])
            
    # Tools
    transcriber = None

    if aai.settings.api_key:
        transcriber = aai.Transcriber()
    ocr_reader = easyocr.Reader(['en'], gpu=(device=="cuda"))
    
    return {
    "clip": (clip_model, clip_processor),
    "blip": (blip_model, blip_processor),
    "yolo": yolo_model,
    "video": (video_model, video_processor),
    "yamnet": (yamnet_model, class_names),
    "aai": transcriber,
    "ocr": ocr_reader,
    "device": device
}

assembly_api_key = st.sidebar.text_input("AssemblyAI API Key", type="password")

if assembly_api_key:
    aai.settings.api_key = assembly_api_key

# Load models once
models = load_heavy_models()

# --- 4. FEATURE EXTRACTION LOGIC ---
class MQAExtractor:
    def __init__(self, m):
        self.m = m
        self.device = m["device"]

    def get_audio_data(self, video_path):
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        
        try:
            video = VideoFileClip(video_path)
            if video.audio:
                # 1. Transcription (AssemblyAI)
                video.audio.write_audiofile(temp_audio, verbose=False, logger=None)
                config = aai.TranscriptionConfig(speech_models=["universal-3-pro"])
                transcript = self.m["aai"].transcribe(temp_audio, config=config)
                
                # 2. Sound Profile (YAMNet)
                video.audio.write_audiofile(temp_wav, codec='pcm_s16le', verbose=False, logger=None)
                waveform, _ = librosa.load(temp_wav, sr=16000)
                scores, _, _ = self.m["yamnet"][0](waveform)
                top_indices = np.argsort(np.mean(scores, axis=0))[-3:][::-1]
                sounds = ", ".join([self.m["yamnet"][1][i] for i in top_indices])
                
                video.close()
                return transcript.text, sounds
            return "No audio track", "No sounds"
        finally:
            for f in [temp_audio, temp_wav]:
                if os.path.exists(f): os.remove(f)

    def get_video_action(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, 8).astype(int)
        for i in range(total):
            ret, frame = cap.read()
            if ret and i in indices:
                frame = cv2.resize(frame, (224, 224))
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(frames) == 8: break
        cap.release()
        
        inputs = self.m["video"][1](list(frames), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.m["video"][0](**inputs)
        return self.m["video"][0].config.id2label[outputs.logits.argmax().item()]

# --- 5. MAIN UI LAYOUT ---
st.title("🤖 Agentic Multimodal QA")
st.caption("Fusing Audio, Visual, and Text signals for deep context understanding.")

if not input_groq_key or not input_aai_key:
    st.warning("⚠️ Please provide your API keys in the sidebar to start.")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Upload Media")
    uploaded_file = st.file_uploader("Upload Video or Image", type=['mp4', 'mov', 'avi', 'jpg', 'png', 'jpeg'])
    query = st.text_input("❓ What would you like to know?", placeholder="e.g., What brand of soda is on the table?")
    run_btn = st.button("🔍 Run Full Pipeline", type="primary", use_container_width=True)

with col2:
    st.subheader("📺 Preview")
    if uploaded_file:
        if uploaded_file.type.startswith('video'):
            st.video(uploaded_file)
        else:
            st.image(uploaded_file)

# --- 6. EXECUTION PIPELINE ---
if run_btn and uploaded_file and query:
    extractor = MQAExtractor(models)
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    
    # Save upload to temp file for processing
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(uploaded_file.read())
        path = tfile.name

    with st.status("🚀 Processing Multimodal Streams...", expanded=True) as status:
        results = {}
        is_vid = uploaded_file.type.startswith('video')

        # --- Audio/Action Logic ---
        if is_vid:
            st.write("🎙️ Processing Audio...")
            results['transcript'], results['sounds'] = extractor.get_audio_data(path)
            st.write("🎞️ Analyzing Action Sequence...")
            results['action'] = extractor.get_video_action(path)
            
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/2))
            _, frame = cap.read()
            raw_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        else:
            raw_img = Image.open(path).convert("RGB")
            results['transcript'], results['sounds'], results['action'] = "N/A", "N/A", "Static Image"

        # --- Visual Spatial Logic ---
        st.write("🖼️ Running Object & Scene Detection...")
        # BLIP
        inputs = models["blip"][1](raw_img, return_tensors="pt").to(models["device"])
        results['scout'] = models["blip"][1].decode(models["blip"][0].generate(**inputs)[0], skip_special_tokens=True)
        # YOLO
        yolo_res = models["yolo"](raw_img, verbose=False, conf=0.5)
        results['objects'] = ", ".join(set([models["yolo"].names[int(c)] for c in yolo_res[0].boxes.cls.cpu().numpy()]))
        # OCR
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_tmp:
            raw_img.save(img_tmp.name)
            results['ocr'] = ", ".join(models["ocr"].readtext(img_tmp.name, detail=0))
            os.remove(img_tmp.name)

        # --- CLIP & LLM Reasoning ---
        st.write("🧠 Reasoning with Llama 3.3...")
        cand_resp = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": f"Query: {query}\nContext: {results['scout']}\nGenerate 15 visual candidates (comma list only)."}],
            model="llama-3.3-70b-versatile"
        )
        candidates = [c.strip() for c in cand_resp.choices[0].message.content.split(',')]
        
        inputs = models["clip"][1](text=candidates, images=raw_img, return_tensors="pt", padding=True).to(models["device"])
        with torch.no_grad():
            logits = models["clip"][0](**inputs).logits_per_image
        results['clip_best'] = candidates[logits.argmax().item()]

        # --- FINAL SYNTHESIS ---
        final_prompt = f"""
        **User Query:** {query}
        **System Evidence:**
        - Speech: {results['transcript']}
        - Background Sounds: {results['sounds']}
        - Visual Scene: {results['scout']}
        - Objects Detected: {results['objects']}
        - Action Recognized: {results['action']}
        - Text Found: {results['ocr']}
        - CLIP Confidence: {results['clip_best']}
        
        **Goal:** Provide a direct, concise answer by combining all evidence. If audio and visual conflict, mention it.
        """
        final_ans = groq_client.chat.completions.create(messages=[{"role": "user", "content": final_prompt}], model="llama-3.3-70b-versatile")
        
        status.update(label="✅ Analysis Complete!", state="complete")

    # --- 7. FINAL DISPLAY ---
    st.divider()
    st.subheader("💡 Final AI Answer")
    st.info(final_ans.choices[0].message.content)
    
    with st.expander("🔍 View Detailed Extraction Logs"):
        st.json(results)

    os.remove(path)
