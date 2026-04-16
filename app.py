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
st.set_page_config(page_title="Multimodal QA System", page_icon="⚙️", layout="wide")

# --- 2. SIDEBAR FOR API KEYS (SECURITY) ---
with st.sidebar:
    st.title("🔑 API Configuration")
    st.markdown("Enter your API keys to start the Multimodal QA System.")
    
    input_groq_key = st.text_input("Groq API Key", type="password", help="Get it from console.groq.com")
    input_aai_key = st.text_input("AssemblyAI API Key", type="password", help="Get it from assemblyai.com")
    
    if input_groq_key:
        os.environ["GROQ_API_KEY"] = input_groq_key
    if input_aai_key:
        aai.settings.api_key = input_aai_key

    st.divider()
    st.info("💡 Your API Keys are not stored in our APP !")

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


# Load models once
models = load_heavy_models()

# --- 4. FEATURE EXTRACTION LOGIC ---
# FEATURE EXTRACTION :
class FeatureExtractor:
    def __init__(self):
        print(" Loading Vision Models (CLIP, BLIP, YOLO, TimeSformer)...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        self.yolo_model = YOLO('yolov8n.pt')
        self.video_processor = VideoMAEImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.video_model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400").to(device)

        print(" Loading Sound Model (YAMNet)...")
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

        class_map_path = self.yamnet_model.class_map_path().numpy().decode('utf-8')
        self.class_names = []
        with open(class_map_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.class_names.append(row['display_name'])

        print(" Initializing AssemblyAI & OCR...")
        self.transcriber = aai.Transcriber()
        self.ocr_reader = easyocr.Reader(['en'], gpu=(device=="cuda"))

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
            except: return "Audio Extraction Failed"

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
            except: return "Sound extraction failed"

        try:
            waveform, _ = librosa.load(audio_path, sr=16000)
            scores, _, _ = self.yamnet_model(waveform)
            avg_scores = np.mean(scores, axis=0)
            top_indices = np.argsort(avg_scores)[-3:][::-1]
            results = [self.class_names[i] for i in top_indices]
            if os.path.exists(temp_wav): os.remove(temp_wav)
            return ", ".join(results)
        except: return "Sound analysis failed"

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
        if not detections: return "No specific objects"
        return ", ".join(set(detections))

    def get_timesformer_action(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: return "Invalid Video"

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
            if len(frames) == 8: break
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

# REASONING ENGINE :
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
        resp = self.client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile")
        return resp.choices[0].message.content


# --- 5. MAIN UI LAYOUT ---
st.title("Multimodal QA System")
st.caption("Answers your queries regarding your TEXT,AUDIO, VIDEO & IMAGE input ")

if not input_groq_key or not input_aai_key:
    st.warning("⚠️ Please provide your API keys in the sidebar to start.")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Upload Media")
    uploaded_file = st.file_uploader("Upload Video or Image", type=['mp4', 'mov', 'avi', 'jpg', 'png', 'jpeg'])
    query = st.text_input("❓ What would you like to know?", placeholder="e.g., What brand of soda is on the table?")
    run_btn = st.button("🔍 Query", type="primary", use_container_width=True)

with col2:
    st.subheader("Preview of the Uploaded Content")
    if uploaded_file:
        if uploaded_file.type.startswith('video'):
            st.video(uploaded_file)
        else:
            st.image(uploaded_file)

# --- 6. EXECUTION PIPELINE ---
if run_btn and uploaded_file and query:
    extractor = FeatureExtractor()
    brain = ReasoningEngine()
    
    # Save upload to temp file for processing
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(uploaded_file.read())
        path = tfile.name

    with st.status("Processing Multimodal Streams...", expanded=True) as status:
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
        
        status.update(label="Analysis Complete!", state="complete")

    # --- 7. FINAL DISPLAY ---
    st.divider()
    st.subheader("💡 Final AI Answer")
    st.info(final_ans.choices[0].message.content)
    
    with st.expander("🔍 View Detailed Extraction Logs"):
        st.json(results)

    os.remove(path)
