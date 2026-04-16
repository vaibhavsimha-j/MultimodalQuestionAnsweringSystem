import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import easyocr
import cv2
import librosa
import csv
import tempfile
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

# PAGE CONFIGURATION
st.set_page_config(page_title="Multimodal QA System", page_icon="⚙️", layout="wide")

# SIDEBAR
with st.sidebar:
    st.title("🔑 API Configuration")
    st.markdown("Enter your API keys to start the Multimodal QA System.")

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get it from console.groq.com"
    )

    assembly_key = st.text_input(
        "AssemblyAI API Key",
        type="password",
        help="Get it from assemblyai.com"
    )

    st.divider()
    st.info("💡 Your API Keys are not stored in our APP !")

if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key

if assembly_key:
    aai.settings.api_key = assembly_key

# CONFIGURATION
device = "cuda" if torch.cuda.is_available() else "cpu"

# CACHE MODELS
@st.cache_resource
class FeatureExtractor:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)

        self.yolo_model = YOLO('yolov8n.pt')

        self.video_processor = VideoMAEImageProcessor.from_pretrained(
            "facebook/timesformer-base-finetuned-k400"
        )
        self.video_model = TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400"
        ).to(device)

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

        if file_path.lower().endswith((('.mp4', '.avi', '.mov', '.mkv'))):
            try:
                video = VideoFileClip(file_path)
                if video.audio:
                    video.audio.write_audiofile(temp_audio, verbose=False, logger=None)
                    target_path = temp_audio
                video.close()
            except:
                return "Audio Extraction Failed"

        try:
            config = aai.TranscriptionConfig(
                speech_model=aai.SpeechModel.best
            )
            transcript = self.transcriber.transcribe(target_path, config=config)

            if os.path.exists(temp_audio) and target_path == temp_audio:
                os.remove(temp_audio)

            return transcript.text if transcript.text else "No speech detected."

        except Exception as e:
            return f"Transcription Failed: {e}"

    def get_sound_profile(self, file_path):
        audio_path = file_path
        temp_wav = "temp_sound_analysis.wav"

        if file_path.lower().endswith((('.mp4', '.avi', '.mov', '.mkv'))):
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
        inputs = self.clip_processor(
            text=text_list,
            images=image,
            return_tensors="pt",
            padding=True
        )

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

        inputs = self.video_processor(list(frames), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.video_model(**inputs)

        return self.video_model.config.id2label[outputs.logits.argmax().item()]

    def get_ocr_text(self, image_path):
        try:
            results = self.ocr_reader.readtext(image_path, detail=0)
            return ", ".join(results) if results else "No text detected"
        except Exception as e:
            return f"OCR Failed: {e}"

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

# LOAD ONLY AFTER API KEYS ARE ENTERED
if groq_key and assembly_key:
    extractor = FeatureExtractor()
    brain = ReasoningEngine()
else:
    extractor = None
    brain = None

# MAIN UI
st.title("Multimodal QA System")
st.caption("Answers your queries regarding your TEXT,AUDIO, VIDEO & IMAGE input")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Media")

    uploaded_file = st.file_uploader(
        "Upload Video or Image",
        type=['mp4', 'avi', 'mov', 'mkv', 'mp3', 'wav', 'm4a', 'flac', 'jpg', 'jpeg', 'png']
    )

    user_query = st.text_input(
        "What would you like to know?",
        placeholder="Whats happening in the video?"
    )

    run_button = st.button("Query", use_container_width=True)

with col2:
    st.subheader("Preview of the Uploaded Content")

    if uploaded_file:
        file_type = uploaded_file.type

        if file_type.startswith("video"):
            st.video(uploaded_file)
        elif file_type.startswith("audio"):
            st.audio(uploaded_file)
        else:
            st.image(uploaded_file, use_container_width=True)

if run_button:
    if not groq_key or not assembly_key:
        st.error("Please enter both API keys.")

    elif not uploaded_file:
        st.error("Please upload a file.")

    elif not user_query:
        st.error("Please enter a query.")

    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        is_video = file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        is_audio = file_path.lower().endswith(('.mp3', '.wav', '.m4a', '.flac'))

        analysis_data = {}

        with st.status("Processing Multimodal Streams...", expanded=True):
            if is_video or is_audio:
                st.write("Processing Audio...")
                analysis_data['transcript'] = extractor.get_audio_transcript(file_path)

                st.write("Classifying Environment Sounds...")
                analysis_data['sound_tags'] = extractor.get_sound_profile(file_path)

            if is_video:
                st.write("Analyzing Video Action...")
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
                st.write("Generating Caption...")
                analysis_data['scout'] = extractor.get_blip_scout(raw_img)

                st.write("Detecting Objects...")
                analysis_data['yolo_objects'] = extractor.get_yolo_detections(raw_img)

                st.write("Reading Text...")
                temp_ocr_path = "temp_frame_ocr.jpg"
                raw_img.save(temp_ocr_path)
                analysis_data['ocr_text'] = extractor.get_ocr_text(temp_ocr_path)

                if os.path.exists(temp_ocr_path):
                    os.remove(temp_ocr_path)

                st.write("Generating CLIP Candidates...")
                prompt = f"Query: {user_query}\nContext: {analysis_data['scout']}\nGenerate 15 visual candidates (comma list)."

                resp = brain.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile"
                )

                candidates = [c.strip() for c in resp.choices[0].message.content.split(',')]
                logits = extractor.get_clip_embeddings(raw_img, candidates)
                analysis_data['clip_match'] = candidates[logits.argmax().item()]

            st.write("Generating Final Answer...")
            final_answer = brain.final_answer(user_query, analysis_data)

        st.divider()
        st.subheader("Final AI Answer")
        st.success(final_answer)

        with st.expander("Detailed Extraction Logs"):
            st.json(analysis_data)

        if os.path.exists(file_path):
            os.remove(file_path)
