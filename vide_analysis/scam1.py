import pandas as pd
import cv2
import os
import pytesseract
import torch
import whisper
import moviepy.editor as mp
import speech_recognition as sr
import torchvision.transforms as transforms
from torchvision import models
from transformers import pipeline
from PIL import Image

# Set device: GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
# data = pd.read_csv(r'D:\Projects\hacky\keywords.csv')
# Set up scam-related keywords
SCAM_KEYWORDS = [
    "free money", "click this link", "urgent", "congratulations", 
    "lottery", "giveaway", "limited offer", "investment", "crypto scheme",
    "xbet","win big","2x","2x profit","High odds","free bonus","free credit",
]
# SCAM_KEYWORDS=data['Scam Keywords']
# Load ResNet50 model for image classification and set it to evaluation mode
resnet = models.resnet50(pretrained=True)
resnet.eval()

# Define image transformation (expects a PIL image)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load Whisper model for speech-to-text and send it to device
whisper_model = whisper.load_model("base").to(device)

# Load NLP model for scam phrase detection (set device index: 0 if using GPU, -1 for CPU)
nlp_model = pipeline("text-classification", 
                     model="distilbert-base-uncased-finetuned-sst-2-english", 
                     device=0 if device == "cuda" else -1)

def extract_frames(video_path, output_folder="frames"):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    return output_folder

# Function to extract audio from a video using moviepy
def extract_audio(video_path, audio_path="audio.wav"):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    return audio_path

# Function to perform OCR on frames using pytesseract
def analyze_text_in_frames(frame_folder):
    scam_texts = []
    for frame in os.listdir(frame_folder):
        frame_path = os.path.join(frame_folder, frame)
        img = Image.open(frame_path)
        text = pytesseract.image_to_string(img)
        # Check for scam keywords in text
        for keyword in SCAM_KEYWORDS:
            if keyword.lower() in text.lower():
                scam_texts.append((frame_path, text))
    return scam_texts

# Function to analyze speech using SpeechRecognition and Google API
def analyze_audio(audio_path):
    recognizer = sr.Recognizer()
    scam_audio_texts = []

    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            for keyword in SCAM_KEYWORDS:
                if keyword.lower() in text.lower():
                    scam_audio_texts.append(text)
        except sr.UnknownValueError:
            pass

    return scam_audio_texts

# Function to analyze image frames using ResNet50 on the proper device
def analyze_frames_with_resnet(frame_folder):
    scam_frames = []
    for frame in os.listdir(frame_folder):
        frame_path = os.path.join(frame_folder, frame)
        img = Image.open(frame_path)
        img_tensor = transform(img).unsqueeze(0)
        img_tensor = img_tensor.to(device)
        resnet.to(device)
        output = resnet(img_tensor)
        if torch.max(output).item() > 0.8:
            scam_frames.append(frame_path)
    return scam_frames

# Function to analyze text using the NLP model
def analyze_text_nlp(text):
    results = nlp_model(text)
    return results[0]["label"] == "NEGATIVE"

# Main function to detect scam ad in a video
def detect_scam_ad(video_path):
    print(f"Processing video: {video_path} 🚀")

    # Extract frames and audio from the video
    frame_folder = extract_frames(video_path)
    audio_path = extract_audio(video_path)

    # Analyze text in extracted frames
    scam_texts = analyze_text_in_frames(frame_folder)
    # Analyze speech from the audio
    scam_audio_texts = analyze_audio(audio_path)

    scam_frames = analyze_frames_with_resnet(frame_folder)

    # Combine all extracted texts and apply NLP scam detection
    all_texts = [text for _, text in scam_texts] + scam_audio_texts
    scam_nlp_detected = any(analyze_text_nlp(text) for text in all_texts)

    # Print results
    print("\n Scam Ad Detection Results:")
    print(f" Scam-related images: {len(scam_frames)}")
    print(f" Scam-related text found in frames: {len(scam_texts)}")
    print(f" Scam-related speech detected: {len(scam_audio_texts)}")
    print(f" NLP scam detection: {'YES' if scam_nlp_detected else 'NO'}")

    # Final decision
    if scam_texts or scam_audio_texts or scam_frames or scam_nlp_detected:
        print("\n Scam Ad Detected! ")
        return True
    else:
        print("\n This ad appears safe.")
        return False

video_file = "train1.mp4" 
detect_scam_ad(video_file)
