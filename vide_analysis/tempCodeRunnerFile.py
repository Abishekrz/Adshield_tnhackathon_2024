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

# Set up scam-related keywords
SCAM_KEYWORDS = ["free money", "click this link", "urgent", "congratulations", "lottery", "giveaway", "limited offer", "investment", "crypto scheme"]

# Load ResNet50 model for image classification
resnet = models.resnet50(pretrained=True)
resnet.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load Whisper model for speech-to-text
whisper_model = whisper.load_model("base")

# Load NLP model for scam phrase detection
nlp_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

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

def extract_audio(video_path, audio_path="audio.wav"):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    return audio_path

def analyze_text_in_frames(frame_folder):
    scam_texts = []
    for frame in os.listdir(frame_folder):
        frame_path = os.path.join(frame_folder, frame)
        text = pytesseract.image_to_string(frame_path)
        
        # Check for scam keywords
        for keyword in SCAM_KEYWORDS:
            if keyword.lower() in text.lower():
                scam_texts.append((frame_path, text))
    
    return scam_texts

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

def analyze_frames_with_resnet(frame_folder):
    scam_frames = []
    for frame in os.listdir(frame_folder):
        frame_path = os.path.join(frame_folder, frame)
        img = cv2.imread(frame_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_tensor = transform(img).unsqueeze(0)
        output = resnet(img_tensor)
        
        # Check for scam-related visuals (adjust threshold as needed)
        if torch.max(output).item() > 0.8:
            scam_frames.append(frame_path)
    
    return scam_frames

def analyze_text_nlp(text):
    results = nlp_model(text)
    return results[0]["label"] == "NEGATIVE"

def detect_scam_ad(video_path):
    print(f"Processing video: {video_path} 🚀")

    # Extract frames and audio
    frame_folder = extract_frames(video_path)
    audio_path = extract_audio(video_path)

    # Analyze text in frames
    scam_texts = analyze_text_in_frames(frame_folder)

    # Analyze speech-to-text
    scam_audio_texts = analyze_audio(audio_path)

    # Analyze image frames with ResNet
    scam_frames = analyze_frames_with_resnet(frame_folder)

    # Analyze extracted text using NLP
    all_texts = [text for _, text in scam_texts] + scam_audio_texts
    scam_nlp_detected = any(analyze_text_nlp(text) for text in all_texts)

    # Print results
    print("\n🔎 Scam Ad Detection Results:")
    print(f"📸 Scam-related images: {len(scam_frames)}")
    print(f"📝 Scam-related text found in frames: {len(scam_texts)}")
    print(f"🔊 Scam-related speech detected: {len(scam_audio_texts)}")
    print(f"🤖 NLP scam detection: {'YES' if scam_nlp_detected else 'NO'}")

    # Final decision
    if scam_texts or scam_audio_texts or scam_frames or scam_nlp_detected:
        print("\n🚨 Scam Ad Detected! 🚨")
        return True
    else:
        print("\n✅ This ad appears safe.")
        return False

# Run the model
video_file = "scam_ad.mp4"  # Replace with your video file
detect_scam_ad(video_file)
