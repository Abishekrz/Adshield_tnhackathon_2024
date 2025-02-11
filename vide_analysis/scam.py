
import cv2
import os
import pytesseract
import cupy as cp
import torch
import whisper
import moviepy.editor as mp
import speech_recognition as sr
import torchvision.transforms as transforms
from torchvision import models
from transformers import pipeline
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Set up scam-related keywords
SCAM_KEYWORDS = ["free money", "click this link", "urgent", "congratulations", "lottery", "giveaway", "limited offer", "investment", "crypto scheme",
                 "you won lottery","claim your price","2x money","3x money","bet"]

# ✅ Load ResNet50 model for image classification (on GPU)
resnet = models.resnet50(pretrained=True).to(device)
resnet.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ Load Whisper model for speech-to-text (on GPU)
whisper_model = whisper.load_model("base").to(device)

# ✅ Load NLP model for scam phrase detection (on GPU)
nlp_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)

# ✅ Extract frames using OpenCV GPU acceleration
def extract_frames(video_path, output_folder="frames"):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gpu_frame = cv2.cuda_GpuMat()  # Upload frame to GPU
        gpu_frame.upload(frame)
        frame = gpu_frame.download()  # Download processed frame (for saving)

        frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    
    cap.release()
    return output_folder

# ✅ Extract audio from video
def extract_audio(video_path, audio_path="audio.wav"):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    return audio_path

# ✅ Analyze text in frames using Tesseract OCR
def analyze_text_in_frames(frame_folder):
    scam_texts = []
    for frame in os.listdir(frame_folder):
        frame_path = os.path.join(frame_folder, frame)
        text = pytesseract.image_to_string(Image.open(frame_path))
        
        # Check for scam keywords
        for keyword in SCAM_KEYWORDS:
            if keyword.lower() in text.lower():
                scam_texts.append((frame_path, text))
    
    return scam_texts

# ✅ Analyze speech using Whisper (GPU)
def analyze_audio(audio_path):
    scam_audio_texts = []
    result = whisper_model.transcribe(audio_path)
    text = result["text"]
    
    for keyword in SCAM_KEYWORDS:
        if keyword.lower() in text.lower():
            scam_audio_texts.append(text)
    
    return scam_audio_texts

# ✅ Analyze frames with ResNet50 (on GPU)
def analyze_frames_with_resnet(frame_folder):
    scam_frames = []
    for frame in os.listdir(frame_folder):
        frame_path = os.path.join(frame_folder, frame)

        # 🔹 Use CuPy to process frames on GPU
        img = cv2.imread(frame_path)  # Read image with OpenCV
        img_gpu = cp.asarray(img)  # Move to GPU
        img_rgb = cv2.cvtColor(cp.asnumpy(img_gpu), cv2.COLOR_BGR2RGB)  # Convert color
        
        # Convert to PIL for torchvision
        img_pil = Image.fromarray(img_rgb)
        img_tensor = transform(img_pil).unsqueeze(0).to("cuda")

        resnet.to("cuda")
        output = resnet(img_tensor)

        if torch.max(output).item() > 0.8:
            scam_frames.append(frame_path)

    return scam_frames


# ✅ Analyze extracted text using NLP model
def analyze_text_nlp(text):
    results = nlp_model(text)
    return results[0]["label"] == "NEGATIVE"

# ✅ Main function: Detect scam ad
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

# ✅ Run the model
video_file = "test2.mp4"  # Replace with your video file
detect_scam_ad(video_file)
