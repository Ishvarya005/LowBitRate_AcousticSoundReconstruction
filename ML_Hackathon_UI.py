import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import torch
import torch.nn as nn
import os

# --- WaveUNet Model ---
class WaveUNet(nn.Module):
    def __init__(self):
        super(WaveUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- Load Model ---
def load_model(path):
    model = WaveUNet()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

# --- Reconstruct Audio ---
def reconstruct(audio, model):
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    with torch.no_grad():
        recon = model(audio_tensor)
    return recon.squeeze().numpy()

# --- SI-SDR ---
def compute_sisdr(original, reconstructed):
    eps = 1e-10
    alpha = np.sum(original * reconstructed) / (np.sum(original ** 2) + eps)
    s_target = alpha * original
    e_noise = reconstructed - s_target
    return 10 * np.log10(np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + eps))

# --- STOI-like ---
def compute_stoi_like(original, reconstructed, sr=16000):
    frame_len = int(0.025 * sr)
    frame_shift = int(0.01 * sr)
    orig_frames = librosa.util.frame(original, frame_length=frame_len, hop_length=frame_shift)
    recon_frames = librosa.util.frame(reconstructed, frame_length=frame_len, hop_length=frame_shift)

    corr = np.sum(orig_frames * recon_frames, axis=0)
    orig_energy = np.linalg.norm(orig_frames, axis=0)
    recon_energy = np.linalg.norm(recon_frames, axis=0)

    stoi_like = np.mean(corr / (orig_energy * recon_energy + 1e-10))
    return stoi_like

# --- WER-like ---
def compute_wer_like(original, reconstructed, sr=16000):
    mfcc1 = librosa.feature.mfcc(y=original, sr=sr, n_mfcc=13).T
    mfcc2 = librosa.feature.mfcc(y=reconstructed, sr=sr, n_mfcc=13).T
    _, wp = librosa.sequence.dtw(mfcc1.T, mfcc2.T, metric='euclidean')
    return len(wp) / max(len(mfcc1), len(mfcc2))

# --- LSD ---
def compute_lsd(original, reconstructed, sr=16000):
    spec_orig = np.log1p(np.abs(librosa.stft(original, n_fft=512)))
    spec_recon = np.log1p(np.abs(librosa.stft(reconstructed, n_fft=512)))
    lsd = np.mean(np.sqrt(np.mean((spec_orig - spec_recon) ** 2, axis=0)))
    return lsd

# --- Streamlit UI ---
st.title("🎧 Audio Reconstruction using Wave-U-Net")

task_type = st.radio("Select Type", ["Speech", "Music"])
bitrate = st.selectbox("Choose Bitrate Model", ["3kbps", "6kbps", "12kbps"] if task_type == "Speech" else ["16kbps", "32kbps", "64kbps"])

# 📁 Model paths
if task_type == "Speech":
    model_paths = {
        "3kbps": r"C:\Users\anany\OneDrive - Amrita vishwa vidyapeetham\AMRITA\SEMESTER-4\ML\HACKATHON\Trained_Models_Speech\WaveUNet_speech_3kbps.pth",
        "6kbps": r"C:\Users\anany\OneDrive - Amrita vishwa vidyapeetham\AMRITA\SEMESTER-4\ML\HACKATHON\Trained_Models_Speech\WaveUNet_speech_6kbps.pth",
        "12kbps": r"C:\Users\anany\OneDrive - Amrita vishwa vidyapeetham\AMRITA\SEMESTER-4\ML\HACKATHON\Trained_Models_Speech\WaveUNet_speech_12kbps.pth"
    }
else:
    model_paths = {
        "16kbps": r"C:\Users\anany\OneDrive - Amrita vishwa vidyapeetham\AMRITA\SEMESTER-4\ML\HACKATHON\Trained_Models_Music\WaveUNet_16kbps.pth",
        "32kbps": r"C:\Users\anany\OneDrive - Amrita vishwa vidyapeetham\AMRITA\SEMESTER-4\ML\HACKATHON\Trained_Models_Music\WaveUNet_32kbps.pth",
        "64kbps": r"C:\Users\anany\OneDrive - Amrita vishwa vidyapeetham\AMRITA\SEMESTER-4\ML\HACKATHON\Trained_Models_Music\WaveUNet_64kbps.pth"
    }

model = load_model(model_paths[bitrate])

orig_file = st.file_uploader("📤 Upload Original Audio", type=["wav"])
comp_file = st.file_uploader("📤 Upload Compressed Audio", type=["wav", "opus"])

if orig_file and comp_file:
    original, sr = librosa.load(orig_file, sr=16000, mono=True)
    compressed, _ = librosa.load(comp_file, sr=16000, mono=True)

    reconstructed = reconstruct(compressed, model)

    # Trim to same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    # 🎯 Metrics
    st.subheader("📊 Evaluation Metrics")
    sisdr = compute_sisdr(original, reconstructed)
    stoi = compute_stoi_like(original, reconstructed, sr)

    st.write(f"🔹 SI-SDR: {sisdr:.2f} dB")
    st.write(f"🔹 STOI-like: {stoi:.4f}")

    if task_type == "Speech":
        wer = compute_wer_like(original, reconstructed, sr)
        st.write(f"🔹 WER-like (Normalized DTW): {wer:.4f}")
    else:
        lsd = compute_lsd(original, reconstructed, sr)
        st.write(f"🔹 LSD: {lsd:.4f}")

    # 🎧 Listen
    st.subheader("🎧 Listen to Audio Samples")
    
    st.markdown("**1️⃣ Original Audio:**")
    st.audio(orig_file, format="audio/wav")
    
    st.markdown("**2️⃣ Compressed Audio:**")
    st.audio(comp_file, format="audio/wav")

    sf.write("reconstructed.wav", reconstructed, sr)
    st.markdown("**3️⃣ Reconstructed Audio:**")
    st.audio("reconstructed.wav", format="audio/wav")
