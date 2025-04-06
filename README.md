# End-to-End Neural Network for Low Bit-Rate Reconstruction of Sound Files

## 📌 Problem Statement

In audio communication systems, especially under low-bandwidth conditions, audio files are often compressed to very low bitrates. While this helps in transmission efficiency, it significantly degrades audio quality. Our objective is to reconstruct high-fidelity audio from these low-bitrate compressed files using deep learning techniques.

---

## 🎯 Motivation

Despite the growing advancements in audio restoration, we observed that there has been relatively limited research focused specifically on *low-bitrate audio reconstruction*, particularly for speech and music. A quick literature scan on platforms like Google Scholar revealed few works in this domain — signaling a research gap and an opportunity to explore and contribute.

---

## 🎧 What is Compression?

Compression is the process of reducing the size of audio files by removing perceptually less important components. In many cases, compression alters the dynamic range:
- **Louder parts may become softer**
- **Softer parts may become louder**
This leads to a loss in detail and quality, especially at lower bitrates (e.g., 3kbps, 6kbps).

---

## ⚙️ Approach

We explored audio reconstruction from compressed files using three datasets:
- **Speech**: `librivox`, `us-gov` subsets from the MUSAN dataset
- **Music**: Subsets from MUSAN
- Each file was compressed using the **Opus codec** at 3kbps, 6kbps, and 12kbps.

---

## 🔊 Why Opus?

Opus is a widely-used open, royalty-free codec designed for interactive speech and audio transmission over the internet. It offers:
- Low-latency encoding
- High audio fidelity even at low bitrates
- Efficient support for both speech and music

---

## 🧠 Models Used

### 🌀 Wave-U-Net (for Music)

- A convolutional encoder-decoder architecture with skip connections
- Well-suited for music source separation and audio-to-audio tasks
- Reconstructs complex musical signals by capturing temporal features at multiple scales

### 🧊 HDemucs (for Speech)

- A hybrid architecture that combines convolutional and recurrent layers
- Trained to separate or enhance audio in noisy or compressed conditions
- Reconstructs stereo audio (we converted mono signals by duplicating channels)

---

## ❌ Other Models Tried

We experimented with:
- **LSTM-based encoder-decoder**
- **SEGAN (Speech Enhancement GAN)**  
These models often produced distorted or over-smoothed outputs, especially on very low bitrates.

---

## 🪜 Steps Followed

1. Preprocessed and compressed the MUSAN dataset using `ffmpeg` and `libopus`
2. Aligned `.wav` originals and `.opus` compressed files
3. Converted mono audio to stereo by duplicating channels
4. Trained:
   - **Wave-U-Net** on music samples
   - **HDemucs** on speech samples
5. Used **MSE Loss**, with experiments using **STFT Loss**
6. Evaluated reconstruction using:
   - **SNR (Signal-to-Noise Ratio)**
   - **SDR (Source-to-Distortion Ratio)**
   - **STOI (Short-Time Objective Intelligibility)**
   - **PESQ (Perceptual Evaluation of Speech Quality)**

---

## 📊 Results

![image](https://github.com/user-attachments/assets/23dd06f6-b9b7-4149-9f9e-93461e6557b4)


- STOI and PESQ were particularly meaningful for speech
- Music reconstruction was evaluated mostly by SNR and SDR

---

## 🌐 Gradio UI

We developed an interactive Gradio interface:
- Upload compressed `.opus` file
- View inline playback of:
  - Compressed input
  - Enhanced output
  - Original reference (if available)

---

## 🚀 Future Work

-  Generalize the models to handle varied input types (music/speech)
-  Work on Noise (technical and non-technical ) reconstruction
-  Integrate real-time streaming audio input
-  Explore diffusion-based reconstruction methods
-  Optimize models for mobile and embedded deployment
-  Incorporate language-specific models for speech enhancement

---

## 📂 Repository Structure

```bash
.
├── data/                     # Original and compressed audio samples
├── models/                   # Training and saved model checkpoints
├── scripts/                  # Data loading, training, evaluation scripts
├── gradio_ui.py             # Gradio app for testing
└── README.md                # You're here
