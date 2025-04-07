# End-to-End Neural Network for Low Bit-Rate Reconstruction of Sound Files

## 📌 Problem Statement

In audio communication systems, especially under low-bandwidth conditions, audio files are often compressed to very low bitrates. While this helps in transmission efficiency, it significantly degrades audio quality. Our goal is to develop a robust deep learning model that reconstructs high-fidelity audio from inputs comppressedusing different bit-rates across multiple audio types such as speech and music 

## 🎯 Motivation

Despite the practical importance of this problem, we observed that very limited research has been done on deep learning-based reconstruction of compressed audio, particularly in the ultra-low bit-rate range (e.g., 3kbps–12kbps). Most of the existing work either focuses on high-bitrate enhancement or only on speech, neglecting real-world challenges such as:

- Cross-domain audio types (speech, music, noise)
- Extreme compression artifacts
- Generalization to unseen audio content

## 🎧 What is Compression?

Compression is the process of reducing the size of audio files by removing perceptually less important components. In many cases, compression alters the dynamic range:
- **Louder parts may become softer**
- **Softer parts may become louder**
This leads to a loss in detail and quality, especially at lower bitrates (e.g., 3kbps, 6kbps).

## Dataset used 

Musan - https://www.openslr.org/17/
MUSAN is a corpus of music, speech and noise comprising approximately 109 hours of recordings. This dataset is suitable for training models for voice activity detection (VAD) and music/speech discrimination. The dataset consists of music from 5 genres, speech from twelve languages, and a wide assortment of technical and non-technical noises.

## 🛠️ Approach

We adopted a **domain-specific compression and reconstruction strategy** instead of uniformly compressing all audio types. For **speech**, we chose **3, 6, and 12 kbps**, as it can tolerate more compression while preserving intelligibility. For **music**, we used **16, 32, and 64 kbps** to preserve harmonic richness and stereo depth. **Noise** was compressed at **8, 16, and 32 kbps**, since ambient textures degrade quickly at lower rates. This selective bitrate design was based on the idea that **different audio domains encode information differently**, and we aimed to find the **lowest possible bitrate** that maintained acceptable perceptual quality.

We used the **Opus codec** for compression as it is highly versatile and efficient across a wide bitrate range, performs well on both speech and music, and offers high audio fidelity even at low bitrates making it ideal for our mixed-domain dataset.

For reconstruction, we used **Wave-U-Net** for both **music and speech**, due to its effectiveness in audio source separation and structure-aware upsampling. For **speech-specific tasks**, we also employed **HDemucs**, which leverages a hybrid encoder-decoder architecture with strong performance in speech enhancement and separation. These models were selected based on their ability to reconstruct high-quality waveforms from severely compressed inputs.

## 🔊 Why Opus?

Opus is a widely-used open, royalty-free codec designed for interactive speech and audio transmission over the internet. It offers:
- Low-latency encoding
- High audio fidelity even at low bitrates
- Efficient support for both speech and music


## 🧠 Models Used

### 🌀 Wave-U-Net (for Music)

- A convolutional encoder-decoder architecture with skip connections
- Well-suited for music source separation and audio-to-audio tasks
- Reconstructs complex musical signals by capturing temporal features at multiple scales

### 🧊 HDemucs (for Speech)

- A hybrid architecture that combines convolutional and recurrent layers
- HDemucs improves upon Wave-U-Net by adding bidirectional LSTMs and multi-scale features, making it better at capturing long-range dependencies in speech.
- Its design is more suited for handling temporal variations and fine details in compressed speech, making it more effective for speech enhancement.
- Trained to separate or enhance audio in noisy or compressed conditions
- Reconstructs stereo audio (we converted mono signals by duplicating channels)

## 🪜 System Architecture 

![image](https://github.com/user-attachments/assets/14ea219c-37fd-476a-8f15-e445995eae1d)


## 📊 Results

i) Wave-U-Net model (Music data) :

![image](https://github.com/user-attachments/assets/f19404d9-95d1-4629-9973-c60828be7fa0)

ii) Wave-U-Net model (Speech data) :

![image](https://github.com/user-attachments/assets/2a8fd39d-585b-4ee5-803a-58d005b00e66)

iii) HDemucs model :

![image](https://github.com/user-attachments/assets/23dd06f6-b9b7-4149-9f9e-93461e6557b4)



## 🌐 Gradio UI

Developed an interactive Gradio interface that allows to:
- Upload compressed `.opus` file, or record audio input real-time
- View inline playback of:
  - Compressed input
  - Enhanced output
- Flag the files for montioring, download the reconstructed .wav file

## 🌐 Streamlit UI

Developed an interactive Streamlit web-interface that allows to:
-Select the task (Speech or Music).
-Pick the model trained on a specific bitrate.
-Upload original and compressed audio files.
-View metrics and listen to:
   Original audio
   Compressed input
   Reconstructed output
   
## 🚀 Future Work

-  Generalize the models to handle varied input types (music/speech)
-  Work on Noise (technical and non-technical ) reconstruction
-  Integrate real-time streaming audio input
-  Explore diffusion-based reconstruction methods
-  Optimize models for mobile and embedded deployment
-  Incorporate language-specific models for speech enhancement

## 📂 Repository Structure

```bash
├── Code_V1.ipynb            # Includes data preparation, analysis, training, evaluation
├── Code_V2.ipynb            # Improved version of Code_V1.ipynb with comments, testing on a ted-talk audio for speech model
├── ananya-hackathon.ipynb   # Wave-U-Net model implementation
├── ML_Hackathon_UI.py       # Streamlit ui for reconstruction using wave-u-net model
├── gradio_ui.py             # Gradio app for reconstruction using HDemucs model
└── README.md                # You're here
