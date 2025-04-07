import gradio as gr
import torch
import torchaudio 
# To handle audio data as tensors and apply transformations (e.g., resampling).
from torchaudio.transforms import Resample
import torch.nn.functional as F
import os
import soundfile as sf
#For saving audio output to .wav files.
import sys

# 🔁 Append cloned demucs repo path to import HDemucs
sys.path.append(os.path.join(os.path.dirname(__file__), "demucs"))
from demucs.hdemucs import HDemucs

# ⚙ Settings
SAMPLE_RATE = 16000
DURATION = 30
TARGET_SAMPLES = SAMPLE_RATE * DURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🧠 Load HDemucs model
model = HDemucs(sources=["speech"]).to(DEVICE)
#Creates an HDemucs instance for the "speech" source.
model.load_state_dict(torch.load("demucs_epoch_8.pth", map_location=DEVICE))
model.eval()
#Sets the model to evaluation mode (.eval() disables dropout and gradient computation).
print("✅ Model loaded successfully.")

# 🎧 Preprocess audio to stereo 30s segment
def prepare_input(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != SAMPLE_RATE: #if sr is not 16kHz
        waveform = Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)

    # Convert to mono
    waveform = waveform.mean(dim=0)

    # Pad or trim to 30 seconds
    if waveform.shape[0] > TARGET_SAMPLES:
        waveform = waveform[:TARGET_SAMPLES]
    else:
        waveform = F.pad(waveform, (0, TARGET_SAMPLES - waveform.shape[0]))

    # Duplicate to stereo format (2 channels) since HDemucs expects stereo
    stereo_wave = torch.stack([waveform, waveform], dim=0)
    return stereo_wave.unsqueeze(0).to(DEVICE)  # Return as a 3D tensor with shape (1, 2, T)

# 🔄 Reconstruct using HDemucs
def reconstruct(audio_file):
    file_path = audio_file  # it's already a string path
    input_tensor = prepare_input(file_path)

    with torch.no_grad():
        output = model(input_tensor)
        if output.ndim == 4:
            output = output.squeeze(1)
        output = output.squeeze(0).cpu()

    out_path = "reconstructed.wav"
    sf.write(out_path, output.permute(1, 0).numpy(), SAMPLE_RATE)
    return out_path


# 🌐 Gradio Interface
#create an instance of the Interface class 
# The Interface class is designed to create demos for machine learning models which accept one or more inputs, and return one or more outputs.
iface = gr.Interface(
    fn=reconstruct, #func to run when the file is uploaded :the function to wrap a user interface (UI) around
    inputs=gr.Audio(label="Upload compressed speech (.opus)", type="filepath"),
    outputs=gr.Audio(label="🔊 Reconstructed Output"),
    #inputs: File upload widget for .opus speech files.
    #outputs: Audio player for the reconstructed output.
    title="🧠 HDemucs Speech Reconstructor",
    description="Upload a compressed speech file (.opus) to reconstruct it using your trained HDemucs model."
)
# when you run iface.launch(), Gradio generates a full-fledged UI around this setup.
#Starts a local Gradio server
iface.launch(share=True)

#Gradio offers a low-level approach for designing web 
# apps with more customizable layouts and data flows with the gr.Blocks class.