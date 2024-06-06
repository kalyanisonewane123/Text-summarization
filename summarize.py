import streamlit as st
from transformers import pipeline
import torch
import librosa
import io
import google.generativeai as genai
import os
from dotenv import load_dotenv
# from fpdf import FPDF

def convert_bytes_to_array(audio_bytes):
    audio_bytes = io.BytesIO(audio_bytes)
    audio, sample_rate = librosa.load(audio_bytes)
    return audio

def transcribe_audio(audio_bytes):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-base",
        chunk_length_s=0,
        device=device,
    )

    audio_array = convert_bytes_to_array(audio_bytes)
    prediction = pipe(audio_array, batch_size=1)["text"]
    return prediction

# def save_text_to_pdf(text, output_path="transcription.pdf"):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
#     for line in text.split('\n'):
#         pdf.multi_cell(0, 10, line)
#     pdf.output(output_path)

st.title("Audio Transcription")

# Initialize session state variables
if "transcription" not in st.session_state:
    st.session_state.transcription = ""

audio_file = st.file_uploader("Please upload an audio file", type=["mp3", "wav"])

if st.button("Transcribe"):
    if audio_file is not None:
        st.audio(audio_file, format='audio/mp3')
        st.session_state.transcription = transcribe_audio(audio_file.read())
        st.text_area("Transcribed Text", st.session_state.transcription, height=200)

load_dotenv()
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')
prompt = """please summarize the provided text, make sure it covers all details"""

if st.button("Summarize"):
    def get_gemini_response(input_text, prompt):
        response = model.generate_content([input_text, prompt])
        return response.text  # Accessing the generated text from the response

    if st.session_state.transcription:
        response = get_gemini_response(st.session_state.transcription, prompt)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.error("Please transcribe the audio first.")
