import socket
import numpy as np
from faster_whisper import WhisperModel
import threading
import queue
import time
import wave
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
HOST = '0.0.0.0'
PORT = 8080
CHUNK_SECONDS = 2.5
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Configure the LLM
genai.configure(api_key=GEMINI_API_KEY)
print("Loading Whisper Model (this takes a moment)...")
whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

# Initialize Gemini with strict rules for the tiny screen
print("Initializing LLM...")
llm = genai.GenerativeModel(
    model_name="gemini-3.5-flash",
    system_instruction="You are a voice assistant built into a tiny microcontroller. Your answers are displayed on a 160x128 pixel screen. You MUST be extremely concise. Keep every answer under 15 words. Do not use markdown formatting."
)
# Start a chat session so it remembers context!
chat_session = llm.start_chat(history=[])
print("Systems Online!")

def transcription_worker(conn, audio_queue):
    """Background thread that runs Whisper AND calls the LLM."""
    while True:
        item = audio_queue.get()
        if item is None:
            break
            
        audio_bytes, actual_framerate = item
        temp_file = "temp_live.wav"
        
        with wave.open(temp_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(actual_framerate)
            wf.writeframes(audio_bytes)
            
        try:
            # 1. Transcribe the audio
            segments, info = whisper_model.transcribe(temp_file, beam_size=1, vad_filter=True)
            user_text = "".join([segment.text for segment in segments]).strip()
            
            if user_text:
                print(f"\nYou asked: {user_text}")
                
                # 2. Send the transcribed text to the LLM
                print("LLM is thinking...")
                response = chat_session.send_message(user_text)
                ai_answer = response.text.strip()
                
                print(f"AI answered: {ai_answer}")
                
                # 3. Clean the response for the ESP32 (Remove all newlines)
                safe_answer = ai_answer.replace("\n", " ").replace("\r", "")
                
                # 4. Send the LLM's answer back to the TFT screen!
                # We prepend "AI: " so it looks like a chat UI
                conn.sendall(f"AI: {safe_answer}\n".encode())
                
        except Exception as e:
            print(f"Error in AI Pipeline: {e}")
        
        audio_queue.task_done()

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"\nLive AI Server listening on {HOST}:{PORT}")
        
        while True:
            print("Waiting for ESP32 to connect...")
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr} - Streaming Live...")
                
                audio_queue = queue.Queue()
                ai_thread = threading.Thread(target=transcription_worker, args=(conn, audio_queue), daemon=True)
                ai_thread.start()
                
                audio_buffer = bytearray()
                start_time = time.time()
                
                while True:
                    try:
                        data = conn.recv(4096)
                        if not data:
                            break
                        
                        audio_buffer.extend(data)
                        current_time = time.time()
                        elapsed = current_time - start_time
                        
                        if elapsed >= CHUNK_SECONDS:
                            total_samples = len(audio_buffer) / 2
                            actual_framerate = int(total_samples / elapsed)
                            
                            audio_queue.put((audio_buffer, actual_framerate))
                            
                            audio_buffer = bytearray()
                            start_time = time.time()
                            
                    except ConnectionResetError:
                        break
                
                print("Client disconnected.")
                audio_queue.put(None)

if __name__ == "__main__":
    start_server()