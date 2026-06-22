import socket
import wave
import speech_recognition as sr
import datetime
import os
import time
import threading  # <--- We are using this to handle the connections!
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
HOST = '0.0.0.0'
PORT = 8080      
SAVE_FOLDER = "recordings"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
print("Initializing LLM...")
llm = genai.GenerativeModel(
    model_name="gemini-3.1-flash-lite", 
    system_instruction="You are a voice assistant built into a tiny microcontroller named Table Bondhu build by Rafsan and Utsa. Your answers are displayed on a 160x128 pixel screen. You MUST be extremely concise. Keep every answer under 15 words. Do not use markdown formatting."
)
chat_session = llm.start_chat(history=[])
recognizer = sr.Recognizer()

def process_audio_and_chat(filename, conn):
    try:
        # 1. Google Speech-to-Text
        print("Sending audio to Google STT...")
        with sr.AudioFile(filename) as source:
            audio_data = recognizer.record(source)
            user_text = recognizer.recognize_google(audio_data)
            
        print(f"\nYou asked: {user_text}")
        
        # 2. Gemini LLM
        print("LLM is thinking...")
        response = chat_session.send_message(user_text)
        ai_answer = response.text.strip()
        print(f"AI answered: {ai_answer}")
        
        # Keep memory optimized (last 6 messages)
        while len(chat_session.history) > 6:
            chat_session.history.pop(0) 
            chat_session.history.pop(0)
            
        # 3. Send back to ESP32
        safe_answer = ai_answer.replace("\n", " ").replace("\r", "")
        conn.sendall(f"AI: {safe_answer}\n".encode())
        
    except sr.UnknownValueError:
        print("Google STT could not understand audio.")
        conn.sendall(b"System: Could not hear you.\n")
    except sr.RequestError as e:
        print(f"Google STT API Error: {e}")
        conn.sendall(b"System: API Error.\n")
    except Exception as e:
        print(f"LLM Error: {e}")
        conn.sendall(b"System: AI Error.\n")

def handle_client(conn, addr):
    """This function runs in its own dedicated background thread."""
    print(f"\n[+] New connection accepted from {addr}")
    audio_buffer = bytearray()
    receiving_audio = False
    start_time = 0
    
    while True:
        try:
            data = conn.recv(4096)
            if not data:
                break
            
            # --- START BUTTON PRESSED ---
            if b"___START___" in data:
                print("\nRecording started...")
                receiving_audio = True
                start_time = time.time()
                audio_buffer = bytearray()
                parts = data.split(b"___START___")
                audio_buffer.extend(parts[-1])
                continue
                
            if receiving_audio:
                audio_buffer.extend(data)
                
                # --- END BUTTON RELEASED ---
                if b"___END___" in audio_buffer:
                    end_time = time.time()
                    duration = end_time - start_time
                    receiving_audio = False
                    
                    clean_audio = audio_buffer.split(b"___END___")[0]
                    
                    # Dynamic hardware stopwatch fix
                    total_samples = len(clean_audio) / 2
                    actual_framerate = int(total_samples / duration) if duration > 0 else 16000
                    
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(SAVE_FOLDER, f"audio_{timestamp}.wav")
                    
                    # Save the WAV file
                    with wave.open(filename, "wb") as wf:
                        wf.setnchannels(1)       
                        wf.setsampwidth(2)       
                        wf.setframerate(actual_framerate)
                        wf.writeframes(clean_audio)
                    
                    # Process the file
                    process_audio_and_chat(filename, conn)
                    
        except ConnectionResetError:
            break
        except Exception as e:
            print(f"Connection error: {e}")
            break
            
    print(f"[-] Client {addr} disconnected.")
    conn.close()

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # This line prevents the "Address already in use" error if you restart the script quickly
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"Push-to-Talk Server listening on {HOST}:{PORT}")
        print("Waiting for ESP32...")
        
        while True:
            # 1. Wait for the ESP32 to connect
            conn, addr = s.accept()
            
            # 2. As soon as it connects, launch a new thread just for this connection!
            client_thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            client_thread.start()

if __name__ == "__main__":
    start_server()