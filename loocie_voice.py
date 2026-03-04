"""
================================================================================
LOOCIE BASE MODEL — VOICE ENGINE
File: loocie_voice.py
Location: LoocieAI_V2_Master/loocie_voice.py
Version: 1.0

Drop this file in your project root and run it alongside the FastAPI server.
It handles wake word, speech-to-text, calls your existing /chat endpoint,
and speaks the response out loud.

Run with:
    python loocie_voice.py

Requires server running on http://127.0.0.1:8080
================================================================================
"""

import os
import io
import time
import wave
import struct
import threading
import tempfile
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import requests

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LOOCIE VOICE] %(message)s"
)
log = logging.getLogger("loocie.voice")

# ── Config ────────────────────────────────────────────────────────────────────
API_URL          = "http://127.0.0.1:8080/chat"
SAMPLE_RATE      = 16000
CHANNELS         = 1
SILENCE_DB       = -40       # dB threshold for silence detection
SILENCE_PAUSE_MS = 800       # ms of silence before stopping recording
MAX_RECORD_SECS  = 30        # max recording length
WAKE_THRESHOLD   = 0.5       # OpenWakeWord confidence threshold
WAKE_COOLDOWN    = 2.0       # seconds between wake word detections
WHISPER_MODEL    = "base"    # tiny/base/small — base is best balance


# ══════════════════════════════════════════════════════════════════════════════
# TTS — TEXT TO SPEECH (using macOS built-in 'say' command)
# Lightweight, no install needed, works offline, sounds natural on Mac
# ══════════════════════════════════════════════════════════════════════════════

class TTSEngine:
    """
    Uses macOS built-in 'say' command for TTS.
    - No install required
    - Works completely offline
    - Natural-sounding voice
    - Fast — no model loading
    
    Voice options (run 'say -v ?' in terminal to see all):
    Samantha — default, clear American female
    Karen     — Australian female  
    Moira     — Irish female
    Fiona     — Scottish female
    """
    
    VOICE = "Samantha"  # Change this to any voice from 'say -v ?'
    RATE  = 185         # Words per minute (default ~180)
    
    def speak(self, text: str):
        """Speak text using macOS say command. Non-blocking."""
        if not text or not text.strip():
            return
        threading.Thread(
            target=self._speak_thread,
            args=(text,),
            daemon=True
        ).start()
    
    def speak_blocking(self, text: str):
        """Speak text and wait until finished."""
        if not text or not text.strip():
            return
        self._speak_thread(text)
    
    def _speak_thread(self, text: str):
        try:
            # Clean text — remove markdown, brackets, excessive punctuation
            clean = self._clean_text(text)
            subprocess.run(
                ["say", "-v", self.VOICE, "-r", str(self.RATE), clean],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError as e:
            log.error(f"TTS error: {e}")
        except FileNotFoundError:
            log.error("'say' command not found. Are you on macOS?")
    
    def _clean_text(self, text: str) -> str:
        """Remove markdown and clean up for natural speech."""
        import re
        # Remove markdown bold/italic
        text = re.sub(r'\*+', '', text)
        # Remove markdown headers
        text = re.sub(r'#+\s', '', text)
        # Remove URLs
        text = re.sub(r'http\S+', 'a link', text)
        # Remove excessive newlines
        text = re.sub(r'\n+', ' ', text)
        # Remove brackets
        text = re.sub(r'[\[\]]', '', text)
        return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO RECORDER
# ══════════════════════════════════════════════════════════════════════════════

class AudioRecorder:
    """Records from microphone until silence or max duration."""
    
    def record_until_silence(self, max_seconds: int = MAX_RECORD_SECS) -> str:
        """Record audio, stop on silence. Returns path to WAV file."""
        frames        = []
        silence_count = 0
        silence_limit = int((SILENCE_PAUSE_MS / 1000) * SAMPLE_RATE)
        
        log.info("🎙️  Listening...")
        
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32'
        ) as stream:
            start = time.time()
            while time.time() - start < max_seconds:
                data, _ = stream.read(1024)
                frames.append(data)
                
                # Check for silence
                rms_db = 20 * np.log10(np.sqrt(np.mean(data**2)) + 1e-10)
                if rms_db < SILENCE_DB:
                    silence_count += 1024
                    if silence_count >= silence_limit:
                        break
                else:
                    silence_count = 0
        
        # Save to temp WAV file
        audio = np.concatenate(frames, axis=0)
        tmp   = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        
        with wave.open(tmp.name, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())
        
        return tmp.name


# ══════════════════════════════════════════════════════════════════════════════
# SPEECH TO TEXT
# ══════════════════════════════════════════════════════════════════════════════

class WhisperSTT:
    """Transcribes audio using local OpenAI Whisper."""
    
    def __init__(self):
        log.info(f"Loading Whisper model '{WHISPER_MODEL}'...")
        import whisper
        self.model = whisper.load_model(WHISPER_MODEL)
        log.info("✅ Whisper loaded.")
    
    def transcribe(self, audio_path: str) -> str:
        """Returns transcribed text or empty string."""
        try:
            result = self.model.transcribe(
                audio_path,
                language="en",
                task="transcribe"
            )
            text = result.get("text", "").strip()
            
            # Filter out hallucinations (Whisper sometimes transcribes silence)
            if len(text) < 3:
                return ""
            
            log.info(f"📝 Heard: '{text}'")
            return text
            
        except Exception as e:
            log.error(f"Transcription error: {e}")
            return ""


# ══════════════════════════════════════════════════════════════════════════════
# WAKE WORD LISTENER
# ══════════════════════════════════════════════════════════════════════════════

class WakeWordListener:
    """Listens for 'Hey Loocie' using OpenWakeWord. Free, no API key."""
    
    CHUNK_SIZE = 1280  # 80ms at 16kHz
    
    def __init__(self, callback):
        self.callback    = callback
        self._running    = False
        self._last_fired = 0.0
        self._thread     = None
    
    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._listen_loop,
            daemon=True
        )
        self._thread.start()
        log.info("👂 Wake word listener started — say 'Hey Loocie'")
    
    def stop(self):
        self._running = False
    
    def _listen_loop(self):
        try:
            import pyaudio
            from openwakeword.model import Model
            
            # Load OpenWakeWord — uses hey_jarvis as closest to hey loocie
            oww = Model(
                wakeword_models=["hey_jarvis"],
                inference_framework="onnx"
            )
            
            pa     = pyaudio.PyAudio()
            stream = pa.open(
                rate=SAMPLE_RATE,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            log.info("✅ OpenWakeWord ready.")
            
            while self._running:
                raw   = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                chunk = np.frombuffer(raw, dtype=np.int16)
                preds = oww.predict(chunk)
                
                for model_name, score in preds.items():
                    if score >= WAKE_THRESHOLD:
                        now = time.time()
                        if now - self._last_fired >= WAKE_COOLDOWN:
                            self._last_fired = now
                            log.info(f"✨ Wake word detected! (score={score:.2f})")
                            threading.Thread(
                                target=self.callback,
                                daemon=True
                            ).start()
            
            stream.stop_stream()
            stream.close()
            pa.terminate()
            
        except ImportError as e:
            log.error(f"OpenWakeWord not available: {e}")
            log.info("Push-to-talk still available via browser.")
        except Exception as e:
            log.error(f"Wake word error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# LOOCIE API CALLER
# ══════════════════════════════════════════════════════════════════════════════

class LooiceChatAPI:
    """Sends messages to your existing Loocie FastAPI /chat endpoint."""
    
    def send(self, message: str) -> str:
        """Send message to Loocie, return her response text."""
        try:
            response = requests.post(
                API_URL,
                json={"message": message},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data  = response.json()
                reply = data.get("reply", data.get("response", ""))
                log.info(f"💬 Loocie: '{reply[:80]}...'")
                return reply
            else:
                log.error(f"API error {response.status_code}: {response.text}")
                return "I'm having trouble connecting right now. Please try again."
                
        except requests.exceptions.ConnectionError:
            log.error("Cannot reach Loocie server. Is it running on port 8080?")
            return "I can't reach my brain right now. Please make sure the server is running."
        except Exception as e:
            log.error(f"API call failed: {e}")
            return "Something went wrong. Please try again."


# ══════════════════════════════════════════════════════════════════════════════
# MAIN VOICE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class LooiceVoice:
    """
    Main voice engine. Ties everything together.
    
    Flow:
      Wake word detected
        → Record until silence
        → Whisper transcribes
        → Send to /chat API
        → Speak response with macOS TTS
    """
    
    def __init__(self):
        log.info("Initialising Loocie Voice Engine...")
        
        self.tts      = TTSEngine()
        self.recorder = AudioRecorder()
        self.stt      = WhisperSTT()
        self.api      = LooiceChatAPI()
        self.wake     = WakeWordListener(callback=self._on_wake_word)
        
        self._processing = False  # Prevent overlapping activations
        
        log.info("✅ Loocie Voice Engine ready.")
    
    def start(self):
        """Start listening for wake word."""
        # Greet on startup
        self.tts.speak_blocking("Loocie voice is online. Say hey Loocie to get started.")
        self.wake.start()
        
        log.info("")
        log.info("=" * 50)
        log.info("  LOOCIE VOICE ACTIVE")
        log.info("  Say 'Hey Loocie' to activate")
        log.info("  Press Ctrl+C to stop")
        log.info("=" * 50)
        log.info("")
    
    def _on_wake_word(self):
        """Called when wake word is detected."""
        if self._processing:
            return  # Already handling a request
        
        self._processing = True
        
        try:
            # Acknowledge
            self.tts.speak("Yes?")
            
            # Record user's message
            audio_path = self.recorder.record_until_silence()
            
            # Transcribe
            text = self.stt.transcribe(audio_path)
            
            # Clean up audio file
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            
            if not text:
                self.tts.speak("I didn't catch that. Try again.")
                return
            
            # Get Loocie's response
            log.info(f"Sending to Loocie: '{text}'")
            response = self.api.send(text)
            
            # Speak the response
            if response:
                self.tts.speak(response)
                
        except Exception as e:
            log.error(f"Voice processing error: {e}")
            self.tts.speak("Something went wrong. Please try again.")
        finally:
            self._processing = False
    
    def shutdown(self):
        self.wake.stop()
        self.tts.speak_blocking("Loocie voice going offline.")
        log.info("Loocie Voice Engine shut down.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # Check server is reachable before starting
    log.info("Checking Loocie server connection...")
    try:
        r = requests.get("http://127.0.0.1:8080/health", timeout=5)
        if r.status_code == 200:
            log.info("✅ Loocie server is online.")
        else:
            log.warning(f"Server responded with status {r.status_code}")
    except Exception:
        log.error("❌ Cannot reach Loocie server at http://127.0.0.1:8080")
        log.error("   Start the server first with:")
        log.error("   python -m uvicorn app.main:app --reload --port 8080")
        sys.exit(1)
    
    # Start voice engine
    engine = LooiceVoice()
    engine.start()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        engine.shutdown()
        print("\nLoocie voice offline.")