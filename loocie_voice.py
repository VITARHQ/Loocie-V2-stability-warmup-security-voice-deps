"""
================================================================================
LOOCIE BASE MODEL — VOICE ENGINE
File: loocie_voice.py
Location: LoocieAI_V2_Master/loocie_voice.py
Version: 1.1 (API key support)

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
from dotenv import load_dotenv

# Load .env so LOOCIE_API_KEY is available to this script
load_dotenv()

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

# API Key (matches your FastAPI middleware)
API_KEY_ENV      = "LOOCIE_API_KEY"
API_KEY_HEADER   = "X-API-Key"


# ══════════════════════════════════════════════════════════════════════════════
# TTS — TEXT TO SPEECH (using macOS built-in 'say' command)
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
        text = re.sub(r'\*+', '', text)          # markdown bold/italic
        text = re.sub(r'#+\s', '', text)         # headers
        text = re.sub(r'http\S+', 'a link', text)  # urls
        text = re.sub(r'\n+', ' ', text)         # newlines
        text = re.sub(r'[\[\]]', '', text)       # brackets
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

                rms_db = 20 * np.log10(np.sqrt(np.mean(data**2)) + 1e-10)
                if rms_db < SILENCE_DB:
                    silence_count += 1024
                    if silence_count >= silence_limit:
                        break
                else:
                    silence_count = 0

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
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        """
        Uses OpenWakeWord (if installed). If it's missing, wake word will be unavailable.
        """
        try:
            import pyaudio
            from openwakeword.model import Model

            model = Model(wakeword_models=["hey_loocie"])
            pa = pyaudio.PyAudio()

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )

            log.info("👂 Wake word listening active...")

            while self._running:
                chunk = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                audio = np.frombuffer(chunk, dtype=np.int16)

                preds = model.predict(audio)
                score = preds.get("hey_loocie", 0.0)

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

    def __init__(self):
        self.api_key = os.getenv(API_KEY_ENV, "").strip()
        if not self.api_key:
            log.warning(f"⚠️  {API_KEY_ENV} is not set. /chat will return 401 if API key auth is enabled.")

    def send(self, message: str) -> str:
        """Send message to Loocie, return her response text."""
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers[API_KEY_HEADER] = self.api_key

            response = requests.post(
                API_URL,
                json={"message": message},
                headers=headers,
                timeout=15
            )

            if response.status_code == 200:
                data  = response.json()
                reply = data.get("reply", data.get("response", ""))
                log.info(f"💬 Loocie: '{reply[:80]}...'")
                return reply

            if response.status_code == 401:
                log.error("API returned 401 Unauthorized (missing/invalid X-API-Key).")
                return "I'm secured right now. Please check the API key configuration."

            log.error(f"API error {response.status_code}: {response.text}")
            return "I'm having trouble connecting right now. Please try again."

        except requests.exceptions.ConnectionError:
            log.error("Cannot reach Loocie server. Is it running on port 8080?")
            return "I can't reach my brain right now. Please make sure the server is running."
        except requests.exceptions.Timeout:
            log.error("API request timed out.")
            return "I timed out talking to my brain. Please try again."
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
            return

        self._processing = True
        try:
            # Record
            wav_path = self.recorder.record_until_silence(max_seconds=MAX_RECORD_SECS)

            # Transcribe
            text = self.stt.transcribe(wav_path)

            # Cleanup audio file
            try:
                os.unlink(wav_path)
            except Exception:
                pass

            if not text:
                self._processing = False
                return

            # Send to API
            reply = self.api.send(text)

            # Speak
            self.tts.speak(reply)

        finally:
            self._processing = False


def main():
    voice = LooiceVoice()
    voice.start()

    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        log.info("Stopping Loocie Voice...")
        voice.wake.stop()


if __name__ == "__main__":
    main()