import speech_recognition as sr

recognizer = sr.Recognizer()

try:
    with sr.Microphone() as source:
        print("Testing microphone... Please speak clearly after the beep.")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Better noise calibration
        print("Beep! 🎙️ Listening now... (5 seconds)")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

        print("Audio captured. Trying to recognize...")
        print("Recognizing using Google API...")
        text = recognizer.recognize_google(audio, language='en-US')  # Force English
        print(f"✅ You said: {text}")

except sr.UnknownValueError:
    print("❌ Speech Recognition could not understand the audio.")
except sr.RequestError as e:
    print(f"❌ Could not request results from Google Speech Recognition service; {e}")
except sr.WaitTimeoutError:
    print("❌ Listening timed out, no speech detected.")
except OSError as e:
    print(f"❌ Microphone error: {e}")
