import speech_recognition as sr
import pyttsx3
from googletrans import Translator
import google.generativeai as genai
import os
import time

# ========== CONFIGURE GEMINI ==========
# Make sure to set your API key as an environment variable
api_key = os.getenv('GOOGLE_API_KEY', "AIzaSyCfF2_zbRDjZIN8FthGEf0plkVsdCz9hLk")
genai.configure(api_key=api_key)

# Configure the model with safety settings
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

# Use the free tier model
model = genai.GenerativeModel(
    model_name='gemini-1.0-pro',  # Changed to free tier model
    generation_config=generation_config,
    safety_settings=safety_settings
)

# ========== TRANSLATION ==========
translator = Translator()

def translate_to_english(text):
    try:
        return translator.translate(text, src='hi', dest='en').text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def translate_to_hindi(text):
    try:
        return translator.translate(text, src='en', dest='hi').text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# ========== TTS ==========
engine = pyttsx3.init()
def speak_text(text, lang='en'):
    try:
        voices = engine.getProperty('voices')
        if lang == 'hi':
            for voice in voices:
                if 'hindi' in voice.name.lower() or 'hi' in voice.id.lower():
                    engine.setProperty('voice', voice.id)
                    break
        else:
            for voice in voices:
                if 'english' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS error: {e}")

# ========== VOICE INPUT ==========
def get_voice_input(language_code='en'):
    r = sr.Recognizer()
    
    try:
        # List and select microphone
        print("\nAvailable microphones:")
        mics = sr.Microphone.list_microphone_names()
        if not mics:
            print("No microphones found! Please check your microphone connection.")
            return "Error: No microphone detected"
            
        selected_mic_index = None
        selected_mic_name = "Default"
        
        for index, name in enumerate(mics):
            print(f"Microphone {index}: {name}")
            # Try to find earphone/headset microphone
            if any(keyword in name.lower() for keyword in ['headset', 'earphone', 'headphone', 'mic', 'microphone']):
                selected_mic_index = index
                selected_mic_name = name
        
        # If no earphone found or specific mic selected, use default microphone (index 0)
        if selected_mic_index is None:
             selected_mic_index = 0
             if mics:
                 selected_mic_name = mics[0]

        print(f"\nAttempting to use microphone '{selected_mic_name}' (Index: {selected_mic_index})")

        # Initialize microphone with selected device
        # Added a check for valid device index
        if selected_mic_index >= len(mics):
             print(f"Error: Selected microphone index {selected_mic_index} is out of range.")
             return "Error: Invalid microphone selected."

        with sr.Microphone(device_index=selected_mic_index) as source:
            print("\n🎤 Adjusting for ambient noise... Please wait...")
            r.adjust_for_ambient_noise(source, duration=2) # Increased duration
            print("🎤 Speak now...")
            
            try:
                # Increased timeout and phrase time limit
                audio = r.listen(source, timeout=10, phrase_time_limit=10)
                
                # Print recognition attempt
                print("Processing speech...")
                
                if language_code == 'hi':
                    text = r.recognize_google(audio, language='hi-IN')
                else:
                    text = r.recognize_google(audio, language='en-US')
                
                if text:
                    print(f"Successfully recognized: {text}")
                    return text
                else:
                    print("No speech detected.")
                    return "No speech detected. Please try again."
                    
            except sr.UnknownValueError:
                print("Speech recognition could not understand audio.")
                return "Sorry, I could not understand. Please speak clearly and try again."
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                return "Sorry, the speech recognition service is currently unavailable."
                
    except Exception as e:
        # This block catches errors during microphone setup or listening start
        print(f"An error occurred during microphone setup or listening: {str(e)}")
        return "Error in voice input. Please check your microphone connection and try again."

# ========== GEMINI RESPONSE ==========
def get_curriculum_response(query, max_retries=3):
    for attempt in range(max_retries):
        try:
            if not query or not query.strip():
                return "Please provide a valid question."
            
            # Create a chat session
            chat = model.start_chat(history=[])
            
            # Generate response
            response = chat.send_message(query)
            
            if not response or not hasattr(response, 'text'):
                return "No response generated from the model."
            
            if not response.text:
                return "The model generated an empty response."
            
            return response.text.strip()
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:  # Rate limit error
                if attempt < max_retries - 1:
                    wait_time = 15  # Wait 15 seconds before retrying
                    print(f"Rate limit reached. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue
                else:
                    return "Sorry, the service is currently busy. Please try again in a few minutes."
            else:
                print(f"Gemini API Error: {error_msg}")
                return f"Error getting response: {error_msg}"
    
    return "Failed to get response after multiple attempts. Please try again later."

# ========== MAIN ==========
def main():
    print("🧠 Shiksha Mitra - Personalized Learning Assistant (Gemini Powered)")
    
    while True:
        try:
            mode = input("\nChoose input mode (text/voice): ").strip().lower()
            if mode not in ['text', 'voice']:
                print("Invalid mode. Please choose 'text' or 'voice'.")
                continue
                
            lang = input("Choose language (en/hi): ").strip().lower()
            if lang not in ['en', 'hi']:
                print("Invalid language. Please choose 'en' or 'hi'.")
                continue

            # Get user query
            if mode == "voice":
                print("\n🎤 Please speak your question clearly...")
                query = get_voice_input(lang)
                
                if query.startswith("Error") or query.startswith("Sorry"):
                    print(f"\nVoice input failed: {query}")
                    retry = input("Would you like to try again? (y/n): ").lower()
                    if retry != 'y':
                        print("Switching to text mode...")
                        query = input("Please type your question: ").strip()
                    else:
                        continue
                
                if not query or query.isspace():
                    print("No input detected. Please try again.")
                    continue
                    
                print(f"\n🎤 You said: {query}")
            else:
                query = input("Enter your query: ").strip()
                if not query:
                    print("Please enter a valid query.")
                    continue
                print(f"\n❓ You asked: {query}")

            # Translate to English if needed
            if lang == 'hi':
                query_en = translate_to_english(query)
                print(f"Translated to English: {query_en}")
            else:
                query_en = query

            # Get response from Gemini
            print("\n🤔 Getting response from AI...")
            response_en = get_curriculum_response(query_en)
            
            if response_en.startswith("Error") or response_en.startswith("Sorry"):
                print(f"Error: {response_en}")
                continue

            # Translate back to Hindi if needed
            if lang == 'hi':
                response = translate_to_hindi(response_en)
                print(f"Translated to Hindi: {response}")
            else:
                response = response_en

            print("\n📚 Response:", response)
            speak_text(response, lang)
            
            # Ask if user wants to continue
            if input("\nDo you want to ask another question? (y/n): ").lower() != 'y':
                print("\nThank you for using Shiksha Mitra! Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            if input("Do you want to try again? (y/n): ").lower() != 'y':
                break

if __name__ == "__main__":
    main()