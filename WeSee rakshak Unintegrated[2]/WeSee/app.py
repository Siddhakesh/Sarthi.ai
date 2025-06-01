from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import speech_recognition as sr
import pyttsx3
from googletrans import Translator
import google.generativeai as genai
import tempfile
import os
import json
import pandas as pd
from gtts import gTTS
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure upload folders
UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = 'static/audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER

# ========== CONFIGURE GEMINI ==========
genai.configure(api_key="AIzaSyCfF2_zbRDjZIN8FthGEf0plkVsdCz9hLk")
model = genai.GenerativeModel('gemini-1.5-flash')

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
def process_audio_file(audio_file, language_code='en'):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
        if language_code == 'hi':
            return r.recognize_google(audio, language='hi-IN')
        else:
            return r.recognize_google(audio, language='en-US')
    except sr.UnknownValueError:
        return "Sorry, I could not understand."
    except sr.RequestError:
        return "API unavailable."
    except Exception as e:
        print(f"Voice input error: {e}")
        return "Error in voice input."

# ========== GEMINI RESPONSE ==========
def get_curriculum_response(query):
    try:
        response = model.generate_content(query)
        if response.text:
            return response.text
        else:
            return "No response generated from the model."
    except Exception as e:
        return f"Error from shiksha_mitra: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/shiksha_mitra.html')
def shiksha_mitra():
    return render_template('shiksha_mitra.html')

@app.route('/mata.html')
def mata():
    return render_template('mata.html')

@app.route('/process', methods=['GET', 'POST'])
def process():
    if request.method == 'POST':
        try:
            if not request.is_json:
                return Response(
                    json.dumps({
                        'error': 'Request must be JSON',
                        'status': 'error'
                    }),
                    status=400,
                    mimetype='application/json'
                )

            data = request.get_json()
            if not data:
                return Response(
                    json.dumps({
                        'error': 'No data provided',
                        'status': 'error'
                    }),
                    status=400,
                    mimetype='application/json'
                )

            mode = data.get('mode', 'text')
            lang = data.get('language', 'en')
            query = data.get('query', '')

            if not query:
                return Response(
                    json.dumps({
                        'error': 'No query provided',
                        'status': 'error'
                    }),
                    status=400,
                    mimetype='application/json'
                )

            if lang == 'hi':
                query_en = translate_to_english(query)
            else:
                query_en = query

            response_en = get_curriculum_response(query_en)

            if lang == 'hi':
                response = translate_to_hindi(response_en)
            else:
                response = response_en

            if mode == 'voice':
                speak_text(response, lang)

            return Response(
                json.dumps({
                    'query': query,
                    'response': response,
                    'status': 'success'
                }),
                status=200,
                mimetype='application/json'
            )

        except Exception as e:
            return Response(
                json.dumps({
                    'error': str(e),
                    'status': 'error'
                }),
                status=500,
                mimetype='application/json'
            )

    return Response(
        json.dumps({'error': 'Method not allowed'}),
        status=405,
        mimetype='application/json'
    )

@app.route('/process_audio', methods=['GET', 'POST'])
def process_audio():
    if request.method == 'POST':
        try:
            if 'audio' not in request.files:
                return Response(
                    json.dumps({
                        'error': 'No audio file provided',
                        'status': 'error'
                    }),
                    status=400,
                    mimetype='application/json'
                )

            audio_file = request.files['audio']
            language = request.form.get('language', 'en')

            if not audio_file:
                return Response(
                    json.dumps({
                        'error': 'Invalid audio file',
                        'status': 'error'
                    }),
                    status=400,
                    mimetype='application/json'
                )

            # Save the audio file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                audio_file.save(temp_audio.name)
                temp_audio_path = temp_audio.name

            try:
                # Process the audio file
                query = process_audio_file(temp_audio_path, language)

                if not query:
                    return Response(
                        json.dumps({
                            'error': 'Could not process audio',
                            'status': 'error'
                        }),
                        status=400,
                        mimetype='application/json'
                    )

                # Translate if needed
                if language == 'hi':
                    query_en = translate_to_english(query)
                else:
                    query_en = query

                # Get response from Gemini
                response_en = get_curriculum_response(query_en)

                # Translate response if needed
                if language == 'hi':
                    response = translate_to_hindi(response_en)
                else:
                    response = response_en

                # Speak the response
                speak_text(response, language)

                return Response(
                    json.dumps({
                        'query': query,
                        'response': response,
                        'status': 'success'
                    }),
                    status=200,
                    mimetype='application/json'
                )

            finally:
                # Clean up the temporary file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)

        except Exception as e:
            return Response(
                json.dumps({
                    'error': str(e),
                    'status': 'error'
                }),
                status=500,
                mimetype='application/json'
            )

    return Response(
        json.dumps({'error': 'Method not allowed'}),
        status=405,
        mimetype='application/json'
    )

def get_language_code(lang):
    """Convert language code to gTTS language code"""
    lang_map = {
        'en': 'en',
        'hi': 'hi',
        'mr': 'mr'
    }
    return lang_map.get(lang.lower(), 'en')

def generate_voice_report(student_data):
    """Generate voice report for a student"""
    try:
        logger.debug(f"Generating voice report for student: {student_data}")
        
        # Clean up attendance value (remove % if present)
        attendance = student_data['Attendance'].replace('%', '')
        
        # Create report text
        report_text = f"""
        Report for {student_data['Name']}.
        Attendance is {attendance} percent.
        Marks obtained are {student_data['Marks']}.
        {student_data['Remarks']}
        """

        logger.debug(f"Generated report text: {report_text}")

        # Generate unique filename
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(app.config['AUDIO_FOLDER'], filename)
        
        # Get language code
        lang_code = get_language_code(student_data['Language'])
        logger.debug(f"Using language code: {lang_code}")
        
        # Generate audio file
        tts = gTTS(text=report_text, lang=lang_code, slow=False)
        tts.save(filepath)
        
        # Verify file was created
        if os.path.exists(filepath):
            logger.debug(f"Audio file created successfully: {filepath}")
            return filename
        else:
            logger.error(f"Failed to create audio file: {filepath}")
            return None
            
    except Exception as e:
        logger.error(f"Error generating voice report: {str(e)}")
        return None

@app.route('/upload_report', methods=['POST'])
def upload_report():
    """Handle file upload and generate voice reports"""
    try:
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not file.filename.endswith(('.xlsx', '.csv')):
            logger.error(f"Invalid file format: {file.filename}")
            return jsonify({'success': False, 'error': 'Invalid file format. Please upload Excel or CSV file'})
        
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        logger.debug(f"File saved to: {filepath}")
        
        # Read the file
        if file.filename.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
        
        # Verify required columns exist
        required_columns = ['Name', 'Attendance', 'Marks', 'Remarks', 'Language']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns: {', '.join(missing_columns)}"
            logger.error(error_msg)
            return jsonify({'success': False, 'error': error_msg})
        
        logger.debug(f"File read successfully. Columns: {df.columns.tolist()}")
        logger.debug(f"Number of rows: {len(df)}")
        
        # Generate reports for each student
        reports = []
        for index, row in df.iterrows():
            logger.debug(f"Processing row {index}: {row.to_dict()}")
            student_data = row.to_dict()
            audio_file = generate_voice_report(student_data)
            
            if audio_file:
                report = {
                    'name': student_data['Name'],
                    'roll_no': f"Student {index + 1}",
                    'audio_file': audio_file
                }
                reports.append(report)
                logger.debug(f"Added report: {report}")
            else:
                logger.error(f"Failed to generate report for student: {student_data['Name']}")
        
        # Clean up uploaded file
        os.remove(filepath)
        
        if not reports:
            return jsonify({
                'success': False,
                'error': 'No reports could be generated. Please check your file format and data.'
            })
        
        logger.debug(f"Generated {len(reports)} reports")
        return jsonify({
            'success': True,
            'reports': reports
        })
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Error processing file: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True) 