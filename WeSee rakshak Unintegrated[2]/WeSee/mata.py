from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from gtts import gTTS
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = 'static/audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER

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

@app.route('/')
def index():
    """Render the main page"""
    return render_template('mata.html')

@app.route('/upload', methods=['POST'])
def upload_file():
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