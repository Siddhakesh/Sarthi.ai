from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import joblib
import os

app = Flask(__name__)
CORS(app)

# Define feature lists based on student_dataset.csv
NUMERICAL_FEATURES = [
    'attendance_rate_last_month',
    'attendance_rate_last_3_months',
    'avg_assignment_score',
    'test_score_math',
    'test_score_english',
    'final_exam_score_last_term',
    'quiz_participation_rate',
    'total_logins_per_week',
    'homework_submission_rate',
    'disciplinary_actions',
    'number_of_siblings',
    'distance_from_school_km'
]

CATEGORICAL_FEATURES = [
    'grade_level',
    'class_participation_level',
    'parent_occupation',
    'family_income_bracket',
    'has_learning_device'
]

TARGET_FEATURE = 'dropout_label'

# Initialize models and preprocessors
model = None
kmeans = None
preprocessor = None
feature_names = None
student_dataset = None  # Add this to store the dataset

def initialize_models_and_preprocessor():
    global model, kmeans, preprocessor, feature_names, student_dataset
    if os.path.exists('models/model.joblib'):
        try:
            model = joblib.load('models/model.joblib')
            kmeans = joblib.load('models/kmeans.joblib')
            preprocessor = joblib.load('models/preprocessor.joblib')
            feature_names = joblib.load('models/feature_names.joblib')
            if os.path.exists('data/student_dataset.csv'):
                student_dataset = pd.read_csv('data/student_dataset.csv')
            print("Models, preprocessor, and dataset loaded successfully.")
        except Exception as e:
            print(f"Error loading models and preprocessor: {e}")
            # Re-initialize if loading fails
            os.makedirs('models', exist_ok=True)
            model = LogisticRegression(random_state=42)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            feature_names = None
            student_dataset = None
            print("Initialized new models due to loading error.")
    else:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Initialize new models
        model = LogisticRegression(random_state=42)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        feature_names = None
        student_dataset = None
        print("Initialized new models (no existing files found).")

def get_risk_level(probability):
    if probability < 0.4:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"

def generate_explanation(student_data):
    explanations = []

    # Convert values to float for comparison, provide default if key is missing
    attendance_rate_last_month = float(student_data.get('attendance_rate_last_month', 100))
    test_score_math = float(student_data.get('test_score_math', 100))
    test_score_english = float(student_data.get('test_score_english', 100))
    engagement_score = float(student_data.get('engagement_score', 100)) # Keep engagement_score if it's mapped from new features
    homework_submission_rate = float(student_data.get('homework_submission_rate', 100))
    disciplinary_actions = float(student_data.get('disciplinary_actions', 0))
    total_logins_per_week = float(student_data.get('total_logins_per_week', 100))


    if attendance_rate_last_month < 70:
        explanations.append(f"Low attendance rate last month ({attendance_rate_last_month}%) 부족")

    if test_score_math < 40:
        explanations.append(f"Poor performance in Math ({test_score_math}%) 부족")

    if test_score_english < 40:
        explanations.append(f"Poor performance in English ({test_score_english}%) 부족")

    if engagement_score < 50: # Keep engagement_score if it's mapped from new features
        # This part might need adjustment based on how engagement_score is derived from new features
        # For now, keeping the old check but using .get()
        explanations.append(f"Low engagement score ({engagement_score}%) 부족")

    if homework_submission_rate < 60:
         explanations.append(f"Low homework submission rate ({homework_submission_rate}%) 부족")

    if disciplinary_actions > 0:
         explanations.append(f"Recorded disciplinary actions ({disciplinary_actions}) 위험")

    if total_logins_per_week < 3:
         explanations.append(f"Low weekly login count ({total_logins_per_week}) 위험")


    return " + ".join(explanations) if explanations else "No significant risk factors identified"

@app.route('/rakshak')
def rakshak():
    return render_template('saathi.html')

@app.route('/rakshak/saathi')
def saathi():
    return render_template('saathi.html')

@app.route('/rakshak/students')
def students():
    return render_template('students.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    global model, preprocessor, feature_names, kmeans
    if model is None or preprocessor is None or feature_names is None or kmeans is None:
        return jsonify({'error': 'Models and preprocessor not trained or loaded.'}), 500

    try:
        data = request.get_json()
        student_data_dict = data.copy()

        # Create a DataFrame from the incoming data
        # Ensure column order matches expected features for preprocessing
        # input_df = pd.DataFrame([student_data_dict]) # Removed

        # Select only the columns that are used as features
        # Handle potential missing columns by adding them with a default value (e.g., 0 or appropriate for category)
        # For simplicity, let's ensure the input_df only contains our defined features
        # And handle missing numerical features with 0, missing categorical with a placeholder if necessary (though one-hot handles NaNs)

        processed_input_data = {}
        # for col in NUMERICAL_FEATURES: # Removed
        #      processed_input_data[col] = [float(input_df.get(col, 0).iloc[0]) if pd.notna(input_df.get(col, 0).iloc[0]) else 0] # Removed
        # for col in CATEGORICAL_FEATURES: # Removed
        #      processed_input_data[col] = [input_df.get(col, 'Unknown').iloc[0] if pd.notna(input_df.get(col, 'Unknown').iloc[0]) else 'Unknown'] # Removed

        for col in NUMERICAL_FEATURES:
             value = student_data_dict.get(col, 0) # Get value from dict, default to 0
             # Ensure value is float, handle None/NaN
             processed_input_data[col] = [float(value) if value is not None and pd.notna(value) else 0]

        for col in CATEGORICAL_FEATURES:
             value = student_data_dict.get(col, 'Unknown') # Get value from dict, default to 'Unknown'
             # Ensure value is string, handle None/NaN
             processed_input_data[col] = [str(value) if value is not None and pd.notna(value) else 'Unknown']


        processed_input_df = pd.DataFrame(processed_input_data)

        # Apply the fitted preprocessor
        scaled_features = preprocessor.transform(processed_input_df)

        # Convert to DataFrame to ensure column names are retained after preprocessing (especially one-hot encoding)
        # This part depends on whether the preprocessor outputs a numpy array or a DataFrame
        # Assuming preprocessor outputs a numpy array based on typical sklearn behavior
        # We need to reconstruct the DataFrame with correct feature names

        # Get the feature names after preprocessing (especially after one-hot encoding)
        # This requires a bit more effort with ColumnTransformer
        # We should already have feature_names saved from training

        if feature_names is None:
             # This shouldn't happen if models are loaded/trained, but as a fallback
             # Try to infer feature names from preprocessor if possible
             # This is complex, so we'll raise an error or use a basic naming if necessary
             # For now, assume feature_names is loaded correctly
             return jsonify({'error': 'Feature names not available. Model might not be trained correctly.'}), 500


        if isinstance(scaled_features, np.ndarray):
             # If output is a numpy array, create DataFrame with saved feature names
             scaled_features_df = pd.DataFrame(scaled_features, columns=feature_names)
        # else: # Removed
        #      # If output is a DataFrame (e.g., from pandas input and output='pandas' in CT) # Removed
        #      feature_names = X_scaled.columns.tolist() # Removed
        #      scaled_features_df = X_scaled # Removed


        # Get risk probability
        risk_probability = model.predict_proba(scaled_features_df)[0][1]
        risk_level = get_risk_level(risk_probability)

        # Get cluster assignment
        # KMeans also expects features in the correct order and format
        cluster = kmeans.predict(scaled_features_df)[0]

        # Generate explanation using the original data dictionary (or processed if needed)
        explanation = generate_explanation(student_data_dict) # Using original dict for easier access to raw values

        return jsonify({
            'risk_level': risk_level,
            'risk_probability': float(risk_probability),
            'cluster': int(cluster),
            'explanation': explanation
        })

    except Exception as e:
        # Log the error for debugging
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/students', methods=['GET'])
def get_students():
    global student_dataset, model, preprocessor, feature_names
    if student_dataset is None:
        print("No student dataset available")
        return jsonify({'error': 'No student dataset available. Please upload a dataset first.'}), 400

    if model is None or preprocessor is None or feature_names is None:
        print("Models or preprocessor not loaded")
        return jsonify({'error': 'Models or preprocessor not trained or loaded.'}), 500

    try:
        students_data = []
        for index, student in student_dataset.iterrows():
            student_data_dict = student.to_dict()

            # Prepare data for prediction for this single student
            processed_input_data = {}
            for col in NUMERICAL_FEATURES:
                 value = student_data_dict.get(col, 0) # Get value from dict, default to 0
                 processed_input_data[col] = [float(value) if value is not None and pd.notna(value) else 0]

            for col in CATEGORICAL_FEATURES:
                 value = student_data_dict.get(col, 'Unknown') # Get value from dict, default to 'Unknown'
                 processed_input_data[col] = [str(value) if value is not None and pd.notna(value) else 'Unknown']

            processed_input_df = pd.DataFrame(processed_input_data)

            # Apply the fitted preprocessor
            # Ensure columns match feature_names after preprocessing
            try:
                scaled_features = preprocessor.transform(processed_input_df)
                if isinstance(scaled_features, np.ndarray):
                    scaled_features_df = pd.DataFrame(scaled_features, columns=feature_names)
                else:
                     scaled_features_df = scaled_features # Should ideally be a DataFrame if preprocessor output is set to pandas

            except Exception as preproc_e:
                 print(f"Preprocessor error for student {student_data_dict.get('name','N/A')}: {preproc_e}")
                 # Handle preprocessing errors - maybe skip student or provide default data
                 continue # Skip this student if preprocessing fails


            # Get risk probability, level, and explanation
            try:
                 risk_probability = model.predict_proba(scaled_features_df)[0][1]
                 risk_level = get_risk_level(risk_probability)
                 explanation = generate_explanation(student_data_dict)
            except Exception as predict_e:
                 print(f"Prediction error for student {student_data_dict.get('name','N/A')}: {predict_e}")
                 # Handle prediction errors
                 risk_probability = 0.0
                 risk_level = "Unknown"
                 explanation = "Could not generate risk details."

            students_data.append({
                'name': student_data_dict.get('name', 'N/A'),
                'student_id': student_data_dict.get('student_id', 'N/A'),
                'grade_level': student_data_dict.get('grade_level', 'N/A'),
                'test_score_math': student_data_dict.get('test_score_math', 'N/A'),
                'test_score_english': student_data_dict.get('test_score_english', 'N/A'),
                'risk_probability': float(risk_probability), # Ensure it's serializable
                'risk_level': risk_level,
                'explanation': explanation
            })

        print(f"Prepared data for {len(students_data)} students for dashboard")
        return jsonify(students_data)
    except Exception as e:
        print(f"Error getting students data for dashboard: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/student/<student_name>', methods=['GET'])
def get_student_details(student_name):
    global student_dataset
    if student_dataset is None:
        return jsonify({'error': 'No student dataset available. Please upload a dataset first.'}), 400
    
    # Find the student in the dataset
    student = student_dataset[student_dataset['student_name'] == student_name]
    if student.empty:
        return jsonify({'error': 'Student not found'}), 404
    
    # Get student details
    student_data = student.iloc[0].to_dict()
    
    # Generate improvement suggestions
    suggestions = []
    
    # Check academic performance
    if student_data['test_score_math'] < 70:
        suggestions.append({
            'subject': 'Mathematics',
            'current_score': student_data['test_score_math'],
            'suggestion': 'Consider additional math practice and tutoring'
        })
    
    if student_data['test_score_english'] < 70:
        suggestions.append({
            'subject': 'English',
            'current_score': student_data['test_score_english'],
            'suggestion': 'Focus on reading comprehension and writing skills'
        })
    
    # Check attendance
    if student_data['attendance_rate_last_month'] < 90:
        suggestions.append({
            'area': 'Attendance',
            'current_rate': student_data['attendance_rate_last_month'],
            'suggestion': 'Improve regular attendance to maintain academic progress'
        })
    
    # Check homework submission
    if student_data['homework_submission_rate'] < 80:
        suggestions.append({
            'area': 'Homework',
            'current_rate': student_data['homework_submission_rate'],
            'suggestion': 'Submit homework assignments regularly to improve grades'
        })
    
    # Check class participation
    if student_data['class_participation_level'] == 'Low':
        suggestions.append({
            'area': 'Class Participation',
            'current_level': student_data['class_participation_level'],
            'suggestion': 'Increase active participation in class discussions'
        })
    
    return jsonify({
        'student_details': student_data,
        'improvement_suggestions': suggestions
    })

@app.route('/api/train', methods=['POST'])
def train():
    global model, preprocessor, feature_names, kmeans, student_dataset
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read the training data
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format. Please upload CSV or Excel files.'}), 400
        
        print(f"Loaded dataset with {len(df)} rows and columns: {df.columns.tolist()}")
        
        # Save the dataset
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/student_dataset.csv', index=False)
        student_dataset = df
        
        # Separate features and target
        if TARGET_FEATURE not in df.columns:
            return jsonify({'error': f'Training data must contain a "{TARGET_FEATURE}" column.'}), 400
            
        X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        y = df[TARGET_FEATURE]

        # Create a column transformer to apply different preprocessing to different columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), NUMERICAL_FEATURES),
                ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
            ])
        
        # Fit and transform the training data
        X_scaled = preprocessor.fit_transform(X)

        # Get the feature names after preprocessing
        if isinstance(X_scaled, np.ndarray):
            numeric_feature_names = NUMERICAL_FEATURES
            try:
                onehot_features = preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES)
                categorical_feature_names = list(onehot_features)
            except AttributeError:
                categorical_feature_names = []
                for i, col in enumerate(CATEGORICAL_FEATURES):
                    pass

            feature_names = numeric_feature_names + categorical_feature_names
        else:
            feature_names = X_scaled.columns.tolist()

        # Train models
        model.fit(X_scaled, y)
        kmeans.fit(X_scaled)
        
        # Save models and preprocessor
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/model.joblib')
        joblib.dump(kmeans, 'models/kmeans.joblib')
        joblib.dump(preprocessor, 'models/preprocessor.joblib')
        joblib.dump(feature_names, 'models/feature_names.joblib')

        # Calculate risk statistics
        risk_probabilities = model.predict_proba(X_scaled)[:, 1]
        total_students = len(df)
        high_risk = sum(1 for p in risk_probabilities if p >= 0.7)
        medium_risk = sum(1 for p in risk_probabilities if 0.4 <= p < 0.7)
        low_risk = sum(1 for p in risk_probabilities if p < 0.4)
        
        print(f"Training completed. Total students: {total_students}, High risk: {high_risk}, Medium risk: {medium_risk}, Low risk: {low_risk}")
        
        return jsonify({
            'message': 'Models and preprocessor trained and saved successfully.',
            'total_students': total_students,
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'low_risk': low_risk
        })
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    initialize_models_and_preprocessor()
    app.run(debug=True) 