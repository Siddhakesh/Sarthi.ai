# Student Risk Detection System (Rakshak)

An AI-powered system for early identification of students at risk of dropping out, enabling timely intervention and support.

## Features

- Real-time risk assessment for students
- Multiple data inputs:
  - Attendance records
  - Academic performance
  - Behavioral engagement
  - Socio-economic data
- Risk scoring (High/Medium/Low)
- Detailed risk explanations
- Visual dashboards for educators

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Access the web interface at `http://localhost:5000`

## Data Format

The system accepts CSV files with the following columns:
- student_id: Unique identifier for each student
- attendance_rate: Percentage of classes attended
- math_score: Score in mathematics (0-100)
- english_score: Score in English (0-100)
- engagement_score: Participation score (0-100)
- family_income: Annual family income
- parent_education: Parent's education level
- siblings_count: Number of siblings

## Model Details

The system uses a combination of:
- Logistic Regression for risk prediction
- K-Means clustering for pattern discovery
- Rule-based explanation generation

## Contributing

Feel free to submit issues and enhancement requests! 