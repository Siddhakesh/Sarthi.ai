import csv
import os
from gtts import gTTS
import tkinter as tk
from tkinter import filedialog, messagebox

# === Folder Setup ===
REPORT_FOLDER = "weekly_voice_reports"
os.makedirs(REPORT_FOLDER, exist_ok=True)

# === Language Settings ===
SUPPORTED_LANGUAGES = {
    'mr': {
        'template': "नमस्कार! आपले मूल {name} ची आठवड्याची शाळा अहवालः उपस्थिती: {attendance}, गुण: {marks}, टीप: {remarks}."
    },
    'hi': {
        'template': "नमस्ते! आपके बच्चे {name} की साप्ताहिक रिपोर्ट: उपस्थिति: {attendance}, अंक: {marks}, टिप्पणी: {remarks}."
    },
    'en': {
        'template': "Hello! Weekly report for your child {name}: Attendance: {attendance}, Marks: {marks}, Remarks: {remarks}."
    }
}

# === Generate Voice Report ===
def generate_voice_report(student):
    try:
        name = student['Name']
        attendance = student['Attendance']
        marks = student['Marks']
        remarks = student['Remarks']
        lang = student.get('Language', 'mr').lower()

        # Validate language
        if lang not in SUPPORTED_LANGUAGES:
            print(f"[⚠] Unsupported language '{lang}' for {name}, falling back to Marathi")
            lang = 'mr'

        text = SUPPORTED_LANGUAGES[lang]['template'].format(
            name=name,
            attendance=attendance,
            marks=marks,
            remarks=remarks
        )

        filename = os.path.join(REPORT_FOLDER, f"{name.replace(' ', '_')}.mp3")
        tts = gTTS(text=text, lang=lang)
        tts.save(filename)
        print(f"[✔] Audio saved: {filename}")
        return filename
    except Exception as e:
        print(f"[❌] Error generating voice report for {name}: {str(e)}")
        return None

# === Process CSV File ===
def process_csv(csv_path):
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            required_fields = ['Name', 'Attendance', 'Marks', 'Remarks']
            
            # Validate CSV structure
            missing_fields = [field for field in required_fields if field not in reader.fieldnames]
            if missing_fields:
                raise ValueError(f"Missing required fields in CSV: {', '.join(missing_fields)}")
            
            success_count = 0
            total_count = 0
            
            for row in reader:
                total_count += 1
                name = row['Name']
                print(f"[INFO] Processing: {name}")
                if generate_voice_report(row):
                    success_count += 1
            
            return success_count, total_count
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process CSV file:\n{str(e)}")
        return 0, 0

# === GUI ===
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        csv_entry.delete(0, tk.END)
        csv_entry.insert(0, file_path)

def run_generation():
    csv_path = csv_entry.get()
    if not csv_path or not os.path.exists(csv_path):
        messagebox.showerror("Error", "Please select a valid CSV file.")
        return
    
    gen_btn.config(state=tk.DISABLED, text="Processing...")
    root.update()
    
    try:
        success_count, total_count = process_csv(csv_path)
        if success_count > 0:
            messagebox.showinfo("Success", 
                f"Generated {success_count} out of {total_count} voice reports.\n"
                f"Reports are saved in: {REPORT_FOLDER}")
    finally:
        gen_btn.config(state=tk.NORMAL, text="Generate Reports")

root = tk.Tk()
root.title("Mitra Maata-Pita – Weekly Voice Report Generator")
root.geometry("600x200")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

csv_label = tk.Label(frame, text="Select Weekly Report CSV:")
csv_label.grid(row=0, column=0, sticky='w')

csv_entry = tk.Entry(frame, width=50)
csv_entry.grid(row=1, column=0)

browse_btn = tk.Button(frame, text="Browse", command=browse_file)
browse_btn.grid(row=1, column=1, padx=10)

gen_btn = tk.Button(frame, text="Generate Reports", command=run_generation, bg="green", fg="white", width=30)
gen_btn.grid(row=2, column=0, columnspan=2, pady=20)

root.mainloop()