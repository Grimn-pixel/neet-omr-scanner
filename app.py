import os
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        # Make sure uploads folder exists
        upload_folder = 'uploads'
        os.makedirs(upload_folder, exist_ok=True)

        # Get uploaded files
        omr_file = request.files['omr_pdf']
        answer_file = request.files['answer_key_pdf']

        # Save files
        omr_path = os.path.join(upload_folder, omr_file.filename)
        answer_path = os.path.join(upload_folder, answer_file.filename)
        omr_file.save(omr_path)
        answer_file.save(answer_path)

        # TEMP: Hardcoded result values for testing
        correct = 4
        wrong = 2
        unattempted = 34
        score = correct * 4 - wrong

        return render_template('result.html',
                               correct=correct,
                               wrong=wrong,
                               unattempted=unattempted,
                               score=score)
    except Exception as e:
        return f"❌ Error during evaluation: {str(e)}"

if name == '__main__':
    app.run(debug=True)
