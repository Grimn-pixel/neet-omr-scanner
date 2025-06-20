from flask import Flask, request, render_template
import os
from utils.omr_reader import process_omr
from utils.answer_key_parser import parse_answer_key

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    omr_file = request.files['omr_sheet']
    key_file = request.files['answer_key']

    omr_path = os.path.join(UPLOAD_FOLDER, omr_file.filename)
    key_path = os.path.join(UPLOAD_FOLDER, key_file.filename)

    omr_file.save(omr_path)
    key_file.save(key_path)

    student_answers = process_omr(omr_path)
    correct_answers = parse_answer_key(key_path)

    correct = wrong = unattempted = 0

    for q in range(1, 181):
        user_ans = student_answers.get(q)
        correct_ans = correct_answers.get(q)

        if not user_ans:
            unattempted += 1
        elif user_ans == correct_ans:
            correct += 1
        else:
            wrong += 1

    total_score = (correct * 4) - (wrong * 1)

    return f"""
    <h2>Results:</h2>
    <p>Correct: {correct}</p>
    <p>Wrong: {wrong}</p>
    <p>Unattempted: {unattempted}</p>
    <p>Total Score: {total_score}</p>
    """

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
