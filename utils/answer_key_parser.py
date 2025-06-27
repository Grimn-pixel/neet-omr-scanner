def parse_answer_key(file_path):
    answer_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('.')
            if len(parts) == 2:
                q_no = int(parts[0])
                ans = parts[1].strip().upper()
                answer_dict[q_no] = ans
    return answer_dict
import cv2
import numpy as np
import imutils

def extract_answer_key(image_path):
    """
    Extract answer key from uploaded image/PDF
    Returns a dictionary with question numbers as keys and correct answers as values
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {}
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        # Find contours for answer bubbles
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        bubble_cnts = []

        # Filter contours to find circular bubbles
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
                bubble_cnts.append(c)

        # Sort bubbles by position
        bubble_cnts = sorted(bubble_cnts, key=lambda c: cv2.boundingRect(c)[1])
        
        answer_key = {}
        QUESTION_COUNT = 180
        CHOICES = 4

        # Process each question's answer choices
        for (q, i) in enumerate(range(0, len(bubble_cnts), CHOICES)):
            if q >= QUESTION_COUNT:
                break
                
            cnts = sorted(bubble_cnts[i:i+CHOICES], key=lambda c: cv2.boundingRect(c)[0])
            filled = None
            
            for j, c in enumerate(cnts):
                mask = np.zeros(gray.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                total = cv2.countNonZero(cv2.bitwise_and(gray, gray, mask=mask))
                
                if filled is None or total < filled[0]:
                    filled = (total, j)
            
            if filled:
                answer_key[q + 1] = chr(65 + filled[1])  # A, B, C, D
                
        return answer_key
        
    except Exception as e:
        print(f"Error processing answer key: {e}")
        return {}

def compare_answers(student_answers, answer_key):
    """
    Compare student answers with answer key
    Returns correct, wrong, and unattempted counts
    """
    correct = 0
    wrong = 0
    unattempted = 0
    
    total_questions = 180  # NEET has 180 questions
    
    for q in range(1, total_questions + 1):
        student_answer = student_answers.get(q)
        correct_answer = answer_key.get(q)
        
        if student_answer is None:
            unattempted += 1
        elif student_answer == correct_answer:
            correct += 1
        else:
            wrong += 1
    
    return correct, wrong, unattempted
