
import cv2
import numpy as np
import imutils

def process_omr(image_path):
    """
    Process OMR sheet and extract student answers
    Returns a dictionary with question numbers as keys and answers as values
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return {}
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        # Find contours for answer bubbles
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        bubble_cnts = []

        # Filter contours to find circular bubbles
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
                bubble_cnts.append(c)

        # Sort bubbles by position (top to bottom, left to right)
        bubble_cnts = sorted(bubble_cnts, key=lambda c: cv2.boundingRect(c)[1])
        
        answers = {}
        QUESTION_COUNT = 180  # NEET has 180 questions
        CHOICES = 4  # A, B, C, D

        # Process each question's answer choices
        for (q, i) in enumerate(range(0, len(bubble_cnts), CHOICES)):
            if q >= QUESTION_COUNT:
                break
                
            # Get the 4 bubbles for this question
            cnts = sorted(bubble_cnts[i:i+CHOICES], key=lambda c: cv2.boundingRect(c)[0])
            filled = None
            
            # Check which bubble is filled (darkest)
            for j, c in enumerate(cnts):
                mask = np.zeros(gray.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                total = cv2.countNonZero(cv2.bitwise_and(gray, gray, mask=mask))
                
                if filled is None or total < filled[0]:
                    filled = (total, j)
            
            # Store the answer (A=0, B=1, C=2, D=3)
            if filled:
                answers[q + 1] = chr(65 + filled[1])  # Convert to A, B, C, D
                
        return answers
        
    except Exception as e:
        print(f"Error processing OMR sheet: {e}")
        return {}
