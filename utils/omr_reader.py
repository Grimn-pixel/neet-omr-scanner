import cv2
import numpy as np
import imutils

def process_omr(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    bubble_cnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
            bubble_cnts.append(c)

    bubble_cnts = sorted(bubble_cnts, key=lambda c: cv2.boundingRect(c)[1])
    answers = {}
    QUESTION_COUNT = 180
    CHOICES = 4

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
            answers[q + 1] = chr(65 + filled[1])
    return answers
