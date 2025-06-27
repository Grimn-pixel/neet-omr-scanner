
import cv2
import numpy as np
import imutils
import fitz  # PyMuPDF for PDF processing

def extract_answer_key(image_path):
    """
    Extract answer key from uploaded image/PDF with improved accuracy
    Returns a dictionary with question numbers as keys and correct answers as values
    """
    try:
        print(f"🔍 Processing answer key: {image_path}")
        
        # Check if it's a PDF file
        if image_path.lower().endswith('.pdf'):
            return process_pdf_answer_key(image_path)
        
        # Process as image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load answer key image: {image_path}")
            return {}
        
        print(f"📐 Answer key image size: {image.shape}")
        
        # Resize if too large
        height, width = image.shape[:2]
        if width > 1200:
            scale = 1200 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Enhanced preprocessing for answer key
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try adaptive thresholding first
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        # Enhanced bubble detection for answer key
        bubble_cnts = []
        for c in cnts:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            (x, y, w, h) = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            
            # Answer key might have different bubble sizes
            if (10 <= w <= 60 and 10 <= h <= 60 and
                0.6 <= aspect_ratio <= 1.4 and
                circularity > 0.4 and
                area > 50):
                bubble_cnts.append(c)
        
        print(f"🎯 Found {len(bubble_cnts)} potential answer bubbles")
        
        if len(bubble_cnts) < 10:
            print("⚠️ Very few bubbles detected in answer key")
            return {}
        
        # Sort bubbles by position
        bubble_cnts = sorted(bubble_cnts, key=lambda c: cv2.boundingRect(c)[1])
        
        answer_key = {}
        CHOICES = 4
        
        # Group bubbles into rows
        rows = []
        current_row = []
        current_y = None
        tolerance = 25
        
        for cnt in bubble_cnts:
            (x, y, w, h) = cv2.boundingRect(cnt)
            center_y = y + h // 2
            
            if current_y is None or abs(center_y - current_y) <= tolerance:
                current_row.append((cnt, x, y, w, h))
                current_y = center_y if current_y is None else current_y
            else:
                if len(current_row) >= CHOICES:
                    rows.append(current_row)
                current_row = [(cnt, x, y, w, h)]
                current_y = center_y
        
        if len(current_row) >= CHOICES:
            rows.append(current_row)
        
        print(f"📝 Detected {len(rows)} answer key rows")
        
        # Process each row
        question_num = 1
        for row in rows:
            if question_num > 180:
                break
                
            row_sorted = sorted(row, key=lambda x: x[1])
            
            if len(row_sorted) > CHOICES:
                row_sorted = row_sorted[:CHOICES]
            elif len(row_sorted) < CHOICES:
                continue
            
            # Find filled bubble
            filled_bubble = None
            max_filled = 0
            
            for choice_idx, (cnt, x, y, w, h) in enumerate(row_sorted):
                mask = np.zeros(gray.shape, dtype="uint8")
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                
                filled_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                bubble_area = cv2.contourArea(cnt)
                
                if bubble_area > 0:
                    fill_ratio = filled_pixels / bubble_area
                    if fill_ratio > 0.2 and filled_pixels > max_filled:
                        max_filled = filled_pixels
                        filled_bubble = choice_idx
            
            if filled_bubble is not None:
                answer_key[question_num] = chr(65 + filled_bubble)
                print(f"Answer {question_num}: {chr(65 + filled_bubble)}")
            
            question_num += 1
        
        print(f"✅ Successfully extracted {len(answer_key)} answers from key")
        return answer_key
        
    except Exception as e:
        print(f"❌ Error processing answer key: {e}")
        import traceback
        traceback.print_exc()
        return {}

def process_pdf_answer_key(pdf_path):
    """
    Process PDF answer key by converting to image first
    """
    try:
        print("📄 Processing PDF answer key...")
        
        # Open PDF
        doc = fitz.open(pdf_path)
        
        # Convert first page to image
        page = doc[0]
        mat = fitz.Matrix(2.0, 2.0)  # Increase resolution
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convert to OpenCV image
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        doc.close()
        
        if image is None:
            print("❌ Failed to convert PDF to image")
            return {}
        
        # Save temporary image and process
        temp_path = pdf_path.replace('.pdf', '_temp.png')
        cv2.imwrite(temp_path, image)
        
        # Process as image
        result = extract_answer_key(temp_path)
        
        # Clean up
        import os
        try:
            os.remove(temp_path)
        except:
            pass
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing PDF answer key: {e}")
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
    
    print(f"🔍 Comparing {len(student_answers)} student answers with {len(answer_key)} answer key entries")
    
    for q in range(1, total_questions + 1):
        student_answer = student_answers.get(q)
        correct_answer = answer_key.get(q)
        
        if student_answer is None:
            unattempted += 1
        elif correct_answer is None:
            # If answer key doesn't have this question, treat as unattempted
            unattempted += 1
        elif student_answer == correct_answer:
            correct += 1
        else:
            wrong += 1
    
    print(f"📊 Comparison complete: {correct} correct, {wrong} wrong, {unattempted} unattempted")
    return correct, wrong, unattempted
