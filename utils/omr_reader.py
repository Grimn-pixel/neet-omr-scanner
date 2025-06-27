
import cv2
import numpy as np
import imutils
from scipy.spatial import distance as dist

def process_omr(image_path):
    """
    Process OMR sheet and extract student answers with improved accuracy
    Returns a dictionary with question numbers as keys and answers as values
    """
    try:
        print(f"🔍 Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return {}
        
        print(f"📐 Original image size: {image.shape}")
        
        # Resize image for better processing if too large
        height, width = image.shape[:2]
        if width > 1200:
            scale = 1200 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"📏 Resized to: {image.shape}")
        
        # Enhanced preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding for better bubble detection
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        print("✅ Image preprocessing completed")
        
        # Find contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        print(f"🔍 Found {len(cnts)} total contours")
        
        # Enhanced bubble detection
        bubble_cnts = []
        for c in cnts:
            # Calculate contour properties
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            (x, y, w, h) = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            
            # More refined bubble criteria
            if (15 <= w <= 50 and 15 <= h <= 50 and  # Size constraints
                0.7 <= aspect_ratio <= 1.3 and        # Nearly square/circular
                circularity > 0.5 and                 # Circular enough
                area > 100):                          # Minimum area
                bubble_cnts.append(c)
        
        print(f"🎯 Found {len(bubble_cnts)} potential bubbles")
        
        if len(bubble_cnts) < 20:  # Too few bubbles detected
            print("⚠️ Warning: Very few bubbles detected, trying alternative method")
            return process_omr_alternative(image_path)
        
        # Sort bubbles by position (top to bottom, left to right)
        def sort_contours(cnts, method="top-to-bottom"):
            reverse = False
            i = 0
            if method == "right-to-left" or method == "bottom-to-top":
                reverse = True
            if method == "top-to-bottom" or method == "bottom-to-top":
                i = 1
            boundingBoxes = [cv2.boundingRect(c) for c in cnts]
            (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                               key=lambda b: b[1][i], reverse=reverse))
            return cnts, boundingBoxes
        
        bubble_cnts, _ = sort_contours(bubble_cnts, method="top-to-bottom")
        
        answers = {}
        CHOICES = 4  # A, B, C, D
        question_num = 1
        
        # Group bubbles into rows (questions)
        rows = []
        current_row = []
        current_y = None
        tolerance = 30  # Vertical tolerance for grouping bubbles in same row
        
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
        
        # Don't forget the last row
        if len(current_row) >= CHOICES:
            rows.append(current_row)
        
        print(f"📝 Detected {len(rows)} question rows")
        
        # Process each row (question)
        for row_idx, row in enumerate(rows):
            if question_num > 180:  # NEET limit
                break
                
            # Sort bubbles in row from left to right
            row_sorted = sorted(row, key=lambda x: x[1])  # Sort by x coordinate
            
            # Take only first 4 bubbles if more than 4 detected
            if len(row_sorted) > CHOICES:
                row_sorted = row_sorted[:CHOICES]
            elif len(row_sorted) < CHOICES:
                continue  # Skip incomplete rows
            
            # Analyze each bubble to find the filled one
            filled_bubble = None
            min_ratio = float('inf')
            
            for choice_idx, (cnt, x, y, w, h) in enumerate(row_sorted):
                # Create mask for this bubble
                mask = np.zeros(gray.shape, dtype="uint8")
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                
                # Calculate fill ratio
                bubble_area = cv2.contourArea(cnt)
                if bubble_area == 0:
                    continue
                    
                # Count filled pixels in the bubble
                filled_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                fill_ratio = filled_pixels / bubble_area
                
                # A filled bubble should have a high fill ratio
                if fill_ratio > 0.3 and fill_ratio < min_ratio:
                    min_ratio = fill_ratio
                    filled_bubble = choice_idx
            
            # Only record answer if we found a clearly filled bubble
            if filled_bubble is not None and min_ratio > 0.3:
                answers[question_num] = chr(65 + filled_bubble)  # Convert to A, B, C, D
                print(f"Q{question_num}: {chr(65 + filled_bubble)} (fill ratio: {min_ratio:.2f})")
            
            question_num += 1
        
        print(f"✅ Successfully extracted {len(answers)} answers")
        return answers
        
    except Exception as e:
        print(f"❌ Error processing OMR sheet: {e}")
        import traceback
        traceback.print_exc()
        return {}

def process_omr_alternative(image_path):
    """
    Alternative OMR processing method for difficult images
    """
    try:
        print("🔄 Using alternative processing method...")
        
        image = cv2.imread(image_path)
        if image is None:
            return {}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try different thresholding approach
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        # Less strict bubble criteria for difficult images
        bubble_cnts = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            aspect_ratio = w / float(h)
            
            if (10 <= w <= 60 and 10 <= h <= 60 and
                0.6 <= aspect_ratio <= 1.4 and
                area > 50):
                bubble_cnts.append(c)
        
        print(f"🎯 Alternative method found {len(bubble_cnts)} bubbles")
        
        # If still too few, return empty (will use demo data)
        if len(bubble_cnts) < 10:
            print("⚠️ Alternative method also failed")
            return {}
        
        # Simple processing for alternative method
        bubble_cnts = sorted(bubble_cnts, key=lambda c: cv2.boundingRect(c)[1])
        
        answers = {}
        for i in range(0, min(len(bubble_cnts), 720), 4):  # Max 180 questions * 4 choices
            if i + 3 >= len(bubble_cnts):
                break
                
            question_bubbles = bubble_cnts[i:i+4]
            question_bubbles = sorted(question_bubbles, key=lambda c: cv2.boundingRect(c)[0])
            
            # Find darkest bubble
            darkest_idx = None
            max_filled = 0
            
            for j, bubble in enumerate(question_bubbles):
                mask = np.zeros(gray.shape, dtype="uint8")
                cv2.drawContours(mask, [bubble], -1, 255, -1)
                filled = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                
                if filled > max_filled:
                    max_filled = filled
                    darkest_idx = j
            
            if darkest_idx is not None and max_filled > 50:
                question_num = (i // 4) + 1
                answers[question_num] = chr(65 + darkest_idx)
        
        print(f"✅ Alternative method extracted {len(answers)} answers")
        return answers
        
    except Exception as e:
        print(f"❌ Alternative method failed: {e}")
        return {}
