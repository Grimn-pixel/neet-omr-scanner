
import cv2
import numpy as np
import imutils
from scipy.spatial import distance as dist
from scipy import ndimage
import math

def process_omr(image_path):
    """
    Advanced OMR sheet processing with multiple detection methods and enhanced accuracy
    Returns a dictionary with question numbers as keys and answers as values
    """
    try:
        print(f"🔍 Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return {}
        
        print(f"📐 Original image size: {image.shape}")
        
        # Advanced image preprocessing
        processed_image = advanced_preprocessing(image)
        
        # Try multiple detection methods for best results
        methods = [
            ("adaptive_threshold", detect_bubbles_adaptive),
            ("contour_analysis", detect_bubbles_contour),
            ("hough_circles", detect_bubbles_hough),
            ("template_matching", detect_bubbles_template)
        ]
        
        best_answers = {}
        best_confidence = 0
        
        for method_name, method_func in methods:
            print(f"🔄 Trying method: {method_name}")
            answers, confidence = method_func(processed_image, image)
            
            if confidence > best_confidence and len(answers) > 0:
                best_answers = answers
                best_confidence = confidence
                print(f"✅ Method {method_name} achieved confidence: {confidence:.2f}")
        
        # If no method worked well, try ensemble approach
        if best_confidence < 0.5:
            print("🔄 Trying ensemble approach...")
            best_answers = ensemble_detection(processed_image, image)
        
        print(f"✅ Successfully extracted {len(best_answers)} answers with confidence {best_confidence:.2f}")
        return best_answers
        
    except Exception as e:
        print(f"❌ Error processing OMR sheet: {e}")
        import traceback
        traceback.print_exc()
        return {}

def advanced_preprocessing(image):
    """
    Advanced image preprocessing pipeline for better bubble detection
    """
    # Resize image for optimal processing
    height, width = image.shape[:2]
    if width > 1500:
        scale = 1500 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # Correct skew if present
    corrected = correct_skew(enhanced)
    
    return corrected

def correct_skew(image):
    """
    Detect and correct skew in the image
    """
    try:
        # Detect edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:10]:  # Use first 10 lines
                angle = theta * 180 / np.pi
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle - 180)
            
            if angles:
                skew_angle = np.median(angles)
                if abs(skew_angle) > 0.5:  # Only correct if skew is significant
                    center = tuple(np.array(image.shape[1::-1]) / 2)
                    rot_mat = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                    image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], 
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    print(f"🔄 Corrected skew by {skew_angle:.2f} degrees")
    except:
        pass
    
    return image

def detect_bubbles_adaptive(processed_image, original_image):
    """
    Advanced adaptive thresholding method for bubble detection
    """
    try:
        # Multiple adaptive threshold approaches
        thresh1 = cv2.adaptiveThreshold(processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        thresh2 = cv2.adaptiveThreshold(processed_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY_INV, 15, 3)
        
        # Combine thresholds
        combined_thresh = cv2.bitwise_or(thresh1, thresh2)
        
        # Morphological operations for cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find and analyze contours
        contours = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        
        bubble_contours = filter_bubble_contours(contours, processed_image)
        answers, confidence = analyze_bubbles(bubble_contours, cleaned, processed_image)
        
        return answers, confidence
        
    except Exception as e:
        print(f"❌ Adaptive method failed: {e}")
        return {}, 0

def detect_bubbles_hough(processed_image, original_image):
    """
    Hough Circle Transform method for circular bubble detection
    """
    try:
        # Apply Hough Circle Transform
        circles = cv2.HoughCircles(processed_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                  param1=50, param2=30, minRadius=8, maxRadius=35)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Convert circles to contours for consistent processing
            bubble_contours = []
            for (x, y, r) in circles:
                # Create circular contour
                theta = np.linspace(0, 2*np.pi, 20)
                contour_x = (x + r * np.cos(theta)).astype(np.int32)
                contour_y = (y + r * np.sin(theta)).astype(np.int32)
                contour = np.column_stack((contour_x, contour_y))
                bubble_contours.append(contour)
            
            # Create threshold image for analysis
            thresh = cv2.adaptiveThreshold(processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            answers, confidence = analyze_bubbles(bubble_contours, thresh, processed_image)
            return answers, confidence * 0.9  # Slight penalty for circle detection
    
    except Exception as e:
        print(f"❌ Hough method failed: {e}")
    
    return {}, 0

def detect_bubbles_contour(processed_image, original_image):
    """
    Enhanced contour-based bubble detection with shape analysis
    """
    try:
        # Otsu's thresholding
        _, thresh = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        
        bubble_contours = filter_bubble_contours_advanced(contours, processed_image)
        answers, confidence = analyze_bubbles(bubble_contours, thresh, processed_image)
        
        return answers, confidence
        
    except Exception as e:
        print(f"❌ Contour method failed: {e}")
        return {}, 0

def detect_bubbles_template(processed_image, original_image):
    """
    Template matching approach for bubble detection
    """
    try:
        # Create circular template
        template_size = 25
        template = np.zeros((template_size, template_size), dtype=np.uint8)
        cv2.circle(template, (template_size//2, template_size//2), template_size//3, 255, -1)
        
        # Template matching
        result = cv2.matchTemplate(processed_image, template, cv2.TM_CCOEFF_NORMED)
        
        # Find matches above threshold
        threshold = 0.6
        locations = np.where(result >= threshold)
        
        # Convert to contours
        bubble_contours = []
        for pt in zip(*locations[::-1]):
            center = (pt[0] + template_size//2, pt[1] + template_size//2)
            radius = template_size//3
            
            # Create circular contour
            theta = np.linspace(0, 2*np.pi, 20)
            contour_x = (center[0] + radius * np.cos(theta)).astype(np.int32)
            contour_y = (center[1] + radius * np.sin(theta)).astype(np.int32)
            contour = np.column_stack((contour_x, contour_y))
            bubble_contours.append(contour)
        
        if bubble_contours:
            thresh = cv2.adaptiveThreshold(processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            answers, confidence = analyze_bubbles(bubble_contours, thresh, processed_image)
            return answers, confidence * 0.8  # Penalty for template matching
    
    except Exception as e:
        print(f"❌ Template method failed: {e}")
    
    return {}, 0

def filter_bubble_contours_advanced(contours, image):
    """
    Advanced contour filtering with multiple criteria
    """
    bubble_contours = []
    
    for contour in contours:
        # Basic measurements
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            continue
        
        # Shape analysis
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        extent = area / float(w * h)
        solidity = area / cv2.contourArea(cv2.convexHull(contour))
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Fit ellipse for additional analysis
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            ellipse_area = np.pi * ellipse[1][0] * ellipse[1][1] / 4
            ellipse_ratio = area / ellipse_area if ellipse_area > 0 else 0
        else:
            ellipse_ratio = 0
        
        # Enhanced filtering criteria
        if (100 <= area <= 2000 and
            0.6 <= aspect_ratio <= 1.4 and
            extent > 0.5 and
            solidity > 0.7 and
            circularity > 0.4 and
            ellipse_ratio > 0.6 and
            15 <= w <= 60 and 15 <= h <= 60):
            bubble_contours.append(contour)
    
    return bubble_contours

def filter_bubble_contours(contours, image):
    """
    Standard contour filtering for bubble detection
    """
    bubble_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            continue
        
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if (100 <= area <= 1500 and
            0.7 <= aspect_ratio <= 1.3 and
            circularity > 0.5 and
            15 <= w <= 50 and 15 <= h <= 50):
            bubble_contours.append(contour)
    
    return bubble_contours

def analyze_bubbles(bubble_contours, thresh_image, gray_image):
    """
    Advanced bubble analysis with intelligent grouping and fill detection
    """
    if len(bubble_contours) < 20:
        return {}, 0
    
    # Sort contours by position
    sorted_contours = sorted(bubble_contours, key=lambda c: cv2.boundingRect(c)[1])
    
    # Intelligent row grouping
    rows = group_bubbles_into_rows(sorted_contours)
    
    answers = {}
    confidence_scores = []
    question_num = 1
    
    for row in rows:
        if question_num > 180:
            break
        
        # Sort bubbles in row from left to right
        row_sorted = sorted(row, key=lambda c: cv2.boundingRect(c)[0])
        
        if len(row_sorted) < 4:
            continue
        
        # Take first 4 bubbles
        row_sorted = row_sorted[:4]
        
        # Analyze fill levels
        fill_scores = []
        for contour in row_sorted:
            fill_score, fill_confidence = analyze_bubble_fill(contour, thresh_image, gray_image)
            fill_scores.append((fill_score, fill_confidence))
        
        # Determine answer with confidence
        max_fill = max(fill_scores, key=lambda x: x[0])
        max_idx = fill_scores.index(max_fill)
        
        # Check if answer is confident enough
        if max_fill[0] > 0.3 and max_fill[1] > 0.6:
            answers[question_num] = chr(65 + max_idx)  # A, B, C, D
            confidence_scores.append(max_fill[1])
            print(f"Q{question_num}: {chr(65 + max_idx)} (fill: {max_fill[0]:.2f}, conf: {max_fill[1]:.2f})")
        
        question_num += 1
    
    # Calculate overall confidence
    overall_confidence = np.mean(confidence_scores) if confidence_scores else 0
    
    return answers, overall_confidence

def group_bubbles_into_rows(contours, vertical_tolerance=35):
    """
    Intelligent grouping of bubbles into rows using clustering
    """
    if not contours:
        return []
    
    # Get y-coordinates of bubble centers
    centers = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        centers.append((contour, y + h//2))
    
    # Sort by y-coordinate
    centers.sort(key=lambda x: x[1])
    
    # Group into rows
    rows = []
    current_row = [centers[0][0]]
    current_y = centers[0][1]
    
    for contour, y in centers[1:]:
        if abs(y - current_y) <= vertical_tolerance:
            current_row.append(contour)
        else:
            if len(current_row) >= 4:  # Valid row should have at least 4 bubbles
                rows.append(current_row)
            current_row = [contour]
            current_y = y
    
    # Add last row
    if len(current_row) >= 4:
        rows.append(current_row)
    
    return rows

def analyze_bubble_fill(contour, thresh_image, gray_image):
    """
    Advanced bubble fill analysis with multiple metrics
    """
    # Create mask for bubble
    mask = np.zeros(gray_image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Calculate fill metrics
    bubble_area = cv2.contourArea(contour)
    if bubble_area == 0:
        return 0, 0
    
    # Method 1: Thresholded pixel count
    filled_pixels = cv2.countNonZero(cv2.bitwise_and(thresh_image, thresh_image, mask=mask))
    fill_ratio_thresh = filled_pixels / bubble_area
    
    # Method 2: Average intensity
    mean_intensity = cv2.mean(gray_image, mask=mask)[0]
    fill_ratio_intensity = (255 - mean_intensity) / 255
    
    # Method 3: Standard deviation (filled bubbles have lower std dev)
    _, std_dev = cv2.meanStdDev(gray_image, mask=mask)
    fill_ratio_std = 1 - (std_dev[0][0] / 128)  # Normalize
    
    # Combine metrics
    fill_score = (fill_ratio_thresh * 0.5 + fill_ratio_intensity * 0.3 + fill_ratio_std * 0.2)
    
    # Calculate confidence based on how distinct the fill is
    confidence = min(1.0, fill_score * 2) if fill_score > 0.2 else 0
    
    return fill_score, confidence

def ensemble_detection(processed_image, original_image):
    """
    Ensemble approach combining multiple detection methods
    """
    try:
        print("🔄 Running ensemble detection...")
        
        # Run all methods
        methods = [
            detect_bubbles_adaptive,
            detect_bubbles_contour,
            detect_bubbles_hough
        ]
        
        all_results = []
        for method in methods:
            answers, conf = method(processed_image, original_image)
            if answers:
                all_results.append((answers, conf))
        
        if not all_results:
            return {}
        
        # Combine results with weighted voting
        final_answers = {}
        max_questions = max(len(result[0]) for result in all_results)
        
        for q in range(1, max_questions + 1):
            votes = {}
            total_weight = 0
            
            for answers, confidence in all_results:
                if q in answers:
                    answer = answers[q]
                    weight = confidence
                    votes[answer] = votes.get(answer, 0) + weight
                    total_weight += weight
            
            if votes and total_weight > 0:
                # Choose answer with highest weighted vote
                best_answer = max(votes.items(), key=lambda x: x[1])
                if best_answer[1] / total_weight > 0.5:  # Majority vote
                    final_answers[q] = best_answer[0]
        
        print(f"✅ Ensemble method produced {len(final_answers)} answers")
        return final_answers
        
    except Exception as e:
        print(f"❌ Ensemble method failed: {e}")
        return {}
