
import cv2
import numpy as np
import imutils
import fitz  # PyMuPDF for PDF processing
from scipy.spatial import distance as dist
from scipy import ndimage

def extract_answer_key(image_path):
    """
    Advanced answer key extraction with multiple processing methods
    Returns a dictionary with question numbers as keys and correct answers as values
    """
    try:
        print(f"🔍 Processing answer key: {image_path}")
        
        # Handle PDF files
        if image_path.lower().endswith('.pdf'):
            return process_pdf_answer_key_advanced(image_path)
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load answer key image: {image_path}")
            return {}
        
        print(f"📐 Answer key image size: {image.shape}")
        
        # Advanced preprocessing
        processed_image = advanced_answer_key_preprocessing(image)
        
        # Try multiple detection methods
        methods = [
            ("enhanced_adaptive", extract_with_adaptive_method),
            ("morphological", extract_with_morphological_method),
            ("contour_analysis", extract_with_contour_analysis),
            ("pattern_recognition", extract_with_pattern_recognition)
        ]
        
        best_answers = {}
        best_confidence = 0
        
        for method_name, method_func in methods:
            print(f"🔄 Trying answer key method: {method_name}")
            answers, confidence = method_func(processed_image, image)
            
            if confidence > best_confidence and len(answers) > 0:
                best_answers = answers
                best_confidence = confidence
                print(f"✅ Method {method_name} achieved confidence: {confidence:.2f}")
        
        # Ensemble approach if no single method is confident enough
        if best_confidence < 0.6:
            print("🔄 Trying ensemble approach for answer key...")
            best_answers = ensemble_answer_extraction(processed_image, image)
        
        print(f"✅ Successfully extracted {len(best_answers)} answers from key")
        return best_answers
        
    except Exception as e:
        print(f"❌ Error processing answer key: {e}")
        import traceback
        traceback.print_exc()
        return {}

def advanced_answer_key_preprocessing(image):
    """
    Advanced preprocessing pipeline specifically for answer keys
    """
    # Resize for optimal processing
    height, width = image.shape[:2]
    if width > 1600:
        scale = 1600 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Advanced noise reduction
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Correct any skew
    corrected = correct_document_skew(enhanced)
    
    # Additional sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(corrected, -1, kernel)
    
    return sharpened

def correct_document_skew(image):
    """
    Advanced skew correction for answer key documents
    """
    try:
        # Edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Morphological operations to connect text lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            if angles:
                # Filter out outliers
                angles = np.array(angles)
                median_angle = np.median(angles)
                filtered_angles = angles[np.abs(angles - median_angle) < 10]
                
                if len(filtered_angles) > 0:
                    skew_angle = np.mean(filtered_angles)
                    
                    if abs(skew_angle) > 0.5:
                        center = tuple(np.array(image.shape[1::-1]) / 2)
                        rot_mat = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                        image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], 
                                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                        print(f"🔄 Corrected answer key skew by {skew_angle:.2f} degrees")
    except:
        pass
    
    return image

def extract_with_adaptive_method(processed_image, original_image):
    """
    Enhanced adaptive thresholding method for answer keys
    """
    try:
        # Multiple adaptive thresholds with different parameters
        thresh_methods = [
            cv2.adaptiveThreshold(processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2),
            cv2.adaptiveThreshold(processed_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY_INV, 15, 4),
            cv2.adaptiveThreshold(processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 21, 6)
        ]
        
        # Combine thresholds
        combined = thresh_methods[0]
        for thresh in thresh_methods[1:]:
            combined = cv2.bitwise_or(combined, thresh)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Extract answers
        answers, confidence = extract_answers_from_processed(cleaned, processed_image)
        
        return answers, confidence
        
    except Exception as e:
        print(f"❌ Adaptive method failed for answer key: {e}")
        return {}, 0

def extract_with_morphological_method(processed_image, original_image):
    """
    Morphological operations approach for answer key extraction
    """
    try:
        # Otsu thresholding
        _, thresh = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Advanced morphological operations
        # Remove vertical lines that might interfere
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
        thresh = cv2.subtract(thresh, vertical_lines)
        
        # Remove horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        thresh = cv2.subtract(thresh, horizontal_lines)
        
        # Enhance circular shapes
        circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, circular_kernel)
        
        answers, confidence = extract_answers_from_processed(thresh, processed_image)
        
        return answers, confidence * 0.9  # Slight penalty for morphological method
        
    except Exception as e:
        print(f"❌ Morphological method failed: {e}")
        return {}, 0

def extract_with_contour_analysis(processed_image, original_image):
    """
    Advanced contour analysis for answer key extraction
    """
    try:
        # Multiple thresholding approaches
        _, otsu_thresh = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adaptive_thresh = cv2.adaptiveThreshold(processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY_INV, 11, 2)
        
        # Combine thresholds
        combined = cv2.bitwise_or(otsu_thresh, adaptive_thresh)
        
        # Find contours
        contours = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        
        # Advanced contour filtering
        filtered_contours = advanced_contour_filtering(contours, processed_image)
        
        if len(filtered_contours) < 10:
            return {}, 0
        
        # Analyze contours for answers
        answers, confidence = analyze_answer_contours(filtered_contours, combined, processed_image)
        
        return answers, confidence
        
    except Exception as e:
        print(f"❌ Contour analysis failed: {e}")
        return {}, 0

def extract_with_pattern_recognition(processed_image, original_image):
    """
    Pattern recognition approach using template matching and feature detection
    """
    try:
        # Create multiple bubble templates
        templates = create_bubble_templates()
        
        best_matches = []
        
        for template in templates:
            # Multi-scale template matching
            scales = [0.8, 1.0, 1.2]
            for scale in scales:
                scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
                if scaled_template.shape[0] > processed_image.shape[0] or scaled_template.shape[1] > processed_image.shape[1]:
                    continue
                
                result = cv2.matchTemplate(processed_image, scaled_template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= 0.6)
                
                for pt in zip(*locations[::-1]):
                    best_matches.append((pt[0], pt[1], scaled_template.shape[1], scaled_template.shape[0]))
        
        if not best_matches:
            return {}, 0
        
        # Remove overlapping matches
        filtered_matches = non_max_suppression(best_matches)
        
        # Convert to answers
        answers = convert_matches_to_answers(filtered_matches, processed_image)
        confidence = min(1.0, len(answers) / 180)  # Confidence based on coverage
        
        return answers, confidence * 0.7  # Penalty for template matching
        
    except Exception as e:
        print(f"❌ Pattern recognition failed: {e}")
        return {}, 0

def create_bubble_templates():
    """
    Create various bubble templates for matching
    """
    templates = []
    
    # Filled circle templates of different sizes
    for size in [20, 25, 30, 35]:
        template = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(template, (size//2, size//2), size//3, 255, -1)
        templates.append(template)
        
        # Thick circle outline
        template_outline = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(template_outline, (size//2, size//2), size//3, 255, 3)
        templates.append(template_outline)
    
    return templates

def non_max_suppression(matches, overlap_thresh=0.3):
    """
    Apply non-maximum suppression to remove overlapping matches
    """
    if len(matches) == 0:
        return []
    
    # Convert to rectangles
    rects = []
    for (x, y, w, h) in matches:
        rects.append([x, y, x + w, y + h])
    
    rects = np.array(rects, dtype=np.float32)
    
    # Calculate areas
    areas = (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])
    
    # Sort by bottom-right y-coordinate
    indices = np.argsort(rects[:, 3])
    
    keep = []
    while len(indices) > 0:
        last = len(indices) - 1
        i = indices[last]
        keep.append(i)
        
        # Find overlapping rectangles
        xx1 = np.maximum(rects[i, 0], rects[indices[:last], 0])
        yy1 = np.maximum(rects[i, 1], rects[indices[:last], 1])
        xx2 = np.minimum(rects[i, 2], rects[indices[:last], 2])
        yy2 = np.minimum(rects[i, 3], rects[indices[:last], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        overlap = (w * h) / areas[indices[:last]]
        
        # Remove overlapping rectangles
        indices = np.delete(indices, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    
    return [matches[i] for i in keep]

def convert_matches_to_answers(matches, image):
    """
    Convert template matches to answer dictionary
    """
    if not matches:
        return {}
    
    # Sort matches by position
    sorted_matches = sorted(matches, key=lambda x: (x[1], x[0]))  # Sort by y, then x
    
    # Group into rows
    rows = []
    current_row = []
    current_y = None
    tolerance = 30
    
    for match in sorted_matches:
        x, y, w, h = match
        center_y = y + h // 2
        
        if current_y is None or abs(center_y - current_y) <= tolerance:
            current_row.append(match)
            current_y = center_y if current_y is None else current_y
        else:
            if len(current_row) >= 4:
                rows.append(current_row)
            current_row = [match]
            current_y = center_y
    
    if len(current_row) >= 4:
        rows.append(current_row)
    
    # Convert to answers
    answers = {}
    for q_num, row in enumerate(rows, 1):
        if q_num > 180:
            break
        
        # Sort row by x-coordinate
        row_sorted = sorted(row, key=lambda x: x[0])
        
        # For simplicity, assume first bubble is filled (template matching found it)
        if row_sorted:
            answers[q_num] = 'A'  # This would need more sophisticated analysis
    
    return answers

def advanced_contour_filtering(contours, image):
    """
    Advanced filtering specifically for answer key contours
    """
    filtered = []
    
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
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Convex hull analysis
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Answer key specific criteria (might be stricter)
        if (80 <= area <= 2500 and
            0.6 <= aspect_ratio <= 1.5 and
            extent > 0.4 and
            circularity > 0.3 and
            solidity > 0.6 and
            12 <= w <= 70 and 12 <= h <= 70):
            filtered.append(contour)
    
    return filtered

def extract_answers_from_processed(thresh_image, gray_image):
    """
    Extract answers from processed threshold image
    """
    # Find contours
    contours = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    
    # Filter contours
    filtered_contours = advanced_contour_filtering(contours, gray_image)
    
    if len(filtered_contours) < 10:
        return {}, 0
    
    # Group into rows and analyze
    answers, confidence = analyze_answer_contours(filtered_contours, thresh_image, gray_image)
    
    return answers, confidence

def analyze_answer_contours(contours, thresh_image, gray_image):
    """
    Analyze contours to extract answer key
    """
    # Sort contours by position
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    
    # Group into rows
    rows = group_contours_into_rows(sorted_contours)
    
    answers = {}
    confidence_scores = []
    
    for q_num, row in enumerate(rows, 1):
        if q_num > 180:
            break
        
        # Sort row by x-coordinate
        row_sorted = sorted(row, key=lambda c: cv2.boundingRect(c)[0])
        
        if len(row_sorted) < 4:
            continue
        
        # Take first 4 bubbles
        row_sorted = row_sorted[:4]
        
        # Find filled bubble
        fill_scores = []
        for contour in row_sorted:
            fill_score, fill_conf = analyze_contour_fill(contour, thresh_image, gray_image)
            fill_scores.append((fill_score, fill_conf))
        
        # Determine answer
        if fill_scores:
            max_fill = max(fill_scores, key=lambda x: x[0])
            max_idx = fill_scores.index(max_fill)
            
            if max_fill[0] > 0.25 and max_fill[1] > 0.5:
                answers[q_num] = chr(65 + max_idx)
                confidence_scores.append(max_fill[1])
    
    overall_confidence = np.mean(confidence_scores) if confidence_scores else 0
    return answers, overall_confidence

def group_contours_into_rows(contours, tolerance=40):
    """
    Group contours into rows for answer key processing
    """
    if not contours:
        return []
    
    # Get centers
    centers = [(contour, cv2.boundingRect(contour)[1] + cv2.boundingRect(contour)[3]//2) 
               for contour in contours]
    
    # Sort by y-coordinate
    centers.sort(key=lambda x: x[1])
    
    # Group into rows
    rows = []
    current_row = [centers[0][0]]
    current_y = centers[0][1]
    
    for contour, y in centers[1:]:
        if abs(y - current_y) <= tolerance:
            current_row.append(contour)
        else:
            if len(current_row) >= 4:
                rows.append(current_row)
            current_row = [contour]
            current_y = y
    
    if len(current_row) >= 4:
        rows.append(current_row)
    
    return rows

def analyze_contour_fill(contour, thresh_image, gray_image):
    """
    Analyze how filled a contour is
    """
    # Create mask
    mask = np.zeros(gray_image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Calculate metrics
    bubble_area = cv2.contourArea(contour)
    if bubble_area == 0:
        return 0, 0
    
    # Filled pixels ratio
    filled_pixels = cv2.countNonZero(cv2.bitwise_and(thresh_image, thresh_image, mask=mask))
    fill_ratio = filled_pixels / bubble_area
    
    # Mean intensity
    mean_val = cv2.mean(gray_image, mask=mask)[0]
    intensity_score = (255 - mean_val) / 255
    
    # Combined score
    fill_score = (fill_ratio * 0.7 + intensity_score * 0.3)
    confidence = min(1.0, fill_score * 1.5) if fill_score > 0.15 else 0
    
    return fill_score, confidence

def process_pdf_answer_key_advanced(pdf_path):
    """
    Advanced PDF processing for answer keys
    """
    try:
        print("📄 Processing PDF answer key with advanced methods...")
        
        doc = fitz.open(pdf_path)
        
        # Process multiple pages if available
        best_answers = {}
        best_confidence = 0
        
        for page_num in range(min(3, len(doc))):  # Process first 3 pages max
            page = doc[page_num]
            
            # High resolution conversion
            mat = fitz.Matrix(3.0, 3.0)  # Higher resolution
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to OpenCV image
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                # Process this page
                answers, confidence = extract_answer_key_from_image(image)
                
                if confidence > best_confidence:
                    best_answers = answers
                    best_confidence = confidence
        
        doc.close()
        
        print(f"✅ PDF processing complete. Best confidence: {best_confidence:.2f}")
        return best_answers
        
    except Exception as e:
        print(f"❌ Error processing PDF answer key: {e}")
        return {}

def extract_answer_key_from_image(image):
    """
    Extract answer key from a single image
    """
    processed_image = advanced_answer_key_preprocessing(image)
    
    # Try the best method first
    answers, confidence = extract_with_adaptive_method(processed_image, image)
    
    if confidence < 0.5:
        # Try other methods
        alt_answers, alt_conf = extract_with_contour_analysis(processed_image, image)
        if alt_conf > confidence:
            answers, confidence = alt_answers, alt_conf
    
    return answers, confidence

def ensemble_answer_extraction(processed_image, original_image):
    """
    Ensemble approach for answer key extraction
    """
    try:
        methods = [
            extract_with_adaptive_method,
            extract_with_contour_analysis,
            extract_with_morphological_method
        ]
        
        all_results = []
        for method in methods:
            answers, conf = method(processed_image, original_image)
            if answers:
                all_results.append((answers, conf))
        
        if not all_results:
            return {}
        
        # Weighted voting
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
                best_answer = max(votes.items(), key=lambda x: x[1])
                if best_answer[1] / total_weight > 0.4:  # Lower threshold for ensemble
                    final_answers[q] = best_answer[0]
        
        return final_answers
        
    except Exception as e:
        print(f"❌ Ensemble extraction failed: {e}")
        return {}

def compare_answers(student_answers, answer_key):
    """
    Enhanced answer comparison with detailed analysis
    """
    correct = 0
    wrong = 0
    unattempted = 0
    
    total_questions = 180  # NEET has 180 questions
    
    print(f"🔍 Comparing {len(student_answers)} student answers with {len(answer_key)} answer key entries")
    
    # Detailed comparison logging
    mismatches = []
    
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
            mismatches.append(f"Q{q}: Student={student_answer}, Correct={correct_answer}")
    
    # Log some mismatches for debugging
    if mismatches:
        print(f"📝 Sample mismatches: {mismatches[:5]}")
    
    print(f"📊 Comparison complete: {correct} correct, {wrong} wrong, {unattempted} unattempted")
    return correct, wrong, unattempted
