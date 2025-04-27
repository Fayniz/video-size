import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from imutils import perspective
import imutils

def calculate_size(frame, ref_object, pixel_per_cm):
    """Calculate and display sizes of detected objects"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # Edge detection
    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    
    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Filter small contours and sort from left to right
    cnts = [x for x in cnts if cv2.contourArea(x) > 100]
    
    if not cnts:
        return frame, None
    
    # Draw contours and calculate dimensions
    frame_with_sizes = frame.copy()
    for cnt in cnts:
        if cnt is ref_object:
            color = (0, 255, 0)  # Green for reference object
        else:
            color = (0, 0, 255)  # Red for other objects
            
        box = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        
        # Draw the bounding box
        cv2.drawContours(frame_with_sizes, [box.astype("int")], -1, color, 2)
        
        # Calculate midpoints for text placement
        mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
        mid_pt_vertical = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
        
        # Calculate width and height in cm
        width = euclidean(tl, tr) / pixel_per_cm
        height = euclidean(tr, br) / pixel_per_cm
        
        # Display measurements
        cv2.putText(frame_with_sizes, f"{width:.1f}cm", 
                   (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame_with_sizes, f"{height:.1f}cm", 
                   (int(mid_pt_vertical[0] + 10), int(mid_pt_vertical[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    return frame_with_sizes, cnts

def select_reference_object(cnts, frame):
    """Allows user to select a reference object and input its size"""
    print("Select a reference object by number:")
    # Draw numbered contours
    numbered_frame = frame.copy()
    for i, cnt in enumerate(cnts):
        # Get center of contour
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
            
        # Draw number
        cv2.putText(numbered_frame, str(i), (cX, cY), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw contour
        cv2.drawContours(numbered_frame, [cnt], -1, (0, 255, 0), 2)
    
    cv2.imshow("Select Reference Object", numbered_frame)
    cv2.waitKey(1)
    
    try:
        ref_idx = int(input(f"Enter the number of the reference object (0-{len(cnts)-1}): "))
        actual_width = float(input("Enter the actual width of the reference object in cm: "))
        
        # Calculate pixels per cm
        ref_object = cnts[ref_idx]
        box = cv2.minAreaRect(ref_object)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        pixel_width = euclidean(tl, tr)
        pixel_per_cm = pixel_width / actual_width
        
        cv2.destroyWindow("Select Reference Object")
        return ref_object, pixel_per_cm
    except Exception as e:
        print(f"Error: {e}")
        cv2.destroyWindow("Select Reference Object")
        return None, None

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Initial flags
    ref_object = None
    pixel_per_cm = None
    setup_mode = True
    
    print("Press 's' to set up reference object, 'q' to quit")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        key = cv2.waitKey(1) & 0xFF
        
        # Setup mode - select reference object
        if setup_mode:
            # Process frame to find contours
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            edged = cv2.Canny(blur, 50, 100)
            edged = cv2.dilate(edged, None, iterations=1)
            edged = cv2.erode(edged, None, iterations=1)
            
            # Find contours
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            
            # Filter small contours
            cnts = [x for x in cnts if cv2.contourArea(x) > 100]
            
            # Draw all contours
            frame_copy = frame.copy()
            cv2.drawContours(frame_copy, cnts, -1, (0, 255, 0), 2)
            cv2.putText(frame_copy, "Setup Mode: Press 's' to select reference object", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Object Measurement", frame_copy)
            
            if key == ord('s') and cnts:
                ref_object, pixel_per_cm = select_reference_object(cnts, frame)
                if ref_object is not None and pixel_per_cm is not None:
                    setup_mode = False
                    print("Reference object set. Measuring mode active.")
        else:
            # Measurement mode
            frame_with_sizes, detected_cnts = calculate_size(frame, ref_object, pixel_per_cm)
            
            if detected_cnts is None or not detected_cnts:
                cv2.putText(frame, "No objects detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Object Measurement", frame)
            else:
                cv2.putText(frame_with_sizes, "Measuring Mode (Press 'r' to reset reference)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Object Measurement", frame_with_sizes)
            
            if key == ord('r'):
                setup_mode = True
                print("Reset reference object.")
        
        # Exit if 'q' is pressed
        if key == ord('q'):
            break
    
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()