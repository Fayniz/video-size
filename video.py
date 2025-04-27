import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from imutils import perspective, contours
import imutils

# Initialize video capture (webcam)
cap = cv2.VideoCapture(2)

# Reference object dimensions (e.g., a 2cm x 2cm square)
REF_WIDTH_CM = 9.6

# Store reference object data
ref_object = None
pixel_per_cm = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = [c for c in cnts if cv2.contourArea(c) > 500]  # Filter small contours

    if cnts:
        # Sort contours left-to-right (leftmost = reference)
        (cnts, _) = contours.sort_contours(cnts)

        # If no reference yet, use the leftmost object
        if ref_object is None and len(cnts) > 0:
            ref_object = cnts[0]
            box = cv2.minAreaRect(ref_object)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, _, _) = box
            ref_width_px = euclidean(tl, tr)
            pixel_per_cm = ref_width_px / REF_WIDTH_CM

        # Measure objects if reference is set
        if pixel_per_cm is not None:
            for cnt in cnts:
                box = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(box)
                box = np.array(box, dtype="int")
                box = perspective.order_points(box)
                (tl, tr, br, bl) = box

                # Draw bounding box
                cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 0), 2)

                # Calculate width and height (cm)
                width_px = euclidean(tl, tr)
                height_px = euclidean(tr, br)
                width_cm = width_px / pixel_per_cm
                height_cm = height_px / pixel_per_cm

                # Display dimensions
                cv2.putText(frame, f"W: {width_cm:.1f}cm", (int(tl[0]), int(tl[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(frame, f"H: {height_cm:.1f}cm", (int(tr[0] + 10), int(tr[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Show frame
    cv2.imshow("Video Measurement", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()