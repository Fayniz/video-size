import cv2
import numpy as np
import math

# Define reference object width in cm
KNOWN_WIDTH = 5.0  # Change this based on your actual reference object

# Initialize video capture
cap = cv2.VideoCapture(0)

# Function to calculate pixels per metric
def find_pixels_per_metric(box):
    (tl, tr, br, bl) = box
    width = math.dist(tl, tr)
    return width / KNOWN_WIDTH  # pixels per cm

# Preprocessing helper
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    return edged

pixels_per_cm = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    edged = preprocess(frame)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 500:
            continue  # Filter out small noise

        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = sorted(box, key=lambda x: (x[0], x[1]))  # Sort for consistent ordering

        # If pixels_per_cm is not set, use the first detected object
        if pixels_per_cm is None:
            pixels_per_cm = find_pixels_per_metric(box)

        # Draw bounding box
        box = np.array(box, dtype="int")  # Ensure box is an integer array
        cv2.drawContours(frame, [box.astype("int")], 0, (0, 255, 0), 2)

        (tl, tr, br, bl) = box

        width = math.dist(tl, tr) / pixels_per_cm
        height = math.dist(tr, br) / pixels_per_cm

        mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
        mid_pt_vertical = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))

        # Put text for width and height
        cv2.putText(frame, "{:.1f}cm".format(width), (mid_pt_horizontal[0] - 20, mid_pt_horizontal[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, "{:.1f}cm".format(height), (mid_pt_vertical[0] + 10, mid_pt_vertical[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
