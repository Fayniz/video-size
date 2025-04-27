import cv2
import math


points = []  # List to store points
def draw_circle(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) == 2:
            points = []
        points.append((x, y))

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", draw_circle)

capture = cv2.VideoCapture(2)

while True:
    _, frame = capture.read()

    for point in points:
        cv2.circle(frame, point, 5, (25, 15, 255), -1)
    
    if len(points) == 2:
        pt1, pt2 = points[0], points[1]
        distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        cv2.putText(frame, f"Distance: {distance:.2f}", (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

capture.release()

cv2.destroyAllWindows()