import cv2
from datetime import datetime

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Camera", frame)
    # cv2.imwrite(str(datetime.now())+".jpg",frame)
    k = cv2.waitKey(1)
    if k == ord('q') or k == ord('Q'):
        break
cap.release()
