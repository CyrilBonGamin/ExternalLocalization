import cv2
import numpy as np
import os

cap2 = cv2.VideoCapture('C:/Users/cpote/Downloads/Telegram Desktop/2023-03-18-163202.webm')

size = (480, 640)

writer = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

while(1):
    ret, frame = cap2.read()
    if not ret:
        break

    pts1 = np.float32([[340, 227], [642, 220], [115, 470], [875, 453]])
    pts2 = np.float32([[0, 0], [480, 0], [0, 640], [480, 640]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    result = cv2.warpPerspective(frame, M, (480, 640))

    writer.write(result)

cap2.release()
writer.release()

cap = cv2.VideoCapture('filename.avi')

frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

# Store selected frames in an array
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)


cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Convert background to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

# Loop over all frames
ret = True
while(ret):

    ret, frame = cap.read()

    if not ret:
        break

    grframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dframe = cv2.absdiff(grframe, grayMedianFrame)

    th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(dframe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 600:
            x, y, w, h = cv2.boundingRect(contour)
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            cv2.putText(frame, (str(round((cx*1.06)/100, 1)) +" "+ str(round((cy*1.17)/100, 1))), (x, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('images', frame)
    cv2.waitKey(20)

# Release video object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()

os.remove('filename.avi')