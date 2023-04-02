import cv2
import numpy as np
import math

cap = cv2.VideoCapture('C:/Users/cpote/Downloads/Telegram Desktop/2023-04-01-185819.webm')

while(1):
    ret, frame = cap.read()
    if not ret:
        break

    pts1 = np.float32([[667, 477], [1234, 477], [160, 960], [1633, 960]])
    pts2 = np.float32([[0, 0], [480, 0], [0, 640], [480, 640]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    result = cv2.warpPerspective(frame, M, (480, 640))

    hsv_img = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 80, 60])
    upper_blue = np.array([126, 255, 255])

    lower_cyan = np.array([40, 50, 80])
    upper_cyan = np.array([100, 255, 255])

    blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    cyan_mask = cv2.inRange(hsv_img, lower_cyan, upper_cyan)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(result, contours, -1, (255, 0, 0), 2)

    center_x = []
    center_y = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 50:
            x, y, w, h = cv2.boundingRect(contour)
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            center_x.append(cx)
            center_y.append(cy)

            cv2.putText(result, (str(round((cx * 1.06) / 100, 1)) + " " + str(round((cy * 1.17) / 100, 1))),
                        (x, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            cv2.circle(result, (cx, cy), 3, (255, 0, 0), -1)

    contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Отображение контуров на исходном изображении
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 50:
            x, y, w, h = cv2.boundingRect(contour)
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            center_x.append(cx)
            center_y.append(cy)

            cv2.putText(result, (str(round((cx * 1.06) / 100, 1)) + " " + str(round((cy * 1.17) / 100, 1))),
                        (x, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            cv2.circle(result, (cx, cy), 3, (0, 255, 0), -1)

    if center_x[0] - center_x[1] == 0 and center_x[0] > center_x[1]:
        angle = 90
    elif center_x[0] - center_x[1] == 0 and center_x[0] < center_x[1]:
        angle = -90
    else:
        angle = 57.3 * math.atan2((center_y[0] - center_y[1]), (center_x[0] - center_x[1]))

    cv2.putText(result, str(int(angle)), (100, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    X = (((center_x[0] + center_x[1])/2) * 1.06)/100
    Y = (((center_y[0] + center_y[1])/2) * 1.17)/100

    cv2.putText(result, (str(round(X, 2)) + " " + str(round(Y, 2))), (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    # Отображение результата
    cv2.imshow('result', result)

    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

# Destroys all of the HighGUI windows.
cv2.destroyAllWindows()

# release the captured frame
cap.release()