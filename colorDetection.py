import cv2
import numpy as np

def markBall(frame, mask, r, g, b):
    canny_output = cv2.Canny(mask, 10, 100)
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])

    for i in range(len(contours)):
        color = (b, g, r)
        # cv2.drawContours(frame, contours_poly, i, color)
        if abs(boundRect[i][0] - boundRect[i][1]) < 400 and abs(boundRect[i][2] - boundRect[i][3]) < 400:
            cv2.rectangle(frame, (int(boundRect[i][0]), int(boundRect[i][1])), \
                          (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)


def main():
    video = cv2.VideoCapture('rgb_ball_720.mp4')
    while(video.isOpened()):
        ret, frame = video.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red, upper_red)

        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        lower_green = np.array([36, 25, 25])
        upper_green = np.array([70, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        markBall(frame, red_mask, 255, 0, 0)
        markBall(frame, blue_mask, 0, 0, 255)
        markBall(frame, green_mask, 0, 255, 0)
        markBall(frame, yellow_mask, 255, 255, 0)

        cv2.imshow('Our video', frame)
        if cv2.waitKey(10) == ord('q'):
            break

if __name__ == '__main__':
    main()