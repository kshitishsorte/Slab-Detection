import cv2
import math
from cv2 import COLOR_BGR2GRAY, COLOR_BGR2HSV, COLOR_BGR2RGB
import numpy as np


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areaList = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(f"Area: {area}")
        areaList.append(area)

    finalarea = max(areaList)
    for i in range(len(areaList)):
        if areaList[i] == finalarea:
            index = i
    finalcontour = contours[index]
    print(f"Final Area : {finalarea}")

    if finalarea > 5:
        peri = cv2.arcLength(finalcontour, True)
        print(f"Peri: {peri}")

        D = abs((peri ** 2) - (16 * area))
        breadth = (peri - math.sqrt(D)) / 4
        length = (peri + math.sqrt(D)) / 4
        print(f"Final Length : {length}")


def empty(a):
    pass


cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 93, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 157, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 37, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 238, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 90, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

while True:
    img = cv2.resize(cv2.imread("plateend.jpg"), (640, 480))
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)

    imgdisp = img.copy()

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    hsv = cv2.cvtColor(img, COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    lines = cv2.HoughLinesP(mask, 2, np.pi / 180, 100, maxLineGap=5, minLineLength=75)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 5)

    imgContour = img.copy()

    imgGray = cv2.cvtColor(imgContour, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.medianBlur(imgGray, 5)

    # For masking the Blue in the original image (original image has straight lines detected)
    lower_mask = np.array([120, 255, 0])
    upper_mask = np.array([179, 255, 255])

    hsv2 = cv2.cvtColor(img, COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv2, lower_mask, upper_mask)

    # Testing contour of the mask2
    edges = cv2.Canny(mask2, 50, 75)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, maxLineGap=250)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # Finding the edges of the Blue lines
    imgCanny = cv2.Canny(mask2, 85, 150)

    lines2 = cv2.HoughLinesP(imgCanny, 1, np.pi / 180, 10, maxLineGap=500, minLineLength=0)
    if lines2 is not None:
        for line in lines2:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    getContours(imgCanny)

    imgBlank = np.zeros_like(img)
    imgStack = stackImages(0.8, ([img, imgGray, mask], [imgCanny, imgContour, hsv2]))

    cv2.imshow("Stack", imgStack)

    cv2.waitKey(1)
