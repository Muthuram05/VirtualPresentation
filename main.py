from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import numpy as np


width, height = 1280, 720
folderPath = "UDS 14068"
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)
imgNumber = 0
hs, ws = int(120 * 1), int(213 * 1)
gestureThreshold = 300
buttonPressed = False
buttonCounter = 0
buttonDelay = 30
annotations = [[]]
annotationNumber = -1
annotationStart = False
detector = HandDetector(detectionCon=0.8, maxHands=1)
while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)
    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        print(fingers)
        cx, cy = hand['center']
        lmList = hand['lmList']
        # small box
        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))
        indexFinger = xVal, yVal
        if cy <= gestureThreshold:
            annotationStart = False
            if fingers == [1, 0, 0, 0, 0]:
                annotationStart = False
                print("left")
                if imgNumber > 0:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = -1
                    imgNumber -= 1
            if fingers == [0, 0, 0, 0, 1]:
                annotationStart = False
                print("right")
                if imgNumber < len(pathImages) - 1:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = -1
                    imgNumber += 1
        # pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotationStart = False
        # painting
        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)
        else:
            annotationStart = False

            # erase
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                if annotationNumber > -1:
                    annotations.pop(-1)
                    annotationNumber -= 1
                    buttonPressed = True
    else:
        annotationStart = False
    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False
    # paint
    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
                cv2.line(imgCurrent, annotations[i][j - 1], annotations[i][j], (0, 0, 150), 10)

    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws: w] = imgSmall
    cv2.imshow("image", img)
    cv2.imshow("test", imgCurrent)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
        
