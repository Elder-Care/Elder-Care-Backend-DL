import cv2
import keyboard
import numpy as np


def savePic(lastframe, frame, numOfFrame, videoWriter):
    SAVE_PATH = PIC_PATH + str(numOfFrame) + '.jpg'

    subtractframe1 = cv2.subtract(frame, lastframe)
    subtractframe2 = cv2.subtract(lastframe, frame)
    subtractframe = cv2.add(subtractframe1, subtractframe2)
    cvtframe = cv2.cvtColor(subtractframe, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    claheframe = clahe.apply(cvtframe)

    gaussianframe = cv2.GaussianBlur(claheframe, (3, 3), 0)

    sobelframe = cv2.Sobel(gaussianframe, -1, 1, 1)

    t2, otsuframe = cv2.threshold(sobelframe, 0, 255, cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilatedframe = cv2.dilate(otsuframe, kernel)

    cv2.imshow('frame', dilatedframe)
    # videoWriter.write(pic)
    # cv2.imwrite(SAVE_PATH, pic)


VIDEO_PATH = './video/record.avi'
PIC_PATH = './picture/'

cameraCapture = cv2.VideoCapture(0)
fps = cameraCapture.get(cv2.CAP_PROP_FPS)

size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

videoWriter = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
success, frame = cameraCapture.read()
win_name = "test"
numOfFrame = 0
firstflag = 0
lastframe = frame
while success > 0:
    if keyboard.is_pressed('q'):
        break
    if firstflag == 0:
        firstflag = 1
        continue
    else:
        savePic(lastframe, frame, numOfFrame, videoWriter)
        cv2.waitKey(30)
        lastframe = frame
    success, frame = cameraCapture.read()
    numOfFrame = numOfFrame + 1

cameraCapture.release()
cv2.destroyAllWindows()
