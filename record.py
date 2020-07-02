import cv2
import keyboard
import numpy as np
import os
import glob


def picProcess(lastframe, frame, numOfFrame):
    save_path = pic_path + str(numOfFrame) + '.jpg'

    subtractframe1 = cv2.subtract(frame, lastframe)
    subtractframe2 = cv2.subtract(lastframe, frame)
    subtractframe = cv2.add(subtractframe1, subtractframe2)
    cvtframe = cv2.cvtColor(subtractframe, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    claheframe = clahe.apply(cvtframe)

    gaussianframe = cv2.GaussianBlur(claheframe, (3, 3), 0)

    sobelframe = cv2.Sobel(gaussianframe, -1, 1, 1)

    t2, otsuframe = cv2.threshold(sobelframe, 0, 255, cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    openframe = cv2.morphologyEx(otsuframe, cv2.MORPH_OPEN, kernel)

    cv2.imshow('frame', openframe)
    # videoWriter.write(pic)
    # cv2.imwrite(SAVE_PATH, pic)


def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih


def draw_person(frame, person):
    x, y, w, h = person
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)


def detect(frame):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    found, w = hog.detectMultiScale(frame)

    found_filtered = []

    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and is_inside(r, q):
                break
            else:
                found_filtered.append(r)

    for person in found_filtered:
        draw_person(frame, person)

    cv2.imshow('people detection', frame)


BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]


def detect_key_point(model_path, frame, inWidth=368, inHeight=368, threshhold=0.2):
    net = cv2.dnn.readNetFromTensorflow(model_path)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    scalefactor = 2.0
    net.setInput(
        cv2.dnn.blobFromImage(frame, scalefactor, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    assert (len(BODY_PARTS) == out.shape[1])
    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]
        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > threshhold else None)
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    t, _ = net.getPerfProfile()
    cv2.imshow('people detection', frame)


video_path = './video/record.avi'
pic_path = './picture/'
model_path = './pb/graph_opt.pb'

cameraCapture = cv2.VideoCapture(0)
fps = cameraCapture.get(cv2.CAP_PROP_FPS)

size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

videoWriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
success, frame = cameraCapture.read()
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
        #picProcess(lastframe, frame, numOfFrame)
        detect(frame)
        # detect_key_point(model_path, frame, inWidth=368, inHeight=368, threshhold=0.05)
        cv2.waitKey(30)
        lastframe = frame
    success, frame = cameraCapture.read()
    numOfFrame = numOfFrame + 1

cameraCapture.release()
cv2.destroyAllWindows()
