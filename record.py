import cv2
import keyboard

cameraCapture = cv2.VideoCapture(0)

fps = cameraCapture.get(cv2.CAP_PROP_FPS)

size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

videoWriter = cv2.VideoWriter('./record.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                              fps, size)
success, frame = cameraCapture.read()
numOfFrame = 0
while success > 0:
    if keyboard.is_pressed('q'):
        break
    if numOfFrame % 5 == 0:
        videoWriter.write(frame)
        success, frame = cameraCapture.read()
    numOfFrame += numOfFrame

cameraCapture.release()
#pushing test