from temp import load_meta_cnn

import cv2

capture = cv2.VideoCapture('./pic/test2.avi')
while True:
    ret, frame = capture.read()
    frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
    a = load_meta_cnn.main(frame)
    #cv2.waitKey(30)
capture.release()
cv2.destroyAllWindows()
