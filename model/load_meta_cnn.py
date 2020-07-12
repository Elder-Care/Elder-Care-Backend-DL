import os
import tensorflow as tf
import cv2
import numpy as np
import face_recognition
from keras.models import model_from_json
ckpt_name = './cnn_emotion_classifier.ckpt'
casc_name = './haarcascade_frontalface_alt.xml'

ckpt_path = os.path.join(ckpt_name)
casc_path = os.path.join(casc_name)


channel = 1
default_height = 48
default_width = 48
confusion_matrix = False
use_advanced_method = True
emotion_labels = ['angry', 'disgust:', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_class = len(emotion_labels)


# config=tf.ConfigProto(log_device_placement=True)
sess = tf.compat.v1.Session()

saver = tf.compat.v1.train.import_meta_graph(ckpt_path+'.meta')
saver.restore(sess, ckpt_path)
graph = tf.compat.v1.get_default_graph()
name = [n.name for n in graph.as_graph_def().node]

x_input = graph.get_tensor_by_name('x_input:0')
dropout = graph.get_tensor_by_name('dropout:0')
logits = graph.get_tensor_by_name('project/output/logits:0')

fanbingbing_image = face_recognition.load_image_file("../temp/images/fanbingbing.jpg")
fanbingbing_face_encoding = face_recognition.face_encodings(fanbingbing_image)[0]

# Load a second sample picture and learn how to recognize it.
huzhiyu_image = face_recognition.load_image_file("../temp/images/huzhiyu.jpg")
huzhiyu_face_encoding = face_recognition.face_encodings(huzhiyu_image)[0]

jiangwen_image = face_recognition.load_image_file("../temp/images/jiangwen.jpeg")
jiangwen_face_encoding = face_recognition.face_encodings(jiangwen_image)[0]

jikejunyi_image = face_recognition.load_image_file("../temp/images/jikejunyi.jpg")
jikejunyi_face_encoding = face_recognition.face_encodings(jikejunyi_image)[0]

pengyuyan_image = face_recognition.load_image_file("../temp/images/pengyuyan.jpg")
pengyuyan_face_encoding = face_recognition.face_encodings(pengyuyan_image)[0]

zhangziyi_image = face_recognition.load_image_file("../temp/images/zhangziyi.jpg")
zhangziyi_face_encoding = face_recognition.face_encodings(zhangziyi_image)[0]

known_face_encodings = [
    fanbingbing_face_encoding,
    huzhiyu_face_encoding,
    jiangwen_face_encoding,
    jikejunyi_face_encoding,
    pengyuyan_face_encoding,
    zhangziyi_face_encoding,
]


face_names = []


def advance_image(images_):
    rsz_img = []
    rsz_imgs = []
    for image_ in images_:
        rsz_img.append(image_)
    rsz_imgs.append(np.array(rsz_img))
    rsz_img = []
    for image_ in images_:
        rsz_img.append(np.reshape(cv2.resize(image_[2:45, :], (default_height, default_width)),
                                  [default_height, default_width, channel]))
    rsz_imgs.append(np.array(rsz_img))
    rsz_img = []
    for image_ in images_:
        rsz_img.append(np.reshape(cv2.resize(cv2.flip(image_, 1), (default_height, default_width)),
                                  [default_height, default_width, channel]))
    rsz_imgs.append(np.array(rsz_img))
    return rsz_imgs


def produce_result(images_):
    images_ = np.multiply(np.array(images_), 1. / 255)
    if use_advanced_method:
        rsz_imgs = advance_image(images_)
    else:
        rsz_imgs = [images_]
    pred_logits_ = []
    for rsz_img in rsz_imgs:
        pred_logits_.append(sess.run(tf.nn.softmax(logits), {x_input: rsz_img, dropout: 1.0}))
    return np.sum(pred_logits_, axis=0)


def produce_results(images_):
    results = []
    pred_logits_ = produce_result(images_)
    pred_logits_list_ = np.array(np.reshape(np.argmax(pred_logits_, axis=1), [-1])).tolist()
    for num in range(num_class):
        results.append(pred_logits_list_.count(num))
    result_decimals = np.around(np.array(results) / len(images_), decimals=3)
    return results, result_decimals

def predict_emotion(image_):
    image_ = cv2.resize(image_, (default_height, default_width))
    image_ = np.reshape(image_, [-1, default_height, default_width, channel])
    return produce_result(image_)[0]


def face_detect(image, casc_path_=casc_path):
    if os.path.isfile(casc_path_):
        face_casccade_ = cv2.CascadeClassifier(casc_path_)
        #img_ = cv2.imread(image)
        img_ = image
        img_gray_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # face detection
        faces = face_casccade_.detectMultiScale(
            img_gray_,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(30, 30),
        )
        return faces, img_gray_, img_
    else:
        print("There is no {} in {}".format(casc_name, casc_path_))


def main(image):

    test = []

    #frame = face_recognition.load_image_file(image)
    frame = image
    face_locations = face_recognition.face_locations(image)
    # 对待识别图片人脸区域位置进行编码，获取128维特征向量
    face_encodings = face_recognition.face_encodings(image, face_locations)
    faces, img_gray, img = face_detect(image)

    emotion_pre_dict = {}
   
        # img_gray full face     face_img_gray part of face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        match = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        if match[0]:
            name = "fanbingbing"
        elif match[1]:
            name = "huzhiyu "
        elif match[2]:
            name = "jiangwen"
        elif match[3]:
            name = 'jikejunyi'
        elif match[4]:
            name = 'pengyuyan'
        elif match[5]:
            name = 'zhangziyi'
        else:
            name = 'Unknown'

        top=int(top)
        right=int(right)
        bottom=int(bottom)
        left=int(left)
        face_img_gray = img_gray[top:bottom, left:right]
        results_sum = predict_emotion(face_img_gray)  # face_img_gray
        for i, emotion_pre in enumerate(results_sum):
            emotion_pre_dict[emotion_labels[i]] = emotion_pre
        # 输出所有情绪的概率
        # 此处即为应返回的值
        label = np.argmax(results_sum)
        emo = emotion_labels[int(label)]

        if emo == 'happy':
            num = 1
        else:
            num = 0
        # 绘制脸部区域框
        if name == "Unknown":
            a = {'elder': 'Unknown', 'emotion': 0, 'volunteer': None}
        elif name == 'huzhiyu ':
            a = {'elder': 'huzhiyu', 'emotion': num}
        else:
            a = {'volunteer': name, 'emotion': num}
        test.append(a)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # 在脸部区域下面绘制人名
        cv2.rectangle(frame, (left, bottom),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name + " is " +emo, (left + 6, bottom - 6),
                    font, 0.5, (255, 0,0), 1)

    # 显示图片

    cv2.imshow("test", frame)
    cv2.waitKey(1)
    return test
