# coding: utf-8
from PyQt5.QtWidgets import QApplication, QDialog, QGridLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer, QDateTime
from datetime import datetime
import tensorflow as tf
from RPi import GPIO
import configparser
import detect_face
import numpy as np
import requests
import logging
import sys
import cv2
import os

logging.basicConfig(
    filename=os.path.join("log", "rpi.log"),
    level=logging.INFO,
    format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode="a"
)

conf = configparser.ConfigParser()
conf.read("rpi.ini")
url = conf.get("rpi", "url")
SN = conf.get("rpi", "SN")
region_name = conf.get("rpi", "region_name")

GPIO.setmode(GPIO.BCM)
GPIO.setup(2, GPIO.IN)

minsize = 128
threshold = [0.6, 0.7, 0.7]
factor = 0.709

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, "./data")


class Video():
    def __init__(self, capture):
        self.capture = capture
        self.currentFrame = np.array([])

    def captureNextFrame(self):
        ret, readFrame = self.capture.read()
        if ret:
            self.currentFrame = cv2.cvtColor(readFrame, cv2.COLOR_BGR2RGB)
            return ret, readFrame

    def convertFrame(self):
        try:
            height, width = self.currentFrame.shape[:2]
            img = QImage(
                self.currentFrame,
                width,
                height,
                QImage.Format_RGB888
            )
            return QPixmap.fromImage(img)
        except:
            return None


class Example(QDialog):
    def __init__(self):
        super(Example, self).__init__()
        desktop_geometry = QApplication.desktop()
        main_window_width = desktop_geometry.width()
        main_window_height = desktop_geometry.height()
        # print(main_window_width, main_window_height)
        self.resize(main_window_width, main_window_height)
        # self.setWindowOpacity(0.5)
        self.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint)

        self.video = Video(cv2.VideoCapture(0))

        self.videoFrame = QLabel(self)
        self.videoFrame.setScaledContents(True)

        self.logo = QLabel(self)
        self.logo.setScaledContents(True)
        self.logo_img = QPixmap("logo.png")
        self.logo.setPixmap(self.logo_img)

        self.time = QLabel(self)
        self.time.setFont(QFont("Microsoft YaHei", 20))
        self.time.setWordWrap(True)
        self.time.setScaledContents(True)

        self.data = QLabel(self)
        self.data.setFont(QFont("Microsoft YaHei", 20))
        self.data.setWordWrap(True)
        self.data.setScaledContents(True)

        self.layout = QGridLayout(self)
        self.layout.addWidget(self.videoFrame, 0, 1, 4, 3)
        self.layout.addWidget(self.logo, 0, 4, 1, 1)
        self.layout.addWidget(self.time, 1, 4, 1, 1)
        self.layout.addWidget(self.data, 2, 4, 2, 1)

        self.show_video_timer = QTimer(self)
        self.show_video_timer.timeout.connect(self.show_video)
        self.show_video_timer.start()

        self.show_time_timer = QTimer(self)
        self.show_time_timer.timeout.connect(self.show_time)
        self.show_time_timer.start()

        self.mtcnn_timer = QTimer(self)
        self.mtcnn_timer.timeout.connect(self.mtcnn)
        self.mtcnn_timer.start(1000)

        self.mtcnnOpened = True

    def show_video(self):
        try:
            self.video.captureNextFrame()
            self.videoFrame.setPixmap(self.video.convertFrame())
        except TypeError:
            print("No Frame")

    def show_time(self):
        datetime = QDateTime.currentDateTime()
        time = datetime.toString("yyyy-MM-dd\ndddd\nhh:mm:ss ")
        self.time.setText(time)

    def cv2ImgToBytes(self, image):
        return cv2.imencode('.png', image)[1].tobytes()

    def mtcnn(self):
        try:
            self.data.setText("")
            if GPIO.input(2):
                if self.mtcnnOpened:
                    ret, img = self.video.captureNextFrame()
                    if ret:
                        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                        faces_num = bounding_boxes.shape[0]
                        if faces_num == 0:
                            # print(faces_num)
                            pass
                        else:
                            # print(faces_num)
                            for face_position in bounding_boxes:
                                face_position = face_position.astype(int)
                                # cv2.rectangle(img, (face_position[0] - 10, face_position[1] - 10),
                                #               (face_position[2] + 10, face_position[3] + 10),
                                #               (0, 255, 0), 2)
                                if face_position[0] < 10 or face_position[1] < 10 or face_position[2] < 10 or \
                                        face_position[3] < 10:
                                    self.data.setText("请站到指定位置！")
                                else:
                                    img_cropped = img[face_position[1] - 50:face_position[3] + 50,
                                                  face_position[0] - 50:face_position[2] + 50]
                                    # img_name = "{}.png".format(datetime.now().strftime("%Y%m%d%H%M%S%f"))
                                    # img_path = os.path.join("images", img_name)
                                    # cv2.imwrite(img_path, img_cropped)
                                    data = {
                                        "region_name": region_name,
                                        "faceName": "{}.png".format(datetime.now().strftime("%Y%m%d%H%M%S%f"))
                                    }
                                    files = {
                                        "faceImg": cv2.imencode('.png', img_cropped)[1].tobytes()
                                        # "faceImg": (img_name, self.cv2ImgToBytes(img_cropped), "image/png", {})
                                    }
                                    response = requests.request("POST", url, data=data, files=files)
                                    logging.info("method:{}, url:{}".format("POST", url))
                                    logging.info("data:{}".format(data))
                                    logging.info("response: " + response.text)
                                    if response.json()["status"] == 1000 and response.json()["result"] == 1:
                                        self.data.setText("识别成功！门打开了！")
                                    else:
                                        self.data.setText("识别失败！")
            else:
                pass
        except:
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Example()
    window.show()
    sys.exit(app.exec_())
