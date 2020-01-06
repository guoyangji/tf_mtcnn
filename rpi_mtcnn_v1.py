import tensorflow as tf
import detect_face
import cv2
import os
from datetime import datetime
from RPi import GPIO
from apscheduler.schedulers.blocking import BlockingScheduler
import logging

logging.basicConfig()
cap = cv2.VideoCapture(0)
sched = BlockingScheduler()
GPIO.setmode(GPIO.BCM)
GPIO.setup(2, GPIO.IN)

minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, "./data")


@sched.scheduled_job('interval', seconds=2)
def my_job():
    if GPIO.input(2):
        ret, img = cap.read()
        if ret:
            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            faces_num = bounding_boxes.shape[0]
            if faces_num == 0:
                print(faces_num)
            else:
                print(faces_num)
                for face_position in bounding_boxes:
                    face_position = face_position.astype(int)
                    cv2.rectangle(img, (face_position[0] - 10, face_position[1] - 10),
                                  (face_position[2] + 10, face_position[3] + 10),
                                  (0, 255, 0), 2)

                path = os.path.join("images", (str(datetime.now()) + ".jpg"))
                cv2.imwrite(path, img)
    else:
        print("waiting~")


sched.start()
