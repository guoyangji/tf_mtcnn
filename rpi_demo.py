import tensorflow as tf
import detect_face
import cv2

cap = cv2.VideoCapture(0)

minsize = 128
threshold = [0.6, 0.7, 0.7]
factor = 0.709
gpu_memory_fraction = 1.0

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, "./data")

while cap.isOpened():
    ret, img = cap.read()

    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    faces_num = bounding_boxes.shape[0]
    for face_position in bounding_boxes:
        face_position = face_position.astype(int)
        # cropped = img[face_position[1] - 50:face_position[3] + 50, face_position[0] - 50:face_position[2] + 50]
        cv2.rectangle(img, (face_position[0] - 10, face_position[1] - 10),
                      (face_position[2] + 10, face_position[3] + 10),
                      (0, 255, 0), 5)
    cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
