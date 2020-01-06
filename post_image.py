import requests
import configparser
from datetime import datetime
import logging
import os
import cv2
import time

logging.basicConfig(
    filename=os.path.join("log", "test.log"),
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

url = "http://139.199.208.26/upload_face/"


def cv2ImgToBytes(image):
    return cv2.imencode('.png', image)[1].tobytes()


img = cv2.imread("Aaron_Peirsol_0001.jpg")
data = {
    "region_name": region_name,
    "faceName": datetime.now().strftime("%Y%m%d%H%M%S%f")+".jpg"
}
files = {
    "faceImg": ("Aaron_Peirsol_0001.jpg", cv2ImgToBytes(img), "image/jpeg", {})
}
response = requests.request("POST", url, data=data, files=files)
logging.info("method:{}, url:{}".format("POST", url))
logging.info("data:{}".format(data))
logging.info("response: " + response.text)
