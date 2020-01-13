
import picamera
from picamera import PiCamera
import picamera.array
import cv2
import numpy as np


class CameraInterface:

    def __init__(self):
        self.camera = PiCamera()

        # With the maximum resolution takes about 10 seconds to process an image
        self.camera.resolution = (2528, 1968)

        # self.camera.resolution = (3280, 2464)
        # self.camera.resolution = (1640, 922)

        self.camera.exposure_mode = 'auto'

        # camera.iso = 150
        # camera.shutter_speed = 150
        # camera.exposure_mode = 'off'

    def capture_frame(self):     # capture frame and filters the image for vegetation detection

        img = np.empty((self.camera.resolution[1], self.camera.resolution[0], 3), dtype=np.uint8)

        self.camera.capture(img, 'rgb')

        # img = picamera.array.PiRGBArray(self.camera).array

        return img


    def save_image(self, newfile, image):

        cv2.imwrite(newfile, image)

        return






