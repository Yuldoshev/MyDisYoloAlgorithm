#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import cv2
import time
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
#import sparsity
from tensorflow_model_optimization.python.core import sparsity

from PIL import Image

from yolo3.model import get_yolo3_model, get_yolo3_inference_model
from yolo3.postprocess_np import yolo3_postprocess_np
from common.data_utils import preprocess_image
from common.utils import get_classes, get_anchors, get_colors, draw_boxes
#from tensorflow.keras.utils import multi_gpu_model

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox, QGroupBox
from PyQt5.uic import loadUi
import PyQt5

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# tf.enable_eager_execution()

default_config = {
    "model_type": 'yolo4_mobilenet',
    "weights_path": 'model.h5',
    "pruning_model": False,
    "anchors_path": 'configs/yolo_anchors.txt',
    "classes_path": 'configs/clas_nomlari.txt',
    "score": 0.1,
    "iou": 0.5,
    "model_image_size": (416, 416),
    "gpu_num": 0,
}


class YOLO_np(object):
    _defaults = default_config

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        super(YOLO_np, self).__init__()
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = get_classes(self.classes_path)
        self.anchors = get_anchors(self.anchors_path)
        self.colors = get_colors(self.class_names)
        K.set_learning_phase(0)
        self.yolo_model = self._generate_model()

    def _generate_model(self):
        '''to generate the bounding boxes'''
        weights_path = os.path.expanduser(self.weights_path)
        assert weights_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        # YOLOv3 model has 9 anchors and 3 feature layers but
        # Tiny YOLOv3 model has 6 anchors and 2 feature layers,
        # so we can calculate feature layers number to get model type
        num_feature_layers = num_anchors // 3

        try:
            yolo_model, _ = get_yolo3_model(self.model_type, num_feature_layers, num_anchors, num_classes,
                                            input_shape=self.model_image_size + (3,), model_pruning=self.pruning_model)
            yolo_model.load_weights(weights_path)  # make sure model, anchors and classes match
            if self.pruning_model:
                yolo_model = sparsity.strip_pruning(yolo_model)
            yolo_model.summary()
        except Exception as e:
            print(repr(e))
            assert yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
        print('{} model, anchors, and classes loaded.'.format(weights_path))
        if self.gpu_num >= 2:
            #yolo_model = multi_gpu_model(yolo_model, gpus=self.gpu_num)
            yolo_model, _ = get_yolo3_model(yolo_model, gpus=self.gpu_num)
        return yolo_model

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'

        image_data = preprocess_image(image, self.model_image_size)
        image_shape = image.size

        start = time.time()
        out_boxes, out_classes, out_scores = self.predict(image_data, image_shape)
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        end = time.time()
        print("Inference time: {:.8f}s".format(end - start))

        # draw result on input image
        image_array = np.array(image, dtype='uint8')
        image_array = draw_boxes(image_array, out_boxes, out_classes, out_scores, self.class_names, self.colors)
        return Image.fromarray(image_array)

    def predict(self, image_data, image_shape):
        num_anchors = len(self.anchors)
        out_boxes, out_classes, out_scores = yolo3_postprocess_np(self.yolo_model.predict(image_data), image_shape,
                                                                  self.anchors, len(self.class_names),
                                                                  self.model_image_size, max_boxes=100)
        return out_boxes, out_classes, out_scores

    def dump_model_file(self, output_model_file):
        self.yolo_model.save(output_model_file)


def detect_video(yolo, video_path, output_path=""):
    vid = cv2.VideoCapture(0 if video_path == '0' else video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    video_FourCC = cv2.VideoWriter_fourcc(*'XVID') if video_path == '0' else cv2.VideoWriter_fourcc(*"mp4v")
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, (5. if video_path == '0' else video_fps), video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    if isOutput:
        out.release()
    cv2.destroyAllWindows()


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()


class ObjectDetection(QDialog):

    def __init__(self):
        super(ObjectDetection, self).__init__()
        loadUi('obj_detection.ui', self)
        self.image = None
        self.processedImage = None
        self.startBtn.clicked.connect(self.start_video)
        self.stopBtn.clicked.connect(self.stop_video)
        self.videoChb.setChecked(True)
        self.videoFileChb.setChecked(False)
        self.videoFileName = '0'
        self.savevideoFileName = 'result.mp4'

    def start_video(self):
        if (self.videoChb.isChecked()):
            self.videoFileName = 0
            self.file_save()
        else:
            self.file_open()
            self.file_save()

        self.capture = cv2.VideoCapture(self.videoFileName)
        # self.video_FourCC = cv2.VideoWriter_fourcc(*'XVID') if self.videoFileName == '0' else cv2.VideoWriter_fourcc(
        #     *"mp4v")
        # self.video_fps = self.capture.get(cv2.CAP_PROP_FPS)
        # self.video_size = (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        #                    int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # self.isOutput = True if self.savevideoFileName != "" else False
        # if self.isOutput:
        #     print("!!! TYPE:", type(self.savevideoFileName), type(self.video_FourCC), type(self.video_fps),
        #           type(self.video_size))
        #     self.out = cv2.VideoWriter(self.savevideoFileName, self.video_FourCC,
        #                                (5. if self.videoFileName == '0' else self.video_fps), self.video_size)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image = Image.fromarray(self.image)
        self.image = yolo.detect_image(self.image)
        self.image = np.asarray(self.image)
        # if self.isOutput:
        #     self.out.write(self.image)
        self.displayImage(self.image, 1)

    def stop_video(self):
        # self.out.release()
        self.capture.release()
        self.timer.stop()
        self.image = cv2.imread("thanks.jpg", cv2.IMREAD_COLOR)
        self.displayImage(self.image, 1)

    def displayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if self.videoFileName == 0:
            img = cv2.flip(img, 1)
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        # BGR>>RGB
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.videoLabel.setPixmap(QPixmap.fromImage(outImage))
            self.videoLabel.setScaledContents(True)

    def file_open(self):
        self.videoFileName, _ = QFileDialog.getOpenFileName(None, "Open VideoFile", "~/",
                                                            "Video Files (*.mp4 *.mpg *.mpeg)")

    def file_save(self):
        self.savevideoFileName = PyQt5.QtWidgets.QFileDialog.getSaveFileName(self, "Save Video File before starting",
                                                                             "~/",
                                                                             filter="Video Files (*.mp4 *.mpg *.mpeg)")


if __name__ == '__main__':
    yolo = YOLO_np()
    # detect_video(yolo, "a.mp4", "b.mp4")
    # detect_img(yolo)
    app = QApplication(sys.argv)
    window = ObjectDetection()
    window.setWindowTitle('Object Detection App')
    window.show()
    sys.exit(app.exec_())
