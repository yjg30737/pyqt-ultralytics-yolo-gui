import os, sys
from pathlib import Path
import cv2

import numpy as np
from PIL import Image
from ultralytics import YOLO


def open_directory(path):
    if sys.platform.startswith('darwin'):  # macOS
        os.system('open "{}"'.format(path))
    elif sys.platform.startswith('win'):  # Windows
        os.system('start "" "{}"'.format(path))
    elif sys.platform.startswith('linux'):  # Linux
        os.system('xdg-open "{}"'.format(path))
    else:
        print("Unsupported operating system.")


class YOLOWrapper:
    def __init__(self):
        self.__model = ''
        self.__model_seg = ''

        self.download_model()

    def download_model(self):
        # Object detection model
        self.__model = YOLO('yolov8n.pt')
        # Semantic segmentation model
        self.__model_seg = YOLO('yolov8n-seg.pt')

        # 0 is object detection, 1 is semantic segmentation
        self.__model_dict = {
            0: self.__model,
            1: self.__model_seg
        }

    def get_result(self, cur_task, src_filename, plot_arg):
        cur_model = self.__model_dict[cur_task]
        if isinstance(cur_model, YOLO):
            try:
                ext = Path(src_filename).suffix
                dst_filename = f'{Path(src_filename).stem}_result{ext}'
                if ext in ['.jpg', '.png', '.jpeg']:
                    results = cur_model(src_filename)
                    # Save result image
                    for r in results:
                        boxes = plot_arg['boxes']
                        labels = plot_arg['labels']
                        conf = plot_arg['conf']
                        im_array = r.plot(boxes=boxes, labels=labels, conf=conf)
                        im = Image.fromarray(im_array[..., ::-1])
                        im.save(dst_filename)
                elif ext in ['.mp4']:
                    # Get original video metadata
                    vcap = cv2.VideoCapture(src_filename)  # Assuming all frames have the same size
                    fps = vcap.get(cv2.CAP_PROP_FPS)
                    width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    size = (width, height)

                    # Save result video
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video = cv2.VideoWriter(dst_filename, fourcc, fps, size)
                    results = cur_model.track(src_filename, stream=True)
                    for r in results:
                        boxes = plot_arg['boxes']
                        labels = plot_arg['labels']
                        conf = plot_arg['conf']

                        frame_ = r.plot(boxes=boxes, labels=labels, conf=conf)
                        frame_ = Image.fromarray(frame_[..., ::-1])
                        frame_ = np.array(frame_)
                        frame_ = frame_[:, :, ::-1]
                        video.write(frame_)
                return dst_filename
            except Exception as e:
                raise Exception(e)
        else:
            raise Exception('You have to call download_model first.')



# for CLI test
# w = YOLOWrapper()
# w.download_model()
# w.get_result(0, 'sample/a.jpg', {'boxes': True, 'labels': True, 'conf': True})
# w.get_result(1, 'sample/b.png', {'boxes': True, 'labels': True, 'conf': True})