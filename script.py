import os, sys
from pathlib import Path
import cv2
from collections import Counter

import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.solutions import object_counter
from ultralytics.utils.plotting import Annotator, colors


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
            1: self.__model_seg,
            2: self.__model_seg
        }

    def get_result(self, cur_task, src_filename, plot_arg):
        cur_model = self.__model_dict[cur_task]
        if isinstance(cur_model, YOLO):
            try:
                result_dict = {}
                ext = Path(src_filename).suffix
                dst_filename = f'{Path(src_filename).stem}_result{ext}'
                if ext in ['.jpg', '.png', '.jpeg']:
                    results = cur_model(src_filename)
                    # Save result image
                    for r in results:
                        boxes = plot_arg['boxes']
                        labels = plot_arg['labels']
                        conf = plot_arg['conf']
                        arr = [int(cls.item()) for cls in r.boxes.cls]
                        arr = Counter(arr)
                        for k, v in arr.items():
                            result_dict[r.names[int(k)]] = v
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
                    if cur_task == 2:
                        # results = cur_model.track(src_filename, stream=True)
                        result_dict = {}

                        while True:
                            ret, im0 = vcap.read()
                            if not ret:
                                print("Video frame is empty or video processing has been successfully completed.")
                                break

                            annotator = Annotator(im0, line_width=2)

                            results = cur_model.track(im0, persist=True)

                            if results[0].boxes.id is not None and results[0].masks is not None:
                                masks = results[0].masks.xy
                                track_ids = results[0].boxes.id.int().cpu().tolist()

                                for mask, track_id in zip(masks, track_ids):
                                    annotator.seg_bbox(mask=mask,
                                                       mask_color=colors(track_id, True),
                                                       track_label=str(track_id))

                            video.write(im0)
                            # cv2.imshow("instance-segmentation-object-tracking", im0)
                            #
                            # if cv2.waitKey(1) & 0xFF == ord('q'):
                            #     break

                        video.release()
                        vcap.release()
                        cv2.destroyAllWindows()
                    else:
                        results = cur_model.track(src_filename, stream=True)
                        for r in results:
                            boxes = plot_arg['boxes']
                            labels = plot_arg['labels']
                            conf = plot_arg['conf']

                            frame_ = r.plot(boxes=boxes, labels=labels, conf=conf)
                            frame_ = Image.fromarray(frame_[..., ::-1])
                            frame_ = np.array(frame_)
                            annotator = Annotator(frame_, line_width=2)

                            if results[0].boxes.id is not None and results[0].masks is not None:
                                masks = results[0].masks.xy
                                track_ids = results[0].boxes.id.int().cpu().tolist()

                                for mask, track_id in zip(masks, track_ids):
                                    annotator.seg_bbox(mask=mask,
                                                       mask_color=colors(track_id, True),
                                                       track_label=str(track_id))

                            frame_ = frame_[:, :, ::-1]
                            video.write(frame_)
                return dst_filename, result_dict
            except Exception as e:
                raise Exception(e)
        else:
            raise Exception('You have to call download_model first.')



# for CLI test
# w = YOLOWrapper()
# w.download_model()
# w.get_result(0, 'sample/a.jpg', {'boxes': True, 'labels': True, 'conf': True})
# w.get_result(1, 'sample/b.png', {'boxes': True, 'labels': True, 'conf': True})