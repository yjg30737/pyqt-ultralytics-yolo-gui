# pyqt-ultralytics-yolo-gui
Example of using YOLOv8 of Ultralytics in to Object Detection, Semantic Segmentation about image and video in PyQt

This is an example of how to easily use Ultralytics' YOLOv8 object detection and image segmentation models in PyQt.

A class called **YOLOWrapper** is created to download the model remotely before the PyQt software is run. After the software is run, the file path is received and object detection or image segmentation is performed according to the user's selection.

The supported files are as follows: **png, jpg, mp4**

After running this, result file would be saved in same directory of main.py. You will see how fast it is :)

This program has a simple customization option, so you can display or remove object boxes, object names, and the probability that an object belongs to a specific class.

## What is YOLO?
YOLO is a model arcitecture that is gaining attention in the real-time object detection and image segmentation fields due to its fast speed. YOLOv8 is one of the most recent version of it.

## Requirements
* PyQt5>=5.14
* pillow
* opencv-python
* numpy
* ultralytics

## How to Run
1. git clone ~
2. pip install -r requirements.txt
3. python main.py
4. Copy the path in the input and click run. There are a three files (2 images, 1 video) in sample folder in case of you don't want to be bothered by finding files to test.

## Preview
![image](https://github.com/yjg30737/pyqt-ultralytics-yolo-gui/assets/55078043/56d6aaf0-7a36-460d-ba8a-6d78e00560c4)

## Output

Object Detection over image

![a_result](https://github.com/yjg30737/pyqt-ultralytics-yolo-gui/assets/55078043/4f475039-ed19-42bd-b7f5-3694c3d77fca)

Semantic Segmentation over image

![b_result](https://github.com/yjg30737/pyqt-ultralytics-yolo-gui/assets/55078043/035b2fae-e04c-497b-a8cb-f94cee0d4c4b)
