import cv2
import numpy as np


class ObjectDetection:
    def __init__(self, weights_path="dnn_model/yolov3_custom.backup", cfg_path="dnn_model/yolov3_customm.cfg"):
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.classes = ['Player', 'Ball', 'Goal']

    def detect(self, frame):
        frame = cv2.resize(frame, (1280, 720))
        hight, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

        self.net.setInput(blob)

        output_layers_name = self.net.getUnconnectedOutLayersNames()

        layerOutputs = self.net.forward(output_layers_name)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > 0.40:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * hight)
                    w = int(detection[2] * width)
                    h = int(detection[3] * hight)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, .2, .4)
        boxes = [box for index, box in enumerate(boxes) if index in indexes]
        confidences = [confidence for index, confidence in enumerate(confidences) if index in indexes]
        class_ids = [class_id for index, class_id in enumerate(class_ids) if index in indexes]

        return boxes, confidences, class_ids
