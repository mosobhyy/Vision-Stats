import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from object_detection import ObjectDetection
from collections import Counter


# Separate player into two teams and referee
class SeparateTeams:

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.ObjectDetection = ObjectDetection()
        self.training_players = []
        new_boxes = []
        colors_array = []
        i = 0
        while i < 5:
            ret, img = self.cap.read()
            img = cv2.resize(img, (1280, 720))
            boxes, confidences, class_ids = self.ObjectDetection.detect(img)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, .2, .4)
            if len(indexes) > 0:
                for j in indexes.flatten():
                    if class_ids[j] == 0:
                        new_boxes.append(boxes[j])
            colors_arrs = self.preprocess(img, new_boxes)
            colors_array.append(colors_arrs)
            for colors_arr in colors_arrs:
                self.training_players.extend(colors_arr)
            i += 1
        self.classifier = KMeans(n_clusters=3).fit(self.training_players)
        print("Training completed on " + str(np.shape(self.training_players)[0]) + " sampels")

        results = []
        for colors_arrs in colors_array:
            for colors_arr in colors_arrs:
                if len(colors_arr) != 0:
                    labels = self.classifier.predict(colors_arr)
                    counts = Counter(labels)
                    results.append(counts.most_common()[0][0])

        results_counts = Counter(results)
        self.class_zero_id = results_counts.most_common()[0][0]
        self.class_one_id = results_counts.most_common()[1][0]
        print("class_zero_id = " + str(self.class_zero_id))
        print("class_one_id = " + str(self.class_one_id))

    # Split players by their colors
    def separate(self, img, boxes):
        results = []
        colors_arrs = self.preprocess(img, boxes)
        for colors_arr in colors_arrs:
            if len(colors_arr) != 0:
                labels = self.classifier.predict(colors_arr)
                counts = Counter(labels)
                if counts.most_common()[0][0] == self.class_zero_id:
                    results.append(0)
                elif counts.most_common()[0][0] == self.class_one_id:
                    results.append(1)
                else:
                    results.append(2)
            else:
                results.append(2)

        return results

    def masking(self, img):
        # do masking
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # green range
        upper_green = np.array([86, 255, 255])
        lower_green = np.array([36, 25, 25])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        res = cv2.bitwise_and(img, img, mask=mask)
        mask_inv = cv2.bitwise_not(mask)
        new_img = cv2.bitwise_or(img, img, mask=mask_inv)

        return new_img

    def preprocess(self, img, boxes):
        colors_arr = []
        b = 5
        new_img = self.masking(img)
        new_img = Image.fromarray(new_img)
        for x, y, w, h in boxes:
            cropped_player = new_img.crop((x + b, y + b, x + w - b, y + h - b))
            cropped_player = np.array(cropped_player.getdata()).reshape(cropped_player.size[0] * cropped_player.size[1],
                                                                        3)
            cropped_player = cropped_player.astype(float)
            cropped_player = cropped_player[~np.all(cropped_player == 0, axis=1)]
            # cropped_player = cropped_player[~np.all(cropped_player >= 230, axis=1)]
            colors_arr.append(cropped_player)

        return colors_arr

    def clf_colors(self):
        colors = []
        colors.append(self.classifier.cluster_centers_[self.class_zero_id])
        colors.append(self.classifier.cluster_centers_[self.class_one_id])
        colors.append([0, 0, 0])
        return colors
