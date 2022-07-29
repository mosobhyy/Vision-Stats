import cv2
import numpy as np


class HeatMap:
    def __init__(self, frame_count_threshold=15):
        self.map = cv2.imread('dst.jpg')
        # video.avi
        self.pts1 = np.float32([[670, 68], [1010, 280], [324, 280], [670, 727]])
        self.pts2 = np.float32([[319, 33], [368, 212], [271, 212], [319, 391]])
        self.count = -1
        self.frame_count_threshold = frame_count_threshold

    def heatMap(self, players_info, player_id=1):
        self.count += 1
        if self.count % self.frame_count_threshold != 0:
            return self.map

        matrix = cv2.getPerspectiveTransform(self.pts1, self.pts2)

        (x, y, w, h) = players_info[player_id]['Position']
        pts3 = np.float32([[x + w, y + h]])
        pts3o = cv2.perspectiveTransform(pts3[None, :, :], matrix)
        x1 = int(pts3o[0][0][0])
        y1 = int(pts3o[0][0][1])
        pp = (x1-7, y1-7)
        pp2 = (x1+7, y1+7)
        image = self.map.copy()
        cv2.rectangle(image, pp, pp2, [0, 0, 200], -1)
        alpha = 0.4
        image_new = cv2.addWeighted(image, alpha, self.map, 1 - alpha, 0)

        self.map = image_new
        return self.map
