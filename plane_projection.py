import cv2
import numpy as np


class PlaneProjection:

    def __init__(self):
        self.dst = cv2.imread('dst.jpg')
        # video.avi
        self.pts1 = np.float32([[670, 68], [1010, 280], [324, 280], [670, 727]])
        self.pts2 = np.float32([[319, 33], [368, 212], [271, 212], [319, 391]])

    def draw_plan(self, players_boxes, players_teams, colors):
        teams_colors = {'Team A': 0, 'Team B': 1}
        matrix = cv2.getPerspectiveTransform(self.pts1, self.pts2)

        ground = self.dst.copy()

        for i, box in enumerate(players_boxes):
            # If Outlier (GoalKeeper OR Referee)
            if players_teams[i] not in teams_colors.keys():
                continue

            x = box[0] + box[2]
            y = box[1] + box[3]
            pts3 = np.float32([[x, y]])
            pts3o = cv2.perspectiveTransform(pts3[None, :, :], matrix)
            x1 = int(pts3o[0][0][0])
            y1 = int(pts3o[0][0][1])
            pp = (x1, y1)
            team_color = teams_colors[players_teams[i]]
            cv2.circle(ground, pp, 7, colors[team_color], -1)
        return ground
