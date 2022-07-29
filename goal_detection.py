import math


class GoalDetection:
    def __init__(self, distance_threshold=150):
        self.distance_threshold = distance_threshold
        self.flag = False
        self.count = 0

    def check_goal(self, goal_position, ball_position):
        if self.flag:
            return self.flag

        (x, y, w, h) = ball_position
        center_ball = int(x + w / 2), int(y + h / 2)

        (x, y, w, h) = goal_position
        center_goal = int(x + w / 2), int(y + h / 2)

        distance = math.hypot(center_ball[0] - center_goal[0], center_ball[1] - center_goal[1])
        if distance < self.distance_threshold:
            self.count += 1
        if self.count > 6:
            self.flag = True

        return self.flag
