import math
from collections import defaultdict


class Possession:
    def __init__(self, distance_threshold=100, possession_threshold=10, missed_possession_threshold=5):
        self.distance_threshold = distance_threshold
        self.possession_threshold = possession_threshold
        self.missed_possession_threshold = missed_possession_threshold
        self.players_possession = defaultdict(lambda: {'Possession': 0, 'Missed': 0})
        self.teams_possession = {'Team A': 1, 'Team B': 1}

    def get_ball_holder_player(self, players_info, ball_position):

        if not ball_position:
            return

        (x, y, w, h) = ball_position
        center_ball_box = int(x + w / 2), int(y + h / 2)

        # get the closest Player to the ball
        closest_player = (0, self.distance_threshold)  # Store (Player_id, distance to the ball)
        exist = False
        ball_holder_player = None
        for player_id, player_info in players_info.items():
            (x, y, w, h) = player_info['Position']
            player_foot = int(x + w / 2), int(y + h)

            distance = math.hypot(center_ball_box[0] - player_foot[0], center_ball_box[1] - player_foot[1])
            if distance < closest_player[-1]:
                exist = True
                closest_player = (player_id, distance)

        # Return ball_holder_player_id if exist
        if exist:
            ball_holder_player = closest_player[0]

        return ball_holder_player

    def get_possession(self, players_info, ball_position):

        ball_holder_player = self.get_ball_holder_player(players_info, ball_position)

        if not ball_holder_player:
            return None, None

        ball_holder_team = players_info[ball_holder_player]['Team']

        if ball_holder_team == 'Outlier':
            return None, None

        # increase player's possession value by 1
        possession = self.players_possession[ball_holder_player]['Possession']
        missed = self.players_possession[ball_holder_player]['Missed']
        possession += 1
        # increase possession value of Team of the player by 1 if he reached to possession_threshold
        if possession >= self.possession_threshold:
            self.teams_possession[ball_holder_team] += 1
            self.players_possession.pop(ball_holder_player)
        else:
            self.players_possession[ball_holder_player] = {'Possession': possession, 'Missed': missed}

        # increase other players' missed value by 1
        for player_id, pos_miss_dict in self.players_possession.copy().items():
            if player_id == ball_holder_player:
                continue

            miss = pos_miss_dict['Missed']

            # Remove player if crossed missed_threshold from the list
            # (Lost the ball or almost does not possess the ball)
            if miss >= self.missed_possession_threshold:
                self.players_possession.pop(player_id)
            else:
                self.players_possession[ball_holder_player] = {'Possession': possession, 'Missed': missed}

        return ball_holder_player, ball_holder_team
