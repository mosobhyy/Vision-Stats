import math
from collections import defaultdict


class Statistics:
    def __init__(self, distance_threshold=100):
        self.distance_threshold = distance_threshold
        self.players_statistics = defaultdict(lambda: {'Possession': 0, 'Missed': 0})

    def get_closest_players(self, players_info, ball_position):

        if not ball_position:
            return None, None

        (x, y, w, h) = ball_position
        center_ball_box = int((x + x + w) / 2), int((y + y + h) / 2)

        # All closest Players to the ball
        closest_players = {}
        # closest Player to the ball
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

            if distance < self.distance_threshold:
                closest_players[player_id] = player_info

        # Return ball_holder_player_id if exist
        if exist:
            ball_holder_player = closest_player[0]

        return closest_players, ball_holder_player

    def get_statistics(self, players_info, ball_position):

        closest_players, ball_holder_player = self.get_closest_players(players_info, ball_position)

        if not ball_holder_player:
            return None

        for player_id, player_info in closest_players.items():
            if player_id == ball_holder_player:
                self.players_statistics[player_id]['Possession'] += 1
            else:
                self.players_statistics[player_id]['Missed'] += 1

        return self.players_statistics
