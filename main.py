import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from object_detection import ObjectDetection
from object_tracking import ObjectTracking
from plane_projection import PlaneProjection
from teams_separation import SeparateTeams
from teams_possession import Possession
from players_statistics import Statistics
from goal_detection import GoalDetection
from heat_map import HeatMap

# Read and Write video
PATH = "Argentina.mp4"
cap = cv2.VideoCapture(PATH)
output = cv2.VideoWriter(PATH + '_output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20, (1280, 720))
plane_out = cv2.VideoWriter(PATH + '_plane.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20, (1280, 425))
heat_map_out = cv2.VideoWriter(PATH + '_heat_map_output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20,
                               (1280, 425))

# Initialize instances
ObjectDetection = ObjectDetection()
ObjectTracking = ObjectTracking(distance_threshold=50, not_exist_threshold=10)
PlaneProjection = PlaneProjection()
SeparateTeams = SeparateTeams(path=PATH)
colors = SeparateTeams.clf_colors()
Possession = Possession(distance_threshold=100, possession_threshold=10, missed_possession_threshold=5)
Statistics = Statistics(distance_threshold=100)
GoalDetection = GoalDetection(distance_threshold=150)
HeatMap = HeatMap(frame_count_threshold=15)

# Initialize variables
teams_colors = {'Team A': 0, 'Team B': 1, 'Outlier': 2}
players_info = defaultdict(
    lambda: {'Position': [], 'Team': 'None', 'Possession': 0, 'Missed': 0, 'Mileage(PX)': 0, 'Mileage(M)': 0})
ball_holder_team_color = None
goal_flag = False

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print('Frame Count:', frame_count)
    print()

    frame = cv2.resize(frame, (1280, 720))
    frame_with_possession = frame.copy()

    # Detect objects on frame
    (boxes, confidences, class_ids) = ObjectDetection.detect(frame)

    # Prepare players' boxes for teams' separation
    players_boxes = []
    for i, box in enumerate(boxes):
        if class_ids[i] == 0:
            players_boxes.append(box)

    # Get most confidence Ball box
    ball_position = None
    ball_indices = list(np.where(np.array(class_ids) == 1)[0])
    if ball_indices:
        ball_confidences = list(np.array(confidences)[list(ball_indices)])
        most_confidence_ball_index = ball_indices[np.argmax(ball_confidences)]
        ball_position = boxes[most_confidence_ball_index]

        # Draw Ball
        (x, y, w, h) = ball_position
        cv2.rectangle(frame, (x, y), (x + w, y + h), [0, 0, 0], 2)
        cv2.putText(frame, "Ball", (round(x) - 10, round(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 0, 0], 2)

    # Get most confidence Goal box
    goal_position = None
    goal_indices = list(np.where(np.array(class_ids) == 2)[0])
    if goal_indices:
        goal_confidences = list(np.array(confidences)[list(goal_indices)])
        most_confidence_goal_index = goal_indices[np.argmax(goal_confidences)]
        goal_position = boxes[most_confidence_goal_index]

        # Draw Goal
        (x, y, w, h) = goal_position
        cv2.rectangle(frame, (x, y), (x + w, y + h), [0, 0, 0], 2)
        cv2.putText(frame, "Goal", (round(x) - 10, round(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 0, 0], 2)

    # Check goal or not
    if goal_position and ball_position:
        goal_flag = GoalDetection.check_goal(goal_position, ball_position)

    # Track objects on frame
    tracked_players, disappeared_players_count, new_players, distances = ObjectTracking.track(players_boxes)

    # Separate Teams
    results = SeparateTeams.separate(frame, players_boxes)

    # Update players_info dictionary with modified positions or teams of player or new players if exist
    for i, box in enumerate(players_boxes):
        (x, y, w, h) = box

        teams_list = ['Team A', 'Team B', 'Outlier']
        index = list(tracked_players.values()).index(box)
        object_id = list(tracked_players.keys())[index]

        index = results[i]
        players_info[object_id]['Position'] = box
        players_info[object_id]['Team'] = teams_list[index]

    # Extract Teams' possession
    cur_players_info = defaultdict(
        lambda: {'Position': [], 'Team': 'None', 'Possession': 0, 'Missed': 0, 'Mileage(PX)': 0, 'Mileage(M)': 0})
    for player_id in players_info.keys():
        if player_id in tracked_players:
            cur_players_info[player_id] = players_info[player_id]
    ball_holder_player, ball_holder_team = Possession.get_possession(cur_players_info, ball_position)

    # Extract Players' statistics
    players_statistics = Statistics.get_statistics(cur_players_info, ball_position)

    # Combine players Mileages with their info
    for object_id, object_distance in distances.items():
        players_info[object_id]['Mileage(PX)'] += object_distance
        players_info[object_id]['Mileage(M)'] += object_distance / 9.691176471

    # Combine players Mileages with their info
    if players_statistics:
        for object_id, possession_missed in players_statistics.items():
            possession, missed = possession_missed['Possession'], possession_missed['Missed']
            players_info[object_id]['Possession'] = possession
            players_info[object_id]['Missed'] = missed

    # Draw Players
    players_boxes = []
    players_teams = []
    for player_id, player_info in players_info.items():

        if player_id not in tracked_players.keys() or player_id in disappeared_players_count.keys():
            continue

        (x, y, w, h) = player_info['Position']
        player_team = player_info['Team']
        team_color = teams_colors[player_team]
        players_boxes.append(player_info['Position'])
        players_teams.append(player_team)

        text = str(player_team) + ' ' + str(player_id)
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors[team_color], 2)
        cv2.putText(frame, text, (round(x) - 10, round(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[team_color], 2)

        # Draw Possession box
        frame_with_possession = frame.copy()
        if ball_holder_player:
            ball_holder_team = players_info[ball_holder_player]['Team']
            ball_holder_team_color = colors[teams_colors[ball_holder_team]]

        # # Background box color equal ball_holder_team_color
        # cv2.rectangle(frame_with_possession, [1000, 0, 280, 170], ball_holder_team_color, -1)

        # Black Background box color
        cv2.rectangle(frame_with_possession, [1000, 0, 280, 170], [0, 0, 0], -1)

        v = 0
        for team, value in Possession.teams_possession.items():
            team_possession_value = value / sum(Possession.teams_possession.values())
            team_possession_ratio = "{0:.1f}".format(team_possession_value * 100)
            text = str(team) + ' possession ratio: {} %'.format(team_possession_ratio)
            cv2.putText(frame_with_possession, text, (1010, 30 + v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
            v += 30

        if ball_holder_player:
            text = 'Ball-holder Team: ' + str(ball_holder_team)
            cv2.putText(frame_with_possession, text, (1010, 30 + v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
            v += 30
            text = 'Ball-holder Player: ' + str(ball_holder_player)
            cv2.putText(frame_with_possession, text, (1010, 30 + v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
            v += 30

        if goal_flag:
            text = 'GOOAAL!!!'
            cv2.putText(frame_with_possession, text, (1010, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
    # *************************************

    print('Players information')
    print(*zip(players_info.items()))
    print()

    print('Tracked players')
    print(tracked_players)
    print()

    print('New Players')
    print(new_players)
    print()

    print('Distance between cur and prev position of player')
    print(distances)
    print()

    print('Disappeared Players count')
    print(disappeared_players_count)
    print()

    print("Teams Possession:", *zip(Possession.teams_possession.items()))
    print()
    print("Players Statistics:", *zip(Statistics.players_statistics.items()))
    print()
    print('Ball-holder Team:', ball_holder_team)
    print('Ball-holder Player:', ball_holder_player)
    print()

    players_info_df = pd.DataFrame.from_dict(players_info, orient='index')
    print(players_info_df)
    print()
    print("**********************************************")

    # *************************************

    # Plane
    plane = PlaneProjection.draw_plan(players_boxes, players_teams, colors)

    # Heat Map
    heat_map = HeatMap.heatMap(players_info)

    plane_overview = np.hstack((plane, cv2.resize(frame, (640, 425))))
    heat_map_overview = np.hstack((heat_map, cv2.resize(frame, (640, 425))))
    cv2.imshow("Frame", frame_with_possession)
    cv2.imshow('Plane', plane_overview)
    cv2.imshow("Heat Map", heat_map_overview)
    output.write(frame_with_possession)
    plane_out.write(plane_overview)
    heat_map_out.write(heat_map_overview)

    # Save Statistics in CSV file
    players_info_df.to_csv(PATH + '_statistics.csv')

    # Save Statistics in text file
    with open(PATH + '_statistics.txt', 'w') as statistics_file:
        statistics_file.write(str(players_info_df))

    # *************************************

    key = cv2.waitKey(1)
    if key == 27:
        break

statistics_file.close()
cap.release()
output.release()
plane_out.release()
heat_map_out.release()
cv2.destroyAllWindows()
