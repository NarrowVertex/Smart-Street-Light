from flask import Flask, request, jsonify
from flask import render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq
INF = float('inf')


class TrafficSystem:
    def __init__(self, graph, map_locations):
        self.graph = graph
        self.map_locations = map_locations

    def update_traffic(self, pi_id, traffic):
        for neighbors in self.graph:
            for j in range(len(neighbors)):
                if neighbors[j][0] == pi_id and neighbors[j][1] != INF:
                    neighbors[j] = (pi_id, traffic)

    def find_user_node(self, user_loc):
        min_distance = INF
        pos = -1
        for i, n in enumerate(self.map_locations):
            distance = (n[0] - user_loc[0]) ** 2 + (n[1] - user_loc[1]) ** 2
            if distance < min_distance:
                min_distance = distance
                pos = i+1
        return pos

    def get_neighboring_node_traffic(self, user_node):
        return self.graph[user_node - 1]
    
    def dijkstra(self, start):
        self.distance = [INF] * len(self.graph)
        q = []
        heapq.heappush(q, (0, start))
        self.distance[start - 1] = 0
        
        while q:
            dist, now = heapq.heappop(q)
            if self.distance[now - 1] < dist:
                continue
            for neighbor in self.graph[now - 1]:
                cost = dist + neighbor[1]
                if cost < self.distance[neighbor[0] - 1]:
                    self.distance[neighbor[0] - 1] = cost
                    heapq.heappush(q, (cost, neighbor[0]))

"""
graph = [
    [(1, 6)],  # Node 0 neighbors: Node 1
    [(0, 3), (2, 5), (4, 2)],  # Node 1 neighbors: Node 0, 2, 4
    [(1, 6)],      # Node 2 neighbors: Node 1
    [(4, 2)],       # Node 3 neighbors: Node 4
    [(1,6), (3,1), (5,4)],  # Node 4 neighbors: Node 1, 3, 5
    [(4, 2)]           # Node 5 neighbors: Node 4
]

map_locations = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2)
]
"""
graph = [
    [(1, 6)],  # Node 0 neighbors: Node 1
    [(0, 3), (2, 5), (4, 2)],  # Node 1 neighbors: Node 0, 2, 4
    [(1, 6)],      # Node 2 neighbors: Node 1
    [(4, 2)],       # Node 3 neighbors: Node 4
    [(1, 6), (3, 1), (5, 4)],  # Node 4 neighbors: Node 1, 3, 5
    [(4, 2)]           # Node 5 neighbors: Node 4
]

map_locations = [
    (0, 1),
    (1, 1),
    (2, 1),
    (0, 0),
    (1, 0),
    (2, 0)
]


# r, u, l, d
arrow_locations = [
    (0, None, None, 2),
    (1, None, 0, 3),
    (None, None, 1, 4),
    (5, 2, None, None),
    (6, 3, 5, None),
    (None, 4, 6, None)
]

traffic_system = TrafficSystem(graph, map_locations)

app = Flask(__name__)

address_book = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.json
    x = data['x']
    y = data['y']
    user_location = (int(x), int(y))
    user_node = traffic_system.find_user_node(user_location)
    neighboring_traffic = traffic_system.get_neighboring_node_traffic(user_node)
    text = ""
    min_node = -1
    min_traffic = INF
    for node in neighboring_traffic:
        text += f'{node[0]} {node[1]} '
        if min_traffic > node[1]:
            min_node = node[0]
            min_traffic = node[1]

    user_node -= 1
    min_node_location = map_locations[min_node]
    user_node_location = map_locations[user_node]
    dx = min_node_location[0] - user_node_location[0]
    dy = min_node_location[1] - user_node_location[1]

    direction = ""
    direction_index = -1
    if dx < 0:
        direction = "left"
        direction_index = 2
    elif dx > 0:
        direction = "right"
        direction_index = 0
    elif dy < 0:
        direction = "down"
        direction_index = 3
    elif dy > 0:
        direction = "up"
        direction_index = 1

    text += "|" + str(user_node) + "|" + str(arrow_locations[user_node][direction_index]) + " " + direction
    print(text)

    return text


@app.route('/traffic_input', methods=['POST'])
def traffic_input():
    data = request.json
    pi_id = int(data['pi_id'])
    traffic = int(data['traffic'])
    # contact = {'pi_id': pi_id, 'traffic': traffic}
    # address_book.append(contact)
    traffic_system.update_traffic(pi_id=pi_id, traffic=traffic)
    return jsonify({'status': 'success', 'received': data}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0")
