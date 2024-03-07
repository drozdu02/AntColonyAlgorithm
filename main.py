import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class AntColonyAlgorithm:
    def __init__(self, num_of_ants, alpha, beta, evaporation_rate, q, graph, points):
        self.num_of_ants = num_of_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        self.graph = graph
        self.points = points
        self.num_of_points = len(points)
        self.pheromones = np.ones((self.num_of_points, self.num_of_points))
        self.sorted_keys = sorted(self.points.keys())

    def calculate_distance_matrix(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def update_pheromones(self, visited_nodes):
        for i in range(len(visited_nodes) - 1):
            current_node = visited_nodes[i]
            next_node = visited_nodes[i + 1]
            self.pheromones[current_node][next_node] *= (1 - self.evaporation_rate)
            self.pheromones[current_node][next_node] += self.q / self.calculate_distance_matrix(
                self.points[self.sorted_keys[current_node]], self.points[self.sorted_keys[next_node]])

    def ant_step(self):
        best_path_length = float('inf')
        best_path = []
        all_paths = []
        for _ in range(self.num_of_ants):
            start_node = list(self.points.keys()).index(start_point)
            visited_nodes = [start_node]
            while len(visited_nodes) < self.num_of_points:
                current_node = visited_nodes[-1]
                unvisited_nodes = [node for node in range(self.num_of_points) if node not in visited_nodes]
                if not unvisited_nodes:
                    break
                probabilities = [self.calculate_probability(current_node, next_node) for next_node in unvisited_nodes]
                total_probability = sum(probabilities)
                probabilities = [prob / total_probability for prob in probabilities]
                next_node = random.choices(unvisited_nodes, probabilities)[0]
                visited_nodes.append(next_node)
            all_paths.append(visited_nodes)
            path_length = sum([self.calculate_distance_matrix(self.points[self.sorted_keys[visited_nodes[i]]],
                                                              self.points[self.sorted_keys[visited_nodes[i + 1]]]) for i
                               in range(len(visited_nodes) - 1)])
            if path_length < best_path_length:
                best_path_length = path_length
                best_path = visited_nodes
            self.update_pheromones(visited_nodes)
        return best_path, all_paths

    def calculate_probability(self, current_node, next_node):
        point1 = self.points[self.sorted_keys[current_node]]
        point2 = self.points[self.sorted_keys[next_node]]
        pheromone = self.pheromones[current_node][next_node]
        visibility = 1 / self.calculate_distance_matrix(point1, point2)
        return (pheromone ** self.alpha) * (visibility ** self.beta)

class GeneratePlot(AntColonyAlgorithm):
    def __init__(self, points, all_paths):
        self.points = points
        self.all_paths = all_paths

    @staticmethod
    def plot_path(points, all_paths):
        colors = plt.cm.rainbow(np.linspace(0, 1, len(all_paths)))
        for path, color in zip(all_paths, colors):
            path_points = [list(points.values())[i] for i in path]
            path_points.append(path_points[0])  # Correct indentation
            path_x, path_y = zip(*path_points)
            plt.plot(path_x, path_y, marker='o', color=color)
        for point, (x, y) in points.items():
            plt.text(x, y, point, fontsize=12)
        plt.title("Trasy mrówek")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()
# fig, ax = plt.subplots()
#
# def update_frame(frame):
#     colors = plt.cm.rainbow(np.linspace(0, 1, len(all_paths)))
#     path = all_paths[frame]
#     path_points = [list(points.values())[i] for i in path]
#     path_points.append(path_points[0])
#     path_x, path_y = zip(*path_points)
#     ax.clear()
#     ax.plot(path_x, path_y, marker='o', color = colors[frame])
#     for point, (x, y) in points.items():
#         ax.text(x, y, point, fontsize=12)
#
#     ax.set_title(f"Ruch mrówek - Krok {frame + 1}")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.grid(True)



if __name__ == "__main__":
    points = {}
    num_points = int(input("Podaj liczbę punktów: "))
    for i in range(num_points):
        point_name = input(f"Podaj nazwę punktu {i + 1}: ")
        x, y = eval(input(f"Podaj współrzędne (x, y) dla punktu {point_name}: "))
        points[point_name] = (x, y)

    start_point = input("Podaj nazwę punktu startowego: ")

    distance_matrix = np.zeros((len(points), len(points)))
    for i, (point1name, point1) in enumerate(points.items()):
        for j, (point2name, point2) in enumerate(points.items()):
            distance_matrix[i][j] = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    colony = AntColonyAlgorithm(num_of_ants=100, alpha=1, beta=2, evaporation_rate=0.5, q=1, graph=distance_matrix,
                                points=points)
    best_path, all_paths = colony.ant_step()
    print("Najkrótsza ścieżka:", [list(points.keys())[node] for node in best_path])
    print("Wszystkie ścieżki mrówek:")
    for i, path in enumerate(all_paths):
        print(f"Mrówka {i + 1}: {[list(points.keys())[node] for node in path]}")

    shortest_path_length = sum([colony.calculate_distance_matrix(points[list(points.keys())[best_path[i]]],
                                                                 points[list(points.keys())[best_path[i + 1]]]) for i in range(len(best_path) - 1)])
    print(f"Długość najkrótszej ścieżki: {shortest_path_length}")
    print(all_paths)

    _, all_paths = colony.ant_step()
    GeneratePlot.plot_path(points, all_paths)
    # ani = FuncAnimation(fig, update_frame, frames=len(all_paths), interval=1000)
    # plt.show()