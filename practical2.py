import heapq


def a_star(graph, heuristic, start, goal):

    open_list = [(0, start)]

    g_cost = {node: float("inf") for node in graph}
    g_cost[start] = 0

    parent = {node: None for node in graph}

    while open_list:

        current_f, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]

            path.reverse()
            return path, g_cost[goal]

        for neighbor, cost in graph[current]:

            new_g_cost = g_cost[current] + cost

            if new_g_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_g_cost
                parent[neighbor] = current

                f_cost = new_g_cost + heuristic[neighbor]
                heapq.heappush(open_list, (f_cost, neighbor))

    return None, float("inf")


graph = {
    1: [(2, 2), (3, 4)],
    2: [(1, 2), (3, 1), (4, 5)],
    3: [(1, 4), (2, 1), (4, 2)],
    4: [(2, 5), (3, 2), (5, 3)],
    5: [(4, 3)]
}


heuristic = {
    1: 7,
    2: 6,
    3: 4,
    4: 2,
    5: 0
}


start = 1
goal = 5


path, cost = a_star(graph, heuristic, start, goal)


if path:
    print("Optimal Path:", " -> ".join(map(str, path)))
    print("Total Cost:", cost)
else:
    print("No path found.")