from collections import deque

graph = {
    1: [2, 3],
    2: [4, 5],
    3: [6, 7],
    4: [],
    5: [8],
    6: [],
    7: [],
    8: []
}

def bfs_traversal(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    print("BFS Traversal Order:", end=" ")

    while queue:
        current = queue.popleft()
        print(current, end=" ")

        for neighbour in graph.get(current, []):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

print("Following is the Breadth-First Search")
bfs_traversal(graph, 1)