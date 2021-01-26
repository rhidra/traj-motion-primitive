import numpy as np, math, sys, time, matplotlib.pyplot as plt, scipy.interpolate, scipy.ndimage
from noise import pnoise2
from functools import reduce
from utils import dist, Node, lineOfSight, phi, lineOfSightNeighbors, corners, pathLength, updateGridBlockedCells, NoPathFound
from collections import deque
from config import *

# Algorithm parameters
H_COST_WEIGHT = 1.7

# Parameters for map generation with Perlin noise
WIDTH = 50
HEIGHT = 50
OBSTACLE_THRESHOLD = .2
OBSTACLE_X_SIZE = 6
OBSTACLE_Y_SIZE = 6
OBSTACLE_X_OFFSET = 0
OBSTACLE_Y_OFFSET = 0

# Return all the children (neighbors) of a specific node
def children(node, grid, obs, crossbar=True, checkLOS=True):
    pos, c = np.array(node.pos), []
    if crossbar:
        directions = np.array([[1,0],[0,1],[0,-1],[-1,0]])
    else:
        directions = np.array([[1,0],[0,1],[1,1],[-1,-1],[1,-1],[-1,1],[0,-1],[-1,0]])
    for d in pos + directions:
        if 0 <= d[0] < grid.shape[0] and 0 <= d[1] < grid.shape[1] and (not checkLOS or lineOfSightNeighbors(node.pos, grid[d[0], d[1]].pos, obs)):
            c.append(grid[d[0], d[1]])
    return c


def pathTie(node, current):
    angle = abs(phi(current.parent, current, node))
    return math.isclose(angle, 180, abs_tol=1e-6)
    

def updateVertex(current, node, grid, obs):
    if current.parent and lineOfSight(current.parent, node, obs) \
                    and current.lb <= phi(current, current.parent, node) <= current.ub \
                    and not pathTie(node, current):
        # Path 2
        # If in line of sight, we connect to the parent, it avoid unecessary grid turns
        new_g = current.parent.G + dist(current.parent, node)
        showPath2 = True
        if new_g < node.G:
            node.G = new_g
            node.parent = current.parent
            node.local = current
            neighbors = list(map(lambda nb: phi(node, current.parent, nb), children(node, grid, obs, crossbar=True)))
            l = min(neighbors)
            h = max(neighbors)
            delta = phi(current, current.parent, node)
            node.lb = max(l, current.lb - delta)
            node.ub = min(h, current.ub - delta)
    else:
        # Path 1
        showPath2 = False
        new_g = current.G + dist(current, node)
        if new_g < node.G:
            node.G = new_g
            node.parent = current
            node.local = current
            node.lb = -45
            node.ub = 45

    return showPath2

# Return the path computed by the A* optimized algorithm from the start and goal points
def find_path(start, goal, grid, obs, openset=set(), closedset=set()):
    startTime = time.time()
    if len(openset) == 0:
        openset.add(start)

    i = 0
    while openset and min(map(lambda o: o.G + H_COST_WEIGHT * o.H, openset)) < goal.G + H_COST_WEIGHT * goal.H and time.time() - startTime < TIME_OUT:
        i = i + 1
        current = min(openset, key=lambda o: o.G + H_COST_WEIGHT * o.H)

        openset.remove(current)
        closedset.add(current)

        # Loop through the node's children/siblings
        for node in children(current, grid, obs, crossbar=False):
            # If it is already in the closed set, skip it
            if node in closedset:
                continue

            if node not in openset:
                node.G = float('inf')
                node.H = dist(node, goal)
                node.parent = None
                node.ub = float('inf')
                node.lb = - float('inf')
                openset.add(node)
            
            updateVertex(current, node, grid, obs)

    if not goal.parent:
        print('  No path found !')
        raise NoPathFound
    
    path = []
    current = goal
    while current.parent:
        path.append(current)
        current = current.parent
    path.append(current)
    return path[::-1]


def clearSubtree(node, grid, obs, openset, closedset):
    under, over = deque(), deque()
    under.append(node)

    while under:
        node = under.popleft()
        over.append(node)
        node.reset()

        openset.discard(node)
        closedset.discard(node)

        for neigh in children(node, grid, [], crossbar=False, checkLOS=False):
            if neigh.local == node:
                under.append(neigh)
    
    while over:
        node = over.popleft()
        for neigh in children(node, grid, obs, crossbar=False, checkLOS=True):
            if neigh in closedset:
                g_old = node.G
                updateVertex(neigh, node, grid, obs)
                if node.G < g_old:
                    openset.add(node)


def phi_star(start, goal, grid_obs, newBlockedCells=[]):
    x, y = np.mgrid[0:grid_obs.shape[0]+1, 0:grid_obs.shape[1]+1]
    grid = np.vectorize(Node)(x, y)
    start, goal = grid[start], grid[goal]
    goal.H, start.G, start.H = 0, 0, dist(start, goal)

    newBlockedCells = iter(newBlockedCells)
    openset = set()
    closedset = set()

    path = find_path(start, goal, grid, grid_obs, openset, closedset)
    return path


def main():
    start = (0, 0)
    goal = (WIDTH-1, HEIGHT-1)

    x_obs, y_obs = np.mgrid[0:WIDTH-1, 0:HEIGHT-1]
    grid_obs = np.vectorize(pnoise2)(x_obs / OBSTACLE_X_SIZE, y_obs / OBSTACLE_Y_SIZE)
    grid_obs[grid_obs > OBSTACLE_THRESHOLD] = Node.OBSTACLE
    grid_obs[grid_obs <= OBSTACLE_THRESHOLD] = Node.FREE
    grid_obs[start], grid_obs[goal[0]-1, goal[1]-1] = Node.FREE, Node.FREE

    path = phi_star(start, goal, grid_obs)
    print('Phi* done')
    
    # Upsample to reduce the closeness of the obstacles to the path
    # This is equivalent to artificially increasing the expected size
    # of the obstacles for the global planning algorithm
    UPSCALING_FACTOR = 4
    new_grid = grid_obs.repeat(UPSCALING_FACTOR, axis=0).repeat(UPSCALING_FACTOR, axis=1)
    new_grid = scipy.ndimage.binary_erosion(new_grid, structure=np.ones((6, 6))).astype(new_grid.dtype)
    path = np.array([p.pos for p in path]) * UPSCALING_FACTOR
    goal = (goal[0]*UPSCALING_FACTOR, goal[1]*UPSCALING_FACTOR)
    print('Upscaling done')

    return path, new_grid, start, goal


if __name__ == '__main__':
    main()
