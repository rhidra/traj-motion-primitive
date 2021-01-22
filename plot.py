import numpy as np, matplotlib.pyplot as plt, matplotlib.patches as patches, matplotlib.collections as collections
from utils import supercover, Node, lineOfSightNeighbors, lineOfSight, dist, phi

fig, ax = plt.subplots()

getPos = lambda x: x.pos if isinstance(x, Node) else x

def display(start=None, goal=None, grid_obs=[], globalPath=[], trajLibrary=[], trajSelected=None, tf=1, hold=False):
    print('plotting...')
    ax.clear()
    ax.set_xlim(-0.5, grid_obs.shape[0])
    ax.set_ylim(-0.5, grid_obs.shape[0])
    
    ax.set_title('Trajectory')

    obs = []
    x, y = np.mgrid[0:grid_obs.shape[0], 0:grid_obs.shape[1]]
    np.vectorize(lambda node, x, y: obs.append(patches.Rectangle([x, y], 1, 1)) if node == Node.OBSTACLE else None)(grid_obs, x, y)
    # obs = [patches.Rectangle([x, y], w, h) for x, y, w, h in extractRect(grid_obs)]
    ax.add_collection(collections.PatchCollection(obs))

    if start is not None:
        ax.add_patch(patches.Circle(getPos(start), .3, linewidth=1, facecolor='green'))
    if goal is not None:
        ax.add_patch(patches.Circle(getPos(goal), .3, linewidth=1, facecolor='blue'))
    
    if len(globalPath) > 0:
        ax.plot(globalPath[:, 0], globalPath[:, 1], 'o-', color='red', markersize=5)
    
    t = np.linspace(0, tf, 100)
    for traj in trajLibrary:
        pos = traj.get_position(t)
        ax.plot(pos[:, 0], pos[:, 1], color='b' if traj._cost < 10 else 'r', alpha=.3)

    if trajSelected is not None:
        pos = trajSelected.get_position(t)
        ax.plot(pos[:, 0], pos[:, 1], color='c', alpha=1)

    if hold and isinstance(hold, bool):
        plt.show()
    else:
        plt.pause(1 if isinstance(hold, bool) else hold)


def extractRect(grid):
    rects = []
    def alreadyDone(i, j):
        for x, y, w, h in rects:
            if x <= i < x+w and y <= j < y+h:
                return True
        return False

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not alreadyDone(i, j) and grid[i, j] == Node.OBSTACLE:
                for k in range(i+1, grid.shape[0]):
                    if grid[k, j] == Node.FREE:
                        break
                    k += 1
                imax = k
                for k in range(j+1, grid.shape[1]):
                    if grid[i:imax, j:k][grid[i:imax, j:k] == Node.FREE].size != 0:
                        break
                    k += 1
                jmax = k
                rects.append([i, j, imax-i, jmax-j])
    return rects
