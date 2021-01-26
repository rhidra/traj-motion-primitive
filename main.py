import numpy as np, matplotlib.pyplot as plt, math, plot
from motion_primitive import MotionPrimitive
from phi_star import main as phi_star_gen
from utils import over_sampling, Node, dist
from local_goal_extractor import LocalGoalExtractor
np.seterr(divide='ignore')

OBSTACLE_DISTANCE_THRESHOLD = 10
Tf = 1

def generateTraj(pos0, vel0, acc0, velf):
    traj = MotionPrimitive(pos0, vel0, acc0, [0,0,-9.81])
    traj.set_goal_velocity(velf)
    traj.set_goal_acceleration([0, 0, 0])
    traj.generate(Tf)
    return traj

def generateTrajLibrary(pos0, vel0, acc0):
    numAngleVariation = 21
    numNormVariation = 10

    theta0 = math.atan2(vel0[1], vel0[0])
    norm0 = np.sqrt(vel0[0]*vel0[0] + vel0[1]*vel0[1])

    trajs = []
    for theta in np.linspace(theta0 - np.pi*.45, theta0 + np.pi*.45, numAngleVariation):
        for norm in np.linspace(np.clip(norm0 - 2, 0, 1e5), norm0 + 4, numNormVariation):
            trajs.append(generateTraj(pos0, vel0, acc0, [norm * np.cos(theta), norm * np.sin(theta), vel0[2]]))
    # for velz in np.linspace(vel0[2] - velzOffset, vel0[2] + velzOffset, numTrajs):

    return trajs

def euclideanDistanceTransform(grid_obs):
    edt = np.zeros(grid_obs.shape)
    x, y = np.meshgrid(np.arange(grid_obs.shape[0]), np.arange(grid_obs.shape[1]))
    for pt in zip(x.reshape(-1), y.reshape(-1)):
        i0, i1 = math.floor(pt[0] - 1 - OBSTACLE_DISTANCE_THRESHOLD), math.floor(pt[0] + 1 + OBSTACLE_DISTANCE_THRESHOLD)
        i0, i1 = np.clip(i0, 0, grid_obs.shape[0]-1), np.clip(i1, 0, grid_obs.shape[0]-1)
        j0, j1 = math.floor(pt[1] - 1 - OBSTACLE_DISTANCE_THRESHOLD), math.floor(pt[1] + 1 + OBSTACLE_DISTANCE_THRESHOLD)
        j0, j1 = np.clip(j0, 0, grid_obs.shape[1]-1), np.clip(j1, 0, grid_obs.shape[1]-1)
        score = np.inf

        for i, row in enumerate(grid_obs[i0:i1]):
            for j, val in enumerate(row[j0:j1]):
                if val == Node.OBSTACLE:
                    d = dist(pt, np.array([i+i0, j+j0]))
                    if d < score:
                        score = d
        edt[pt] = score
    return edt


def main():
    path, grid_obs, start, goal = phi_star_gen()

    # path = np.array(over_sampling([p.pos for p in path], max_length=1))
    # path = np.array([p.pos for p in path])
    path = np.concatenate((path, np.array([2] * len(path)).reshape(-1, 1)), axis=1)
    edt = euclideanDistanceTransform(grid_obs)
    print('EDT done')

    pos0 = [0, 0, 2]
    vel0 = [.1, .1, 0]
    acc0 = [0, 0, 0]

    localGoalExtractor = LocalGoalExtractor(path, initPos=pos0, initVel=vel0)
    previousTraj = []

    for i, goalLocal in enumerate(localGoalExtractor):
        trajs = generateTrajLibrary(pos0, vel0, acc0)

        for traj in trajs:
            traj.compute_cost(goalLocal, edt)

        trajSelected = min(trajs, key=lambda t: t._cost)

        if trajSelected._cost == np.inf:
            plot.display(start, goal, grid_obs, edt=edt, globalPath=path, trajLibrary=trajs, trajHistory=previousTraj, point=goalLocal, tf=Tf)
            raise ValueError('Cannot find an appropriate trajectory')
        
        pos0 = trajSelected.get_position(Tf)
        vel0 = trajSelected.get_velocity(Tf)
        acc0 = trajSelected.get_acceleration(Tf)

        localGoalExtractor.setPosition(pos0)
        localGoalExtractor.setVelocity(vel0)

        # Test input feasibility
        # inputsFeasible = traj.check_input_feasibility(fmin, fmax, wmax, minTimeSec)

        if i % 10 == 0:
            plot.display(start, goal, grid_obs, edt=edt, globalPath=path, trajLibrary=trajs, trajSelected=trajSelected, trajHistory=previousTraj, point=goalLocal, tf=Tf)
        previousTraj.append(trajSelected)


if __name__ == '__main__':
    main()