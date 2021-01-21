import numpy as np, matplotlib.pyplot as plt, math
from quadrocoptertrajectory import RapidTrajectory


Tf = 1

def generateTraj(pos0, vel0, acc0, velf):
    traj = RapidTrajectory(pos0, vel0, acc0, [0,0,-9.81])
    traj.set_goal_velocity(velf)
    traj.set_goal_acceleration([0, 0, 0])
    traj.generate(Tf)
    return traj

def generateTrajLibrary(pos0, vel0, acc0):
    numAngleVariation = 11
    numNormVariation = 4
    velxOffset = 1
    velyOffset = 1
    velzOffset = .1

    theta0 = math.atan2(vel0[1], vel0[0])
    norm0 = np.sqrt(vel0[0]*vel0[0] + vel0[1]*vel0[1])

    trajs = []
    for theta in np.linspace(theta0 - np.pi/2, theta0 + np.pi/2, numAngleVariation):
        for norm in np.linspace(np.clip(norm0 - 2, 0, 1e5), norm0 + 2, numNormVariation):
            trajs.append(generateTraj(pos0, vel0, acc0, [norm * np.cos(theta), norm * np.sin(theta), vel0[2]]))
    # for velz in np.linspace(vel0[2] - velzOffset, vel0[2] + velzOffset, numTrajs):

    return trajs


def main():

    # Define the trajectory starting state:
    pos0 = [0, 0, 2]
    vel0 = [1, 0, 0]
    acc0 = [0, 0, 0]

    trajs = generateTrajLibrary(pos0, vel0, acc0)

    # Test input feasibility
    # inputsFeasible = traj.check_input_feasibility(fmin, fmax, wmax, minTimeSec)
    for traj in trajs:
        numPlotPoints = 100
        time = np.linspace(0, Tf, numPlotPoints)
        position = np.zeros([numPlotPoints, 3])
        velocity = np.zeros([numPlotPoints, 3])
        acceleration = np.zeros([numPlotPoints, 3])
        thrust = np.zeros([numPlotPoints, 1])
        ratesMagn = np.zeros([numPlotPoints,1])

        for i in range(numPlotPoints):
            t = time[i]
            position[i, :] = traj.get_position(t)
            velocity[i, :] = traj.get_velocity(t)
            acceleration[i, :] = traj.get_acceleration(t)
            thrust[i] = traj.get_thrust(t)
            ratesMagn[i] = np.linalg.norm(traj.get_body_rates(t))

        plt.subplot(211)
        plt.plot(position[:, 0], position[:, 1], color='b', alpha=.3)

        plt.subplot(212)
        plt.plot(position[:, 0], position[:, 2], color='b', alpha=.3)

    plt.show()



if __name__ == '__main__':
    main()