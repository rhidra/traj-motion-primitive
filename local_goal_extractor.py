import numpy as np

class LocalGoalExtractor:
    def __init__(self, path, initPos=[0, 0, 2], initVel=[.1, .1, 0]):
        self.path = path
        self.setPosition(initPos)
        self.setVelocity(initPos)

    def __iter__(self):
        return self
    
    def __next__(self):
        # Compute closest path point to the current position
        a, b = self.path[:-1], self.path[1:]
        p = np.repeat(self.pos.reshape(1,-1), a.shape[0], axis=0)
        d = ptSegmentDist(p, a, b)
        segIdx = np.argmin(d)
        segDist, segA, segB = d[segIdx], a[segIdx], b[segIdx]

        # Solve segment - sphere intersection
        # ref: https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        u = (segB - segA) / np.linalg.norm(segB - segA)
        normSqr = np.dot(segB - segA, segB - segA)
        delta = np.square(np.dot(u, (segA - self.pos))) - (np.linalg.norm(segA - self.pos) ** 2 - segDist*segDist)
        if np.isclose(delta, 0):
            d = - np.dot(u, (segA - self.pos))
            posProj = segA + d * u
        elif delta > 0:
            d1 = - np.dot(u, (segA - self.pos)) - np.sqrt(delta)
            d2 = - np.dot(u, (segA - self.pos)) + np.sqrt(delta)
            posProj1, posProj2 = segA + d1 * u, segA + d2 * u
            dot1 = np.dot(segB - segA, posProj1 - segA)
            posProj = posProj1 if 0 <= dot1 <= normSqr or np.isclose(dot1, 0) or np.isclose(dot1, normSqr) else posProj2
        else:
            raise ValueError('Cannot solve Segment/Sphere intersection during local goal extraction (delta={})'.format(delta))

        # Forward projection in the path
        proj = forwardProject(posProj, segIdx, self.path, distance=1*np.linalg.norm(self.vel))
        return proj

    def setPosition(self, pos):
        self.pos = np.array(pos)

    def setVelocity(self, vel):
        self.vel = np.array(vel)


def ptSegmentDist(p, a, b):
    """
    Compute min distance between points p and segments [a, b], in 3D.
    All points are shape (x, 3)
    """
    # normalized tangent vector
    d = (b - a) / np.linalg.norm(b - a, axis=1)[:, None]

    # signed parallel distance components
    s = np.sum((a - p) * d, axis=1)
    t = np.sum((p - b) * d, axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    c = np.cross(p - a, d)

    return np.hypot(h, np.linalg.norm(c, axis=1))


def forwardProject(initPt, initIdx, path, distance=2):
    idx = initIdx
    pt = initPt
    while True:
        if idx + 1 >= len(path):
            return path[-1]
        a, b = path[idx], path[idx + 1]
        norm = np.linalg.norm(b - a)
        proj = pt + (b - a) * distance / norm
        diff = np.linalg.norm(a - proj) - norm
        if diff > 0:
            idx += 1
            pt = b
            distance = diff
        else:
            return proj
    raise Exception()