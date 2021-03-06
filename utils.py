import math, numpy as np

def getPos(node):
    return node.pos if isinstance(node, Node) else node

def dist(p1, p2, sqrt=True):
    if p1 is None or p2 is None:
        return math.inf
    p1, p2 = getPos(p1), getPos(p2)
    sqr = (p1[0] - p2[0])**2 + (p1[1]-p2[1])**2
    return math.sqrt(sqr) if sqrt else sqr


# Utility class to handle nodes more easily
class Node:
    OBSTACLE = 1
    FREE = 0

    def __init__(self, x, y):
        self.pos = [x, y]
        self.reset()

    def reset(self):
        self.parent = None
        self.local = None
        self.H = 0
        self.G = math.inf
        self.lb = -math.inf
        self.ub = math.inf


    def __repr__(self):
        return 'Node: ' + self.pos.__repr__()

class NoPathFound(BaseException):
    pass

def phi(a,b,c):
    a, b, c = getPos(a), getPos(b), getPos(c)
    angle = (180./math.pi) * (-math.atan2(a[1]-b[1], a[0]-b[0]) + math.atan2(c[1]-b[1], c[0]-b[0]))
    if angle > 180:
        angle = - (180 - (angle % 180)) # Set angle between [-180; 180]
    return angle


def phi3d_(a,b,c):
    a, b, c = getPos(a), getPos(b), getPos(c)
    p = (180./math.pi) * (-math.atan2(a[1]-b[1], a[0]-b[0]) + math.atan2(c[1]-b[1], c[0]-b[0]))
    if p > 180:
        p = - (180 - (p % 180)) # Set angle between [-180; 180]

    t = (180./math.pi) * (-math.atan2(a[2]-b[2], math.sqrt((a[0]-b[0]) ** 2 + (a[1]-b[1]) ** 2)) + math.atan2(c[2]-b[2], math.sqrt((c[0]-b[0]) ** 2 + (c[1]-b[1]) ** 2)))
    if t > 180:
        t = - (180 - (t % 180)) # Set angle between [-180; 180]
    return p, t


def phi3d(ax,ay,az, bx,by,bz ,cx,cy,cz):
    p = (180./math.pi) * (-math.atan2(ay-by, ax-bx) + math.atan2(cy-by, cx-bx))
    if p > 180:
        p = - (180 - (p % 180)) # Set angle between [-180; 180]

    t = (180./math.pi) * (-math.atan2(az-bz, math.sqrt((ax-bx) ** 2 + (ay-by) ** 2)) + math.atan2(cz-bz, math.sqrt((cx-bx) ** 2 + (cy-by) ** 2)))
    if t > 180:
        t = - (180 - (t % 180)) # Set angle between [-180; 180]
    return p, t


# Add points in a path
def over_sampling(path, max_length=5):
    new = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        new.append(a)
        last = a
        norm = dist(a, b)
        dir_x = (b[0]-a[0]) * max_length / norm
        dir_y = (b[1]-a[1]) * max_length / norm
        while dist(last, b) > max_length * max_length:
            last = [last[0] + dir_x, last[1] + dir_y]
            new.append(last)
    new.append(path[-1])
    return new

# Return the list of corner coordinates for all block cells
def corners(cells):
    pts = set()
    for x, y in cells:
        pts.add((x, y))
        pts.add((x + 1, y))
        pts.add((x + 1, y + 1))
        pts.add((x, y + 1))
    return pts

def updateGridBlockedCells(blockedCells, grid):
    for x, y in blockedCells:
        grid[x, y] = Node.OBSTACLE if grid[x, y] == Node.FREE else Node.FREE

# Improved Bresenham algorithm to retrieve the entire super cover of the line
def supercover(a, b):
    (x1, y1), (x2, y2) = a.pos, b.pos

    x1 = x1 - 1 * (x2 - x1 < 0)
    y1 = y1 - 1 * (y2 - y1 < 0)
    x2 = x2 - 1 * (x2 - x1 > 0)
    y2 = y2 - 1 * (y2 - y1 > 0)

    dx, dy = x2 - x1, y2 - y1
    x, y = x1, y1
    pts = []
    pts.append([x1, y1])

    if dy < 0:
        ystep = -1
        dy = -dy
    else:
        ystep = 1
    
    if dx < 0:
        dx = -dx
        xstep = -1
    else:
        xstep = 1
    
    ddy = 2 * dy
    ddx = 2 * dx

    if ddx >= ddy:
        error = dx
        errorprev = dx

        for _ in range(dx):
            x += xstep
            error += ddy
            if error > ddx:
                y += ystep
                error -= ddx
                if error + errorprev < ddx:
                    pts.append([x, y-ystep])
                elif error + errorprev > ddx:
                    pts.append([x-xstep, y])
                else:
                    pts.append([x, y-ystep])
                    pts.append([x-xstep, y])
            pts.append([x, y])
            errorprev = error
    else:
        errorprev = dy
        error = dy
        for _ in range(dy):
            y += ystep
            error += ddx
            if error > ddy:
                x += xstep
                error -= ddy
                if error + errorprev < ddy:
                    pts.append([x-xstep, y])
                elif error + errorprev > ddy:
                    pts.append([x, y-ystep])
                else:
                    pts.append([x-xstep, y])
                    pts.append([x, y-ystep])
            pts.append([x, y])
            errorprev = error
    return pts


def lineOfSight(a, b, obs):
    for pt in supercover(a, b):
        try:
            if obs[pt[0], pt[1]] == Node.OBSTACLE:
                return False
        except IndexError:
            continue
    return True


def lineOfSightNeighbors(a, b, obs):
    (xa, ya), (xb, yb) = a, b
    dx, dy = xb - xa, yb - ya

    try:
        if dx != 0 and dy != 0:
            # Diagonal case
            return obs[xa - 1 * (dx < 0), ya - 1 * (dy < 0)] == Node.FREE
        elif dx == 0:
            # Vertical case
            yobs = ya - 1 * (dy < 0)
            return obs[xa, yobs] == Node.FREE or obs[xa - 1, yobs] == Node.FREE
        else:
            # Horizontal case
            xobs = xa - 1 * (dx < 0)
            return obs[xobs, ya] == Node.FREE or obs[xobs, ya - 1] == Node.FREE
    except IndexError:
        return True


# Bresenham (1950) fast line of sight algorithm for computer graphics
def lineOfSight_(a, b, grid):
    (x0, y0), (x1, y1) = a.pos, b.pos
    dx, dy = x1 - x0, y1 - y0
    f = 0
    
    if dy < 0:
        dy = -dy
        sy = -1
    else:
        sy = 1
    
    if dx < 0:
        dx = -dx
        sx = -1
    else:
        sx = 1
    
    if dx >= dy:
        while x0 != x1:
            f = f + dy
            if f >= dx:
                if grid[x0 + int((sx-1)/2), y0 + int((sy-1)/2)].value == Node.OBSTACLE:
                    return False
                y0 = y0 + sy
                f = f - dx
            if f != 0 and grid[x0 + int((sx-1)/2), y0 + int((sy-1)/2)].value == Node.OBSTACLE:
                return False
            if dy == 0 and grid[x0 + int((sx-1)/2), y0].value == Node.OBSTACLE and grid[x0 + int((sx-1)/2), y0-1].value == Node.OBSTACLE:
                return False
            x0 = x0 + sx
    else:
        while y0 != y1:
            f = f + dx
            if f >= dy:
                if grid[x0 + int((sx-1)/2), y0 + int((sy-1)/2)].value == Node.OBSTACLE:
                    return False
                x0 = x0 + sx
                f = f - dy
            if f != 0 and grid[x0 + int((sx-1)/2), y0 + int((sy-1)/2)].value == Node.OBSTACLE:
                return False
            if dx == 0 and grid[x0, y0 + int((sy-1)/2)].value == Node.OBSTACLE and grid[x0-1, y0 + int((sy-1)/2)].value == Node.OBSTACLE:
                return False
            y0 = y0 + sy
    return True


def pathLength(path):
    l, prev = 0, None
    for pt in path:
        if prev is not None:
            l += dist(prev, pt)
        prev = pt
    return l