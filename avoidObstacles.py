import numpy as np

def spherePlaneIntersection(n, x0, R):
    offset = np.einsum("i,ji->j", n, x0)
    projection = x0 - np.einsum("i,j->ij", offset, n)
    r = np.sqrt(R ** 2 - offset ** 2)
    return projection, r

def angle(v1, v2):
    costheta = np.einsum("ij,j->i", v1, v2)
    sintheta = np.linalg.norm(np.cross(v1, v2), axis=1)
    return np.arctan2(sintheta, costheta)

def queryClosestInterval(intervals, query):
    n = len(intervals)
    points = intervals.transpose().reshape(-1)
    changes = np.concatenate((np.ones(n), -np.ones(n)))
    sortedIndices = np.argsort(points)
    points = points[sortedIndices]
    changes = changes[sortedIndices]
    eventCount = np.concatenate(([0], np.cumsum(changes)))
    closestIndex = np.searchsorted(points, query)
    if eventCount[closestIndex] == 0:
        return query
    else:
        closestLeft = np.max(np.argwhere(eventCount[:closestIndex] == 0))
        closestRight = closestIndex + np.min(np.argwhere(eventCount[closestIndex:] == 0))
        if query - points[closestLeft] < points[closestRight - 1] - query:
            return points[closestLeft]
        else:
            return points[closestRight - 1]

def rotationMatrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def trimPointcloud(pointcloud, threshold_distance):
    return pointcloud[np.linalg.norm(pointcloud, axis=1) < threshold_distance]

def determineBestTrajectory(radius, length, pointcloud, threshold_distance, normal, desired):
    trimmed_pointcloud = trimPointcloud(pointcloud, threshold_distance)
    xs, rs = spherePlaneIntersection(normal, trimmed_pointcloud, radius)
    thetas = angle(xs, desired)
    dthetas = np.arcsin(radius / np.linalg.norm(xs, axis=1))
    intervals = np.vstack((thetas - dthetas, thetas + dthetas)).transpose()
    optimalTheta = queryClosestInterval(intervals, 0.0)
    return np.dot(rotationMatrix(normal, -optimalTheta), desired)

if __name__ == "__main__":
    print(determineBestTrajectory(0.5, 1.0, np.array([[0.25, 1, 0]]), np.array([0, 0, 1]), np.array([0, 1, 0])))
