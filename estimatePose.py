import numpy as np
import cv2
from optparse import OptionParser
import pcl
import matplotlib.pyplot as plt
import scipy.spatial
import time
import icp

def estimate_transform(A, B):
    assert len(A) == len(B)

    N = A.shape[0]

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    H = np.matmul(np.transpose(AA), BB)

    U, S, Vt = np.linalg.svd(H)

    R = np.matmul(Vt.T, U.T)

    t = np.matmul(-R, centroid_A.T) + centroid_B.T

    transform = np.hstack((R, t.reshape(3, 1)))
    return transform


def merge_pointclouds_no_filtering(p_source, p_target, resolution):
    return pcl.PointCloud(np.vstack((p_source.to_array(), p_target.to_array())))

def merge_pointclouds_kdtree(p_source, p_target, resolution):
    points_source = p_source.to_array()
    points_target = p_target.to_array()
    tree_target = scipy.spatial.KDTree(points_target)
    distances, indices = tree_target.query(points_source)
    filtered_points_source = [points_source[i] for i, distance in enumerate(distances) if distance < resolution]
    return pcl.PointCloud(np.vstack((points_target, filtered_points_source)))

merge_pointclouds = merge_pointclouds_no_filtering

def estimate_poses_icp(pointclouds):
    icp = pcl.IterativeClosestPoint()

    poses = [np.eye(4)]

    p_source = pcl.PointCloud(pointclouds[0][:,:3].astype("float32"))
    for i in range(1, len(pointclouds)):
        p_target = pcl.PointCloud(pointclouds[i][:,:3].astype("float32"))
        converged, transform, p_estimate, fitness = icp.icp(p_source, p_target)
        if converged:
            p_source = merge_pointclouds(p_estimate, p_target, None)
            poses.append(np.matmul(poses[i-1], np.linalg.inv(transform)))
            print(poses[i])

    return np.array(poses)

def estimate_poses_matching(images, projections):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    poses = [np.eye(4)]

    kp_query, des_query = orb.detectAndCompute(images[0], None)
    for i in range(1, len(images)):
        kp_train, des_train = orb.detectAndCompute(images[i], None)
        matches = bf.match(des_query, des_train)
        matches = sorted(matches, key=lambda x: x.distance)[:50]

        ixs_source = np.array([kp_train[match.trainIdx].pt for match in matches])[:,::-1].astype("int")
        ixs_target = np.array([kp_query[match.queryIdx].pt for match in matches])[:,::-1].astype("int")

        points_source = projections[i,ixs_source[:,0],ixs_source[:,1]]
        points_target = projections[i-1,ixs_target[:,0],ixs_target[:,1]]

        transform = estimate_transform(points_source, points_target)
        transform = np.vstack((transform, np.array([[0, 0, 0, 1]])))
        poses.append(np.matmul(poses[i-1], transform))
        print(poses[i])

        kp_query = kp_train
        des_query = des_train

    return np.array(poses)

if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-i", "--input", dest="input", help="Read data in .npz format from INPUTFILE", metavar="INPUTFILE")

    (options, args) = parser.parse_args()

    data = np.load(options.input)

    np.set_printoptions(suppress=True)

    images = data["images"]
    projections = data["projections"]

    start = time.time()

    poses = estimate_poses_matching(images, projections)

    end = time.time()

    print(end - start)

    ps = poses[:,:3,3]

    plt.plot(ps[:,0])
    plt.plot(ps[:,1])
    plt.plot(ps[:,2])
    plt.show()
