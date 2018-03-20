#import the necessary modules
import freenect
import cv2
import numpy as np
import scipy.signal
from optparse import OptionParser

calibration_matrix = np.array([[526.37013657, 0.00000000, 313.68782938],
    [0.00000000, 526.37013657, 259.01834898],
    [0.00000000, 0.00000000, 1.00000000]])

pixel_size = 2800

pointclouds = []

def point_cloud(depth):
    fx = calibration_matrix[1,1]
    fy = calibration_matrix[0,0]
    cx = calibration_matrix[1,2]
    cy = calibration_matrix[0,2]
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = depth > 0
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (c - cx) / fx, 0)
    y = np.where(valid, z * (r - cy) / fy, 0)
    return np.dstack((x, y, z))[valid]

working = False
n = 5
i = 0
depths = np.empty((n, 480, 640))
pointclouds = []
timestamps = []

def depth_callback(dev, depth, timestamp):
    print(".")
    global i, working
    if not working:
        working = True
        depths[i] = depth
        filtered_depth = np.median(depths, axis=0)
        pointclouds.append(point_cloud(filtered_depth))
        timestamps.append(timestamp)
        i += 1
        i = i % n
        working = False

if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-o", "--output", dest="output", help="Output data in .npz format to OUTPUTFILE", metavar="OUTPUTFILE")

    (options, args) = parser.parse_args()

    context = freenect.init()
    device = freenect.open_device(context, 0)
    freenect.set_depth_mode(device, freenect.RESOLUTION_MEDIUM, freenect.DEPTH_MM)

    try:
        print("Capturing...")
        freenect.runloop(dev=device, depth=depth_callback)
    except KeyboardInterrupt:
        if options.output:
            print("Dumping...")
            np.savez_compressed(options.output, tangoPointclouds=pointclouds, tangoPointcloudTimestamps=timestamps)
