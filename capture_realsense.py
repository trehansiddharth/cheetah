import numpy as np
import cv2
import realsense
from optparse import OptionParser
import time

def preview(img16):
    img8 = (255 * img16.astype("float") / np.max(img16.astype("float"))).astype("uint8")
    cv2.imshow("preview", img8)
    return cv2.waitKey(10)

if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-o", "--output", dest="output", help="Output data in .npz format to OUTPUTFILE", metavar="OUTPUTFILE")

    (options, args) = parser.parse_args()

    cap = realsense.Realsense()

    images = []
    projections = []
    timestamps = []
    while True:
        color, depth, projection = cap.capture()
        #pointcloud = vertices[np.product(vertices, axis=2).astype("bool")].reshape(-1, 3)
        timestamp = np.int64(time.time() * 1000)
        cv2.imshow("preview", color)
        if (cv2.waitKey(1) == ord('a')):
            break
        images.append(color)
        projections.append(projection)
        timestamps.append(timestamp)
    np.savez_compressed(options.output, images=np.array(images), projections=np.array(projections), timestamps=np.array(timestamps))
