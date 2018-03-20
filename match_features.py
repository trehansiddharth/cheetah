import numpy as np
import cv2
from matplotlib import pyplot as plt
from optparse import OptionParser

if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-i", "--input", dest="input", help="Input data in .npz format from INPUTFILE", metavar="INPUTFILE")

    (options, args) = parser.parse_args()

    data = np.load(options.input)

    img1 = data["images"][30] # query
    img2 = data["images"][40] # train

    projection1 = data["projections"][30]
    projection2 = data["projections"][40]

    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=2)

    plt.imshow(img3),plt.show()
