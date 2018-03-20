import run_realsense.run_realsense as run_realsense
import numpy as np

class Realsense:
    def __init__(self):
        self._realsense = run_realsense.RealsenseReader()

    def capture(self):
        obj = self._realsense.get_frames()
        color = np.asarray(obj.color)
        depth = np.asarray(obj.depth)
        vertices = np.asarray(obj.vertices).reshape(480, 640, 3)
        return color, depth, vertices
