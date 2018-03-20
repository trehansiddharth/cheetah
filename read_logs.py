import lcm
import state_estimate_t
import contact_estimate_t
import motive_t
import pose_t
import pointcloud_t
import numpy as np
import pyquaternion
from optparse import OptionParser

def parse_cheetah_state(event):
    state_estimate = state_estimate_t.state_estimate_t.decode(event.data)
    q = pyquaternion.Quaternion(np.array(state_estimate.quat))
    gyro = state_estimate.gyro
    accelerometer = state_estimate.accelerometer
    p = np.array(np.array(state_estimate.p0))
    pose = np.empty((3, 4))
    pose[:,:-1] = q.rotation_matrix
    pose[:,-1] = p
    return { "cheetahTimestamps" : event.timestamp,
        "cheetahPoses" : pose,
        "cheetahGyros" : gyro,
        "cheetahAccelerometers" : accelerometer }

def parse_tango_poses(event):
    msg = pose_t.pose_t.decode(event.data)
    q = pyquaternion.Quaternion(np.array(msg.orientation))
    p = np.array(msg.position)
    pose = np.empty((3, 4))
    pose[:,:-1] = q.rotation_matrix
    pose[:,-1] = p
    return { "tangoTimestamps" : event.timestamp, "tangoPoses" : pose }

def parse_tango_pointclouds(event):
    pointcloud = pointcloud_t.pointcloud_t.decode(event.data)
    return { "tangoPointcloudTimestamps" : event.timestamp,
        "tangoPointclouds" : np.array(pointcloud.points).reshape(-1, 4) }

def read_log(log, channel, f):
    variables = {}

    for event in log:
        if event.channel == channel:
            for name, value in f(event).items():
                if name in variables.keys():
                    variables[name].append(value)
                else:
                    variables[name] = [value]
    return { name : np.array(value) for name, value in variables.items() }

if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-i", "--input", dest="input", help="Read LCM log events from INPUTFILE", metavar="INPUTFILE")
    parser.add_option("-o", "--output", dest="output", help="Output data in .npz format to OUTPUTFILE", metavar="OUTPUTFILE")

    (options, args) = parser.parse_args()

    lcm_log = lcm.EventLog(options.input, "r")

    cheetahStateVariables = read_log(lcm_log, "CHEETAH_state_estimate", parse_cheetah_state)
    tangoVariables = read_log(lcm_log, "TANGO_POSE", parse_tango_poses)
    pointcloudVariables = read_log(lcm_log, "TANGO_POINTCLOUD", parse_tango_pointclouds)

    np.savez_compressed(options.output, **{ **cheetahStateVariables, **tangoVariables, **pointcloudVariables })
