import os
import argparse
import numpy
import h5py
import skimage.io
import glob
import json

parser = argparse.ArgumentParser(description='Read in logfile output and produce hdf5 file')

parser.add_argument('logFolder', type = str, help = 'Folder with all the log files')
parser.add_argument('output', type = str, help = 'HDF5 file to save ready to train data in')

args = parser.parse_args()

if not os.path.exists(args.logFolder):
    print "Log folder does not exist. Try again"
    exit(-1)

logs = glob.glob("{0}/*.log".format(args.logFolder))
xs = []
rewards = []
frames = []
for i, filename in enumerate(logs):
    print "Parsing {0} {1}/{2}".format(filename, i, len(logs))
    with open(filename, "r") as f:
        xs_ = []
        rewards_ = []
        frames_ = []
        for line in f:
            if line.strip() == '':
                continue

            time, state, action = json.loads(line)

            if state is None or (state['x'] == 0 and state['y'] == 0):
                continue

            x = numpy.zeros((1, 8))
            x[0, action] = 1
            reward = state["lastUnitClicked"] * 5.0

            xs_.append(x)
            rewards_.append(reward)
            frames_.append(os.path.join(args.logFolder, os.path.basename(state["screen"])))
        rewards.extend(rewards_[1:])
        xs.extend(xs_[:-1])
        frames.extend(frames_[:-1])

rewards = numpy.array(rewards)
xs = numpy.array(xs)

with h5py.File(args.output, "w") as f:
    f.create_dataset("X", data = xs)
    f.create_dataset("rewards", data = rewards)
    fframes = f.create_dataset("frames", shape = (len(frames), 400, 640, 3), dtype = 'uint8')
    for i, filename in enumerate(frames):
        print "Copying image {0}/{1} into dataset".format(i, len(frames))
        fframes[i] = skimage.io.imread(filename)
