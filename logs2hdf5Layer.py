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
parser.add_argument('-N', type = int, default = -1, help = 'Number of samples to copy. If -1, use all')

args = parser.parse_args()

if not os.path.exists(args.logFolder):
    print "Log folder does not exist. Try again"
    exit(-1)

logs = glob.glob("{0}/*.log".format(args.logFolder))
clickHistories = []
missingHp = []
clicks = []
rewards = []
layers = []
for i, filename in enumerate(logs):
    print "Parsing {0} {1}/{2}".format(filename, i, len(logs))
    clickHistory = numpy.zeros((3, 8))
    with open(filename, "r") as f:
        clickHistories_ = []
        missingHp_ = []
        clicks_ = []
        rewards_ = []
        layers_ = []
        lastHp = None
        for line in f:
            if line.strip() == '':
                continue

            try:
                time, state, action = json.loads(line)
            except Exception as e:
                print "Error with: ", line
                continue

            if state is None or (state['x'] == 0 and state['y'] == 0):
                continue

            click, angleI = action

            clickHistory = numpy.roll(clickHistory, 1)
            clickHistory[:, 0] = 0
            clickHistory[{1 : 0, 2 : 1, 49 : 2}[click], 0] = 1
            
            x = [0] * 8
            try:
                x[angleI] = 1
            except:
                print "Error with: ", line
                continue

            hp = state['hp'] / 256
            xp = state['xp']
            if not lastHp:
                lastHp = hp
                lastXp = xp
                maxHp = hp

            reward = state["lastUnitClicked"] * (state["lastUnitTypeClicked"] == 1) * 5.0
            #reward = min(30.0, 0.0 * max(0.0, hp - lastHp) + (xp - lastXp) / 10.0)
            #if reward > 0.0:
            #    print (hp - lastHp) * 4.0, (xp - lastXp) / 10.0
            #if (xp - lastXp) > 0:
            #    print state["lastUnitClicked"], xp - lastXp
            
            lastHp = hp
            lastXp = xp

            clickHistories_.append(clickHistory)
            missingHp_.append(maxHp - hp)
            clicks_.append(numpy.array(x))
            rewards_.append(reward)
            layers_.append(numpy.array(state["layer"]))
            
        rewards.extend(rewards_[1:])
        clickHistories.extend(clickHistories_[:-1])
        missingHp.extend(missingHp_[:-1])
        clicks.extend(clicks_[:-1])
        layers.extend(layers_[:-1])

rewards = numpy.array(rewards)
clickHistories = numpy.array(clickHistories)
missingHp = numpy.array(missingHp)
clicks = numpy.array(clicks)
layers = numpy.array(layers)

print "{0} out of {1} clicks hit".format(sum(rewards > 0), len(rewards))

if args.N != -1:
    rewards = rewards[:args.N]
    clickHistories = clickHistories[:args.N]
    missingHp = missingHp[:args.N]
    clicks = clicks[:args.N]
    layers = layers[:args.N]

#N = len(rewards)
#for i in range(N):
#    for j in range(i + 1, min(i + 15, N)):
#        rewards[i] += rewards[j] * 0.75**(j - i)

with h5py.File(args.output, "w") as f:
    f.create_dataset("clickHistories", data = clickHistories)
    f.create_dataset("clicks", data = clicks)
    f.create_dataset("missingHp", data = missingHp)
    f.create_dataset("layers", data = layers)
    f.create_dataset("rewards", data = rewards)
