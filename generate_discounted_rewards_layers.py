import os
import argparse
import numpy
import h5py
import keras
import time

parser = argparse.ArgumentParser(description='Train neural network for diablo2')
parser.add_argument('training', type = str, default = None, help = 'hdf5 file with training data')
parser.add_argument('rewardModel', type = str, default = None, help = 'Model that knows rewards')
parser.add_argument('outputRewards', type = str, default = None, help = 'Where rewards will be saved')
parser.add_argument('--discount', type = float, default = 0.5, help = 'Discounted reward given at each previous action')
parser.add_argument('--maxDiscountSteps', type = int, default = 10, help = 'Discount only goes back this far')

args = parser.parse_args()

if args.training is None:
    print "Please specify a training dataset"
    exit(-1)

if args.training is None:
    print "Please specify a reward model"
    exit(-1)

with h5py.File(args.training, 'r') as f:
    clickHistories = f['clickHistories'][:]
    clicks = f['clicks'][:]
    missingHp = f['missingHp'][:]
    layers = f['layers'][:]
    rewards = f['rewards'][:]

model = keras.models.load_model(args.rewardModel)

max_rewards = []

for i in range(8):
    clicks = numpy.zeros((clickHistories.shape[0], 8))
    clicks[:, i] = 1
    max_rewards.append(model.predict([layers, clickHistories, clicks, missingHp]).flatten())

max_rewards = numpy.max(max_rewards, axis = 0)

N = len(rewards)
for i in range(N):
    for j in range(i + 1, min(i + args.maxDiscountSteps, N)):
        rewards[i] += max_rewards[j] * args.discount**(j - i)

with h5py.File(args.outputRewards, "w") as f:
    f.create_dataset('rewards', data = rewards)
