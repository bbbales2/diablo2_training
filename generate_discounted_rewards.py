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
parser.add_argument('--batchsize', type = int, default = 1000, help = 'Size of batches')

args = parser.parse_args()

if args.training is None:
    print "Please specify a training dataset"
    exit(-1)

if args.training is None:
    print "Please specify a reward model"
    exit(-1)

with h5py.File(args.training, 'r') as f:
    Xshape = f['X'].shape
    framesShape = f['frames'].shape

model = keras.models.load_model(args.rewardModel)

def generator(filename, batchsize):
    with h5py.File(filename, 'r') as f:
        N = f['X'].shape[0]

        for i in range(0, N, batchsize):
            ir = min(N, i + batchsize)

            tmp = time.time()
            X = f['X'][i : ir].reshape(-1, 8)
            frames = f['frames'][i : ir, :, :, :]
            rewards = f['rewards'][i : ir].flatten()

            print "Processing {0}/{1}, readtime {2}".format(i, N, time.time() - tmp)
            
            yield (frames, X, rewards.flatten())

rewards = []
max_rewards = []
for frames, X, rewards_ in generator(args.training, args.batchsize):
    rewards.extend(rewards_)#model.predict([frames, X]).flatten())

    max_rewards_ = []
    for i in range(8):
        X = numpy.zeros((frames.shape[0], 8))
        X[:, i] = 1
        max_rewards_.append(model.predict([frames, X]).flatten())

    max_rewards_ = numpy.max(numpy.array(max_rewards_).transpose(), axis = 1)

    max_rewards.extend(max_rewards_)

N = len(rewards)
for i in range(N):
    for j in range(i + 1, min(i + args.maxDiscountSteps, N)):
        rewards[i] += max_rewards[j] * args.discount**(j - i)

with h5py.File(args.outputRewards, "w") as f:
    f.create_dataset('rewards', data = rewards)
