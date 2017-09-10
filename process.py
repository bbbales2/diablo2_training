import keras
import h5py
import numpy

model = keras.models.load_model("cold_plains.keras")

def generator(filename, batchsize):
    with h5py.File(filename, 'r') as f:
        N = f['X'].shape[0]

        for i in range(0, N, batchsize):
            print "{0}/{1}".format(i, N)
            
            ir = min(N, i + batchsize)
            
            X = f['X'][i : ir].reshape(-1, 8)
            frames = f['frames'][i : ir, 120:300, 230:410, :]
            rewards = f['rewards'][i : ir].flatten()
            yield ([frames, X], rewards.flatten())

batchsize = 500
results = []
for v in generator("cold_plains.hdf5", batchsize):
    predicts = []
    for i in range(8):
        X = numpy.zeros((batchsize, 8))
        X[:, i] = 1
        predicts.append(model.predict([v[0][0], X]).flatten())

    predicts = numpy.array(predicts).transpose()

    results.extend(numpy.max(predicts, axis = 1))

with h5py.File("cold_plains_mod.hdf5", "r+") as f:
    data = f['rewards']
    data[...] = results
