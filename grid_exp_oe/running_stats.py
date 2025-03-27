import numpy as np


# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
def update(existing_aggregate, new_value):
    (count, mean, M2) = existing_aggregate
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return (count, mean, M2)

# Retrieve the mean, variance and sample variance from an aggregate
def finalize(existing_aggregate):
    (count, mean, M2) = existing_aggregate
    if count < 2:
        raise ValueError("not enough data points, more than 2 are required")
    else:
        (mean, variance, sample_variance) = (mean, M2 / count, M2 / (count - 1))
        return (variance, sample_variance)


# Batch update for mean and variance
def batch_update(existing_aggregate, new_values):
    (count, mean, M2) = existing_aggregate
    n = len(new_values)
    new_count = count + n
    new_mean = mean + (np.sum(new_values - mean) / new_count)
    delta = new_values - mean
    delta2 = new_values - new_mean
    new_M2 = M2 + np.sum(delta * delta2)
    return (new_count, new_mean, new_M2)


# Finalize for batch data
def batch_finalize(existing_aggregate):
    (count, mean, M2) = existing_aggregate
    if count < 2:
        raise ValueError("not enough data points, more than 2 are required")
    else:
        variance = M2 / count
        sample_variance = M2 / (count - 1)
        return (variance, sample_variance)


class RunningStats:
    def __init__(self, shape):
        self.shape = shape
        self.mean = np.zeros(shape, dtype=np.float32)
        self.m2 = np.zeros(shape, dtype=np.float32)
        self.count = 0
        self.var = np.ones(shape, dtype=np.float32)
        self.std = np.ones(shape, dtype=np.float32)

    def _values_as_arrays(self):
        self.mean = np.array(self.mean)
        self.std = np.array(self.std)
        self.var = np.array(self.var)

    def batch_update(self, new_values):
        self.count, self.mean, self.m2 = batch_update(
            (self.count, self.mean, self.m2),
            new_values
        )
        if self.count > 2:
            self.var, _ = batch_finalize((self.count, self.mean, self.m2))

        self.std = np.sqrt(self.var)

        self._values_as_arrays()

    def update(self, new_value):
        self.count, self.mean, self.m2 = update(
            (self.count, self.mean, self.m2),
            new_value
        )
        if self.count > 2:
            self.var, _ = finalize((self.count, self.mean, self.m2))

        self.std = np.sqrt(self.var)
        self._values_as_arrays()

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return (x *  self.std) + self.mean
