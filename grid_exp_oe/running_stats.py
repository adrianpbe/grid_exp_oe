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


class RunningStats:
    def __init__(self):
        self.mean = 0
        self.m2 = 0
        self.count = 0
        self.var = 1
        self.std = 1

    def batch_update(self, new_value):
        for v in new_value:
            self.update(v)

    def update(self, new_value):
        if isinstance(new_value, np.ndarray):
            new_value = np.squeeze(new_value)
        self.count, self.mean, self.m2 = update(
            (self.count, self.mean, self.m2),
            new_value
        )
        if self.count > 2:
            self.var, _ = finalize((self.count, self.mean, self.m2))

        self.std = np.sqrt(self.var)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return (x *  self.std) + self.mean
