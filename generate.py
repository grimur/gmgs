import numpy


class TestData(object):
    def __init__(self, classes=2, num=100, dims=2, means=None, stds=None):
        self.classes = {}
        if means is not None:
            self.means = means
        else:
            self.means = [numpy.random.random(dims) for x in range(classes)]
        if stds is not None:
            self.stds = stds
        else:
            self.stds = [numpy.random.random(dims)*0.1 for x in range(classes)]

        for i in range(num):
            label = numpy.random.randint(classes)
            label_means = self.means[label]
            label_stds = self.stds[label]
            point = numpy.random.normal(label_means, label_stds)
            if label in self.classes:
                self.classes[label].append(point)
            else:
                self.classes[label] = [point]

    def get_by_class(self):
        keys = self.classes.keys()
        keys.sort()
        return [self.classes[x] for x in keys]

    def get_all(self):
        all_samples = numpy.vstack(self.classes.values())
        numpy.random.shuffle(all_samples)
        return all_samples

