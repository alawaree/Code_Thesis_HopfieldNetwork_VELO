import math


class Clustering:
    def __init__(self, K=4):
        self.bins_count = K
        self.bins = [[] for i in range(self.bins_count)]

    def get_bins(self, event):
        for hit in event.hits:
            hit_r, hit_phi = get_polar_coordinates(hit.x, hit.y)
            for b, bin in enumerate(self.bins):
                if b * (2 * math.pi / self.bins_count) < abs(hit_phi) <= (b+1) * (2 * math.pi / self.bins_count):
                    bin.append(hit)
        return self.bins


def get_polar_coordinates(x, y):
    r = math.sqrt(x ** 2 + y ** 2)
    phi = math.atan2(x, y)
    if phi < 0:
        phi = math.pi - phi
    return r, phi
