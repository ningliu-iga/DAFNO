import shapely
from math import *
import numpy as np
import shapely.geometry as geom
import matplotlib.pyplot as plt
from scipy import interpolate

interp_level = 5
rr_all = np.load('Random_UnitCell_rr_10_interp.npy')
size = 41

gridx = np.linspace(-0.5, 0.5, size)
gridx = np.repeat(gridx.reshape(size, 1), size, axis=1)
gridy = np.linspace(-0.5, 0.5, size)
gridy = np.repeat(gridy.reshape(1, size), size, axis=0)

dtheta = 2. * pi / 40.
theta = dtheta * np.arange(0, 41)
theta_new = np.concatenate((dtheta / interp_level * np.arange(0, 40 * interp_level), [theta[-1]]))

dist_all = np.zeros([size, size, rr_all.shape[2]])

for j in range(rr_all.shape[2]):
    print(j)
    rr_test = rr_all[:, 0, j]
    f = interpolate.interp1d(theta, rr_test, kind='cubic')
    rr_new = f(theta_new)
    coords = np.zeros([rr_new.shape[0], 2])

    for i in range(rr_new.shape[0]):
        coords[i, 0] = cos(theta_new[i]) * rr_new[i]
        coords[i, 1] = sin(theta_new[i]) * rr_new[i]

    line = geom.LineString(coords)
    point = geom.Point(0, 0)
    for xx in range(size):
        for yy in range(size):
            point = geom.Point(gridx[xx, yy], gridy[xx, yy])
            dist_all[xx, yy, j] = line.distance(point)
myf = open('Random_UnitCell_dist_10_interp.npy', 'wb')
np.save(myf, dist_all)
