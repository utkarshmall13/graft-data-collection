'''
# Sampler for NAIP data

idir is a directory containing metadata about ground images for e.g. its geolocations.

This file takes in such metadata of ground images and finds optimal points to sample **NAIP** satellite images.
'''

from multiprocessing import Pool
import csv
from os import listdir
from os.path import join
import numpy as np
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Polygon, Point
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-idir', '--input-dir', default='../data/metadata_sdb_existing')
parser.add_argument('-hw', '--half-width', default=0.001)


args = parser.parse_args()

idir = args.input_dir
# halfwidth is the distance in lat/long degrees corresponding to 112 pixels of sentinel-2 image of 10 m resolution
# so a 224 pixel square around a point is considered to be the area of interest 
# halfwidth_sentinel = 0.01
# halfwidth of planet image of resolution 5m is 0.005
# halfwidth of naip image of resolution 1m is 0.001
halfwidth = args.half_width
radius = np.sqrt(2)*halfwidth # the area to search for nearest neighbors

files = sorted(listdir(idir))
# for file in files:
def dodat(file):
    data = []
    with open(join(idir, file)) as ifd:
        reader = csv.reader(ifd)
        for row in reader:
            data.append(row)
    return data

# with Pool() as pool:
#     data = pool.map(dodat, files)
data = []
for file in files:
    data.append(dodat(file))

allrows = []
for datum in data:
    for row in datum:
        allrows.append(row)

print('Total unique images:', len(allrows))
ids = [int(tmp[0]) for tmp in allrows]

# get longitudes and latitudes contained in 3rd and 4th column

lon = np.array([row[3] for row in allrows]).astype(float)
lat = np.array([row[4] for row in allrows]).astype(float)
dates = []
for row in tqdm(allrows):
    datetime_object = datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S')
    month = (datetime_object.year-2012)*12+datetime_object.month-1
    dates.append(month)
dates = np.array(dates)

inds = np.array(list([i for i in range(len(lon))]))
bbox = [(np.min(lon), np.min(lat)), (np.max(lon), np.max(lat))]
print("Geographical Bounds: ", bbox)

# Understanding distribution in dataset
# we bin data by latitude and longitude and year+month
bins = {}
for ind, lo, la, date in zip(inds, lon, lat, dates):
    key = int(np.floor(lo))//10, int(np.floor(la))//10, date//3
    if key in bins:
        bins[key].append(ind)
    else:
        bins[key] = [ind]
print('Total bins:', int(len(bins)))

PoIs = []
ImageInds = []

# the main sampling loop
# PoIs is the list of points of interest. 
# We sample points of interest from ground image coordinates. 
# If a point of interest is within a 224*sqrt(2) pixel circle it falls within the area of interest and is taken_care_of
# We do not sample points of interest that are already taken care of
# Imageinds is the list of indices of ground images that fall within the area of interest of a point of interest
total_done = 0
for kind, key in tqdm(enumerate(bins.keys()), total=len(bins)):
    coords = [[lon[bins[key][i]], lat[bins[key][i]]] for i in range(len(bins[key]))]
    total_done+=len(bins[key])
    coords = np.array(coords)
    knns = NearestNeighbors(radius=radius).fit(coords)
    taken_care_of = [False for tmp in range(len(bins[key]))]
    
    for i, ind in enumerate(bins[key]):
        if taken_care_of[i]:
            continue
        taken_care_of[i] = True
        PoI = np.array([lon[ind], lat[ind]])
        # if it falls in the circle
        dists, nninds = knns.radius_neighbors(np.array([PoI]))
        bbox_nninds = []
        for nnind in nninds[0]:
            nnpoint = np.array([lon[bins[key][nnind]], lat[bins[key][nnind]]])
            # additionally check if it falls in the square
            if np.abs(nnpoint[0]-PoI[0])<halfwidth and np.abs(nnpoint[1]-PoI[1])<halfwidth:
                bbox_nninds.append(nnind)

        PoIs.append(PoI)
        for bbox_nnind in bbox_nninds:
            taken_care_of[bbox_nnind] = True
        ImageInds.append([bins[key][nnind] for nnind in bbox_nninds])

Dates = []
for ind in tqdm(range(len(PoIs))):
    date = []
    for i in ImageInds[ind]:
        date.append(datetime.strptime(allrows[i][2], '%Y-%m-%d %H:%M:%S'))
    Dates.append(str(date[len(date)//2]))

ImageIds = [[int(allrows[tmp][0]) for tmp in row] for row in ImageInds]

for i in range(len(ImageInds)):
    assert len(ImageInds[1500])==len(ImageIds[1500])

image_data = {}
for row in allrows:
    image_data[int(row[0])] = [row[0], row[2], row[3], row[4], row[14]]
with open('image_data.pkl', 'wb') as ofd:
    pickle.dump({'image_data': image_data}, ofd)
print("Saving image data")
with open('satellite_centers.pkl', 'wb') as ofd:
    pickle.dump({'PoIs': PoIs, 'ImageIds': ImageIds, 'Dates': Dates}, ofd)
print("Saving satellite centers")