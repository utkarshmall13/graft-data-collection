import argparse
import csv
import json
from multiprocessing.dummy import Pool, Lock
import os
from os.path import join, isdir, isfile
from os import mkdir, listdir
from collections import OrderedDict
import time
from datetime import datetime, timedelta
import warnings
warnings.simplefilter('ignore', UserWarning)
import pickle

import ee
import numpy as np
import urllib3
# from skimage.exposure import rescale_intensity
# from torchvision.datasets.utils import download_and_extract_archive
# import shapefile
from shapely.geometry import shape, Point
import urllib.request
import random

class GeoSampler:

    def sample_point(self):
        raise NotImplementedError()

class NAIPSampler(GeoSampler):
    def __init__(self, rows):
        self.rows = rows
        self.fnames = [tmp[0] for tmp in rows]

    def sample_point(self, idx):
        row = [self.fnames[idx], (float(self.rows[idx][1]), float(self.rows[idx][2])), '2008-01-01', '2023-02-14']
        return row

    def __iter__(self):
        return iter(self.fnames)

    def __len__(self):
        return len(self.fnames)

    @staticmethod
    def km2deg(kms, radius=6371):
        return kms / (2.0 * radius * np.pi / 360.0)

def get_collection():
    collection = ee.ImageCollection('USDA/NAIP/DOQQ')
    return collection


def filter_collection(collection, coords, period=None, halfwidth=0.005):
    filtered = collection
    if period is not None:
        filtered = filtered.filterDate(*period)  # filter time
    filtered = filtered.filterBounds(ee.Geometry.Point([coords[0]-halfwidth, coords[1]-halfwidth]))  # filter region
    filtered = filtered.filterBounds(ee.Geometry.Point([coords[0]-halfwidth, coords[1]+halfwidth]))  # filter region
    filtered = filtered.filterBounds(ee.Geometry.Point([coords[0]+halfwidth, coords[1]-halfwidth]))  # filter region
    filtered = filtered.filterBounds(ee.Geometry.Point([coords[0]+halfwidth, coords[1]+halfwidth]))  # filter region
    if filtered.size().getInfo() == 0:
        raise ee.EEException(
            f'ImageCollection.filter: No suitable images found in ({coords[1]:.4f}, {coords[0]:.4f}) between {period[0]} and {period[1]}.')
    return filtered


def get_properties(image):
    properties = {}
    for property in image.propertyNames().getInfo():
        properties[property] = image.get(property)
    return ee.Dictionary(properties).getInfo()


def get_patch(collection, coords, bands=None, scale=None, save_path=None, fname=None):
    if isfile(join(save_path, fname.split('/')[-1])):
        return None
    if bands is None:
        bands = RGB_BANDS

    collection = collection.sort('system:time_start', False)

    # thanks to shitty ee api
    collection = collection.toList(collection.size())


    region = ee.Geometry.Rectangle([[coords[0]-halfwidth, coords[1]-halfwidth], [coords[0]+halfwidth, coords[1]+halfwidth]])   
    for ind in range(collection.size().getInfo()):
        timestamp = collection.get(ind).getInfo()['properties']['system:index']
        patch = ee.Image(collection.get(ind)).select(*bands)
        # url = patch.getDownloadURL({'bands': bands,'scale': 1, 'format': 'GEO_TIFF', 'crs':'EPSG:4326', 'region': region, 'min': 0, 'max': 255, 'gamma': 1.0})
        # urllib.request.urlretrieve(url, join(save_path, fname.split('/')[-1].split('.')[0]+'.tiff'))
        url = patch.getThumbURL({'bands': bands,'scale': 1, 'format': 'jpg', 'crs':'EPSG:4326', 'region': region, 'min': 0, 'max': 255})
        urllib.request.urlretrieve(url, join(save_path, fname.split('/')[-1]))
        # get only the first one
        break
    return None


def date2str(date):
    return date.strftime('%Y-%m-%d')


def get_period(date, days=10):
    date1 = date[0] - timedelta(days=days)
    date2 = date[1] + timedelta(days=days)
    return date1, date2


def get_patches(collection, coords, startdate, enddate, debug=False, halfwidth=0.005, **kwargs):
    period = (startdate, enddate)
    # print(period)
    try:
        filtered_collection = filter_collection(collection, coords, period, halfwidth=halfwidth)
        patches = get_patch(filtered_collection, coords, **kwargs)
    except Exception as e:
        if debug:
            print(e)
        # raise
        return None
    return patches


class Counter:
    def __init__(self, start=0):
        self.value = start
        self.lock = Lock()

    def update(self, delta=1):
        with self.lock:
            self.value += delta
            return self.value


if __name__ == '__main__':
    b4 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--which', type=str, default="NAIP", choices=['NAIP', 'Sentinel-2', 'Sentinel-2-Temporal'])
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--cloud_pct', type=int, default=10)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--indices_file', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_path', default='../data/naipimages')
    args = parser.parse_args()

    ee.Initialize()
    collection = get_collection()

    scale = {'B1': 60, 'B2': 10, 'B3': 10, 'B4': 10, 'B5': 20, 'B6': 20, 'B7': 20, 'B8': 10, 'B8A': 20, 'B9': 60, 'B11': 20, 'B12': 20}
    RGB_BANDS = ['B4', 'B3', 'B2']
    if args.which=="NAIP":
        Sampler = NAIPSampler
        save_path = args.save_path
        scale = {'R': 1, 'G': 1, 'B': 1}
        RGB_BANDS = ['R', 'G', 'B']
        halfwidth = 0.0021

    if not isdir(save_path):
        mkdir(save_path)

    start_time = time.time()
    counter = Counter()
    print(time.time()-b4)

    with open('satellite_centers.pkl', 'rb') as ifd:
        data = pickle.load(ifd)
        PoIs = data['PoIs']
        ImageIds = data['ImageIds']

    PoIs = [[str(ind).zfill(8)+'.jpg', poi[0], poi[1]] for ind, poi in enumerate(PoIs)]


    binsize = 5000
    PoIss = [PoIs[i*binsize: min(i*binsize+binsize, len(PoIs))] for i in range((len(PoIs)-1)//binsize+1)]

    inds = list(range(len(PoIss)))
    random.shuffle(inds)

    for ind in inds:
        print(ind)
        PoIs = PoIss[ind]
        inter_dir = str(ind).zfill(4)
        if not isdir(join(save_path, inter_dir)):
            mkdir(join(save_path, inter_dir))
        rows = []
        print("Total Images:", len(PoIs))
        rows = [tmp for tmp in PoIs if not isfile(join(save_path, inter_dir, tmp[0]))]
        print("Images to be downloaded:", len(rows))
        random.shuffle(rows)

        sampler = Sampler(rows)
 
        def worker(idx):
            pts = sampler.sample_point(idx)
            patches = get_patches(collection, (pts[1]), pts[2], pts[3], bands=RGB_BANDS, scale=scale, debug=args.debug, save_path=join(save_path, inter_dir), fname=pts[0], halfwidth=halfwidth)
            return
        print(inter_dir, len(sampler))
        indices = range(len(sampler))

        if args.num_workers == 0:
            for i in indices:
                worker(i)
                break
        else:
            with Pool(args.num_workers) as p:
                p.map(worker, indices)
