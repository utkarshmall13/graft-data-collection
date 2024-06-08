import requests
import json
import csv
from datetime import date,timedelta
from os.path import join,isdir,isfile
from os import mkdir
from copy import deepcopy
import numpy as np
np.random.seed(seed=42)
import fiona
from shapely.geometry import shape, mapping, Point, Polygon, MultiPolygon
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
# np.random.seed(seed=40)


parser = argparse.ArgumentParser()
parser.add_argument('-odir', '--output-dir', default='../data/metadata')
parser.add_argument('-key', '--key', default='PUT_YOUR_FLICKR_KEY_HERE')
parser.add_argument('-ns', '--num-samples', default=10000)
parser.add_argument('-si', '--start-index', default=0)
parser.add_argument('--start-year', '-sy', default=2013, type=int)

args = parser.parse_args()

KEY = args.key
outdir = args.output_dir
num_samples = args.num_samples
start_index = args.start_index
start_year = str(args.start_year)

if not isdir(outdir):
    mkdir(outdir)

def get_photos(page=1, coords=None):
    params = {
        'content_type': '1',
        'per_page': '250',
        'media': 'photos',
        'method': 'flickr.photos.search',
        'format': 'json',
        'advanced': 1,
        'lat': coords[1],
        'lon': coords[0],        
        'radius': 2,
        'nojsoncallback': 1,
        'extras': 'realname,geo,tags,date_taken,url_o,url_c,url_l,url_m,url_n,url_q,url_s,url_sq,url_t,url_z,url_w,url_b,url_h',
        'page': page,
        'accuracy':15,
        'api_key': KEY,
        'min_taken_date': start_year+'-01-01 00:00:00',
        # 'sort': 'interestingness-desc'

    }

    results = requests.get('https://api.flickr.com/services/rest', params=params).json()['photos']
    return results

keys = ['id','owner','datetaken','longitude','latitude','title','farm','context','secret','place_id','woeid','realname','server','tags']
seqs = ['c', 'z', 'w', 'n', 'b', 'h', 'o', 'm', 'l','q', 's', 'sq', 't']

# this shapefile is only for USA, you need to look for new shapefiles if you want to dowload countries.
shapes = fiona.open("../utils/cb_2018_us_state_5m/cb_2018_us_state_5m.shp")
polys = []
# only downloading for hawaii, use custom shapefile for other regions.
for x in shapes:
    if x['properties']['STUSPS'] in ['HI']:
        polys.append(shape(x['geometry']))

# find bound to sample from
bounds = [360, 360, -360, -360]
for poly in polys:
    bound = poly.bounds
    if bounds[0]>bound[0]:
        bounds[0] = bound[0]
    if bounds[1]>bound[1]:
        bounds[1] = bound[1]
    if bounds[2]<bound[2]:
        bounds[2] = bound[2]
    if bounds[3]<bound[3]:
        bounds[3] = bound[3]
print("Bounds: ", bounds)

lons = np.random.uniform(bounds[0], bounds[2], num_samples)
lats = np.random.uniform(bounds[1], bounds[3], num_samples)
for i in tqdm(range(start_index, len(lons))):
    point = Point(lons[i], lats[i])
    within = False
    for poly in polys:
        if point.within(poly):
            within = True
    if within:
        rows = [['id','owner','datetaken','longitude','latitude','title','farm','context','secret','place_id','woeid','realname','server','tags','url','height','width']]
        page = 1
        try:
            res = get_photos(page=page, coords = (lons[i], lats[i]))
            total_pages = res['pages']
            print('total_pages', total_pages)
            while page<=total_pages and page<5:
                for img in res['photo']:
                    rows.append([])
                    for key in keys:
                        if(key in img):
                            rows[-1].append(img[key])
                        else:
                            rows[-1].append('')
                    not_broken = True
                    for seq in seqs:
                        if('url_'+seq in img):
                            rows[-1].append(img['url_'+seq])
                            rows[-1].append(img['height_'+seq])
                            rows[-1].append(img['width_'+seq])
                            not_broken = False
                            break
                    if(not_broken):
                        data.pop()
                page+=1
                if(page<=total_pages and page<4):
                    res = get_photos(page=page, coords = (lons[i], lats[i]))
            if total_pages>0:
                with open(join(outdir,str(i).zfill(7)+'_'+str(round(lons[i], 5))+'_'+str(round(lats[i], 5))+'.csv'),'w') as ifd:
                    writer = csv.writer(ifd,delimiter=',')
                    for row in rows:
                        writer.writerow(row)        
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(e)
            pass

    # print(point)



