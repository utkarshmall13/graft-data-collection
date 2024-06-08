from os import mkdir, listdir
from os.path import join, isdir, isfile
import urllib.request 
from multiprocessing.dummy import Pool, Lock
import numpy as np
import csv
import sys
from PIL import Image, ImageFile
import random
import argparse
import requests
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--start-ind', '-si', default=0, type=int)
parser.add_argument('--end-ind', '-ei', default=-1, type=int)
parser.add_argument('-idir', '--input-dir', default='../data/metadata_sdb')
parser.add_argument('-odir', '--output-dir', default='../data/images')

args = parser.parse_args()


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

idir = args.input_dir
odir = args.output_dir

if not isdir(odir):
    mkdir(odir)


def init(lo):
	global lock
	lock = lo


files = sorted(listdir(idir))
sind = args.start_ind
eind = args.end_ind

if eind<0:
	files = files[sind:]
else:
	files = files[sind:eind]
random.shuffle(files)
print("total metadata file:", len(files))

for file in tqdm(files):
	with open(join(idir, file), newline='\n') as ifd:
		reader = csv.reader(ifd, delimiter=',')
		urls = []
		ids = []
		try:
			for i, row in enumerate(reader):
				try:
					urls.append(row[14])
					ids.append(row[0])
				except:
					print(row)
					pass
		except:
			print(file)
			raise

	def get_image(url):
		fname = url.split('/')[-1].split('_')[0]+'.'+url.split('.')[-1]
		noextname = url.split('/')[-1].split('_')[0]
		if fname[0:1]>='0' and fname[0:1]<='9':
			if isfile(join(odir, noextname[-4], noextname[-3], noextname[-2], noextname[-1], fname)):
				return

			dir0 = join(odir, noextname[-4])
			lock.acquire()
			direxists = isdir(dir0)
			if not direxists:
				mkdir(dir0)
			lock.release()

			dir1 = join(odir, noextname[-4], noextname[-3])
			lock.acquire()
			direxists = isdir(dir1)
			if not direxists:
				mkdir(dir1)
			lock.release()

			dir2 = join(odir, noextname[-4], noextname[-3], noextname[-2])
			lock.acquire()
			direxists = isdir(dir2)
			if not direxists:
				mkdir(dir2)
			lock.release()

			dir3 = join(odir, noextname[-4], noextname[-3], noextname[-2], noextname[-1])
			lock.acquire()
			direxists = isdir(dir3)
			if not direxists:
				mkdir(dir3)
			lock.release()

			fname = join(dir3, fname)
			try:
				img = Image.open(requests.get(url, stream=True).raw)
				h, w = img.size[0], img.size[1]
				if min(h, w)>256:
					if h>w:
						img = img.resize((int(256*h/w), 256))
					else:
						img = img.resize((256, int(256*w/h)))
				img.save(join(fname))
			except Exception as e:
				print(e)
				print(file, ',', url)
				pass

	l = Lock()
	pool = Pool(4, initializer=init, initargs=(l, ))
	pool.map(get_image, urls)
	pool.close()
	print(file)








