from os import listdir
from os.path import join, isfile
import csv
from multiprocessing import Pool
import argparse
from os import mkdir
from os.path import isdir


parser = argparse.ArgumentParser()
parser.add_argument('-idir', '--input-dir', default='../data/metadata_sdb')
parser.add_argument('-odir', '--output-dir', default='../data/metadata_sdb_existing')
parser.add_argument('-imdir', '--images-dir', default='../data/images')

args = parser.parse_args()

idir = args.input_dir
odir = args.output_dir
imgdir = args.images_dir
if not isdir(odir):
    mkdir(odir)


files = sorted(listdir(idir))
for file in files:
	with open(join(idir, file)) as ifd:
		reader = csv.reader(ifd, delimiter=',')

		def worker(row):
			url = row[-3]
			fname = url.split('/')[-1].split('_')[0]+'.'+url.split('.')[-1]
			noextname = url.split('/')[-1].split('_')[0]
			exfname = join(imgdir, noextname[-4], noextname[-3], noextname[-2], noextname[-1], fname)
			exfname_noroot = join(noextname[-4], noextname[-3], noextname[-2], noextname[-1], fname)
			if isfile(exfname) and '.jpg' in fname:
				return row+[exfname_noroot]
		idata = []
		for row in reader:
			idata.append(row)
		pool = Pool(4)
		data = pool.map(worker, idata)
		data = [tmp for tmp in data if tmp is not None]

	with open(join(odir, file), 'w') as ofd:
		writer = csv.writer(ofd, delimiter=',')
		for datum in data:
			writer.writerow(datum)
	print(file)
