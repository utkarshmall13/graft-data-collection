from multiprocessing import Pool
import csv
from os import listdir
from os.path import join
import numpy as np
import argparse
import matplotlib.pyplot as plt
from os import mkdir
from os.path import isdir

parser = argparse.ArgumentParser()
parser.add_argument('-idir', '--input-dir', default='../data/metadata')
parser.add_argument('-odir', '--output-dir', default='../data/metadata_sdb')

args = parser.parse_args()

idir = args.input_dir
odir = args.output_dir
if not isdir(odir):
    mkdir(odir)


files = [join(idir, tmp) for tmp in sorted(listdir(idir))]

def worker(fname):
    try:
        with open(fname) as ifd:
            reader = csv.reader(ifd)
            data = []
            uniusers = {}
            unititle = {}
            unicoords = {}
            for i, row in enumerate(reader):
                if i==0:
                    continue
                # same user either have photo from same location or same photobunch
#                 if row[1] in uniusers and (row[5] in unititle or (float(row[3]), float(row[4])) in unicoords):
#                     continue
                data.append(row)
                uniusers[row[1]] = 0
                unititle[row[5]] = 0
                unititle[(float(row[3]), float(row[4]))] = 0
    except Exception as e:
        print('Cannot read file:', fname)
        raise e
    return data

print("Total metadata file:", len(files))
with Pool() as pool:
    data = pool.map(worker, files)

allrows = []
for datum in data:
    for row in datum:
        allrows.append(row)
print("Total non-unique image:", len(allrows))

# statistical analsys
arr = []
for i in range(1, 11):
    ids = [int(row[0]) for row in allrows[:(len(allrows)*i)//10]]
    a, b, c, d = np.unique(ids, return_index=True, return_inverse=True, return_counts=True)
    # print(len(a)/((len(allrows)*i)//10))
    arr.append([len(a)/((len(allrows)*i)//10), len(a)])

ids = [int(row[0]) for row in allrows]
a, b, c, d = np.unique(ids, return_index=True, return_inverse=True, return_counts=True)
print("Total unique image:", len(a))

plt.subplot(2, 1, 1)
plt.plot([tmp/10+0.1 for tmp in range(10)], [tmp[0] for tmp in arr])
plt.title("fraction of images that are unique in first x fraction")
plt.subplot(2, 1, 2)
plt.title("number of images that are unique in first x fraction")
plt.plot([tmp/10+0.1 for tmp in range(10)], [tmp[1] for tmp in arr])
plt.plot([tmp/10+0.1 for tmp in range(0, 10, 9)], [tmp[1] for tmp in arr[::9]], '--')
plt.savefig("stats.png")
plt.close()
print("As long as the second curve is linear, not plateauing more images are needed")

allrows = [allrows[ind] for ind in b]

# sanity check
ids = [int(row[0]) for row in allrows]
a, b, c, d = np.unique(ids, return_index=True, return_inverse=True, return_counts=True)
assert len(a)==len(allrows)

lon = np.array([row[3] for row in allrows]).astype(float)
lat = np.array([row[4] for row in allrows]).astype(float)

plt.figure(figsize=(25, 15))
# plt.scatter(lon, lat, s=0.2, alpha=0.1)
plt.scatter(lon, lat, s=5)
plt.title('scatter plot of ground image location')
plt.savefig("scatter_plot.png")
plt.close()

plt.figure(figsize=(25, 15))
plt.hexbin(lon, lat, gridsize=200, bins='log', cmap='Blues')
plt.title('histogram of ground images')
plt.savefig("histogram.png")
plt.close()

np.save('samples.npy', np.array([lon, lat]))

lon = np.array([row[3] for row in allrows]).astype(float)
inds = np.argsort(lon)
allrows = [allrows[tmp] for tmp in inds]

imgsperbin = 10000
totbins = ((len(allrows)-1)//imgsperbin)+1
print("Total new metadata files:", totbins)
for i in range(totbins):
    with open(join(odir, str(i).zfill(4)+'.csv'), 'w') as ofd:
        writer = csv.writer(ofd)
        for j in range(imgsperbin):
            ind = i*imgsperbin+j
            if ind < len(allrows):
                writer.writerow(allrows[ind])