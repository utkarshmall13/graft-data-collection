import csv
from multiprocessing import Pool
import argparse
from os.path import isdir
from os import listdir, mkdir
from os.path import join, isfile, isdir
from multiprocessing import Pool
from numpy import random
import numpy as np
import csv
import pickle
from tqdm import tqdm

# directory setup
parser = argparse.ArgumentParser()
parser.add_argument('-satdir', '--satellite-dir', default='../data/naipimages')
parser.add_argument('-imgdir', '--images-dir', default='../data/images')
parser.add_argument('-metadir', '--metadata-dir', default='../data/metadata_sdb_existing')
parser.add_argument('-splitdir', '--split-dir', default='../data/split_m2o')
parser.add_argument('-f', '--test-fraction', default=0.1, type=float)

args = parser.parse_args()
    
imgdir = args.images_dir
splitdir = args.split_dir
metadir = args.metadata_dir
satdir = args.satellite_dir
# should be very small 1 to 5 percent of the dataset.
test_fraction = args.test_fraction

if not isdir(splitdir):
    mkdir(splitdir)

# utility functions
#############################################################################
def listadir(dir, level):
    assert level>=1
    if level==1:
        subdirs = [join(dir, tmp) for tmp in sorted(listdir(dir)) if isdir(join(dir, tmp))]
        return subdirs
    else:
        subdirs = [join(dir, tmp) for tmp in sorted(listdir(dir)) if isdir(join(dir, tmp))]
        ret = [listadir(subdir, level-1) for subdir in subdirs]
        ret = [item for items in ret for item in items]
        return ret

def listfiles(dir):
    return [join(dir, tmp) for tmp in sorted(listdir(dir))]

def blocks(total, positive, length):
    totalblk = total//length+1
    posblk = positive//length+1
    atevery = totalblk//posblk
    smtotal = [1 if tmp%atevery==0 else 0 for tmp in range(totalblk) ]
    blocktotal = [[i for tmp in range(length)] for i in smtotal]
    blocktotal = [tmp2 for tmp in blocktotal for tmp2 in tmp]
    blocktotal = blocktotal[:total]
    counter = 0
    for i in range(len(blocktotal)):
        if blocktotal[i]>0:
            counter+=1
            if counter>positive:
                blocktotal[i] = 0
    return blocktotal
#############################################################################

# get pair information from metadata
iofiles = sorted(listdir(metadir))
def worker(file):
    with open(join(metadir, file)) as ifd:
        ids = []
        reader = csv.reader(ifd, delimiter=',')
        for row in reader:
            ids.append(int(row[0]))
    return ids
with Pool() as pool:
    data = pool.map(worker, iofiles)
data = [tmp2 for tmp in data for tmp2 in tmp]
data = set(data)
print("Total Flickr images:", len(data))

satintdirs = listadir(satdir, 1)
with Pool(8) as pool:
    allsatfiles = pool.map(listfiles, satintdirs)
allsatfiles = [item for items in allsatfiles for item in items]
print("Total satellite images:", len(allsatfiles))

# Create splits on the basis of satellite images
#############################################################################
with open('satellite_centers.pkl', 'rb') as ifd:
    satcenters = pickle.load(ifd)

traindata = []
testdata = []
# specify the number of test samples
ttestsplit = blocks(len(allsatfiles), int(len(allsatfiles)*test_fraction), 1000)

for sind, satfile in enumerate(allsatfiles):
    if ttestsplit[sind]==0:
        fulldata = traindata
    elif ttestsplit[sind]==1:
        fulldata = testdata
    satid = int(satfile.split('.')[-2].split('/')[-1])
    gimageids = satcenters['ImageIds'][satid]
    validgimids = []
    for gimageid in gimageids:
        if gimageid in data:
            validgimids.append(gimageid)
    if len(validgimids)>25:
        # if more than 25 images are present, then select 25.
        validgimids = validgimids[:25]
    for validgimid in validgimids:
        validgimid = str(validgimid)
        fulldata.append(['/'.join(satfile.split('/')[-2:]), join(validgimid[-4], validgimid[-3], validgimid[-2], validgimid[-1], validgimid+'.jpg')])
    if sind%10000==9999:
        print(sind, len(fulldata))
#############################################################################
# sanity checks
print("Sanity checking splits")
print("# of Unique train satellite images:", len(set([tmp[0] for tmp in traindata])))
print("# of Unique test satellite images:", len(set([tmp[0] for tmp in testdata])))
print("# of Unique train flickr images:", len(traindata))
print("# of Unique test flickr images:", len(testdata))

sanity_alldata = [tuple(tmp) for tmp in traindata]
for tmp in testdata:
    sanity_alldata.append(tuple(tmp))
print("Only unique pairs are present in the dataset:", len(sanity_alldata) == len(set(sanity_alldata)))
assert len(sanity_alldata) == len(set(sanity_alldata))

traingim = [int(row[1].split('.')[0].split('/')[-1]) for row in traindata]
print("Train flickr images:", len(traingim))
traingim = set(traingim)
print("Unique Train flickr images:", len(traingim))

overlapind = [False for i in range(len(testdata))]
for i,row in enumerate(testdata):
    if int(row[1].split('.')[0].split('/')[-1]) in traingim:
        overlapind[i] = True
print()
print("Test Flickr images that are also present in train:", sum(overlapind), "out of", len(testdata))

print("Remove the overlapping test images from the test set and add them to the train set")
traindata.extend([testdata[i][:] for i in range(len(overlapind)) if overlapind[i]])
testdata = [testdata[i][:] for i in range(len(overlapind)) if not overlapind[i]]

print("\nSanity re-checking splits")
print("# of Unique train satellite images:", len(set([tmp[0] for tmp in traindata])))
print("# of Unique test satellite images:", len(set([tmp[0] for tmp in testdata])))
print("# of Unique train flickr images:", len(traindata))
print("# of Unique test flickr images:", len(testdata))

# sanity check last print should be zero
traingim = set([int(row[1].split('.')[0].split('/')[-1]) for row in traindata])
overlapind = [False for i in range(len(testdata))]
for i,row in enumerate(testdata):
    if int(row[1].split('.')[0].split('/')[-1]) in traingim:
        overlapind[i] = True
print("Test Flickr images that are also present in train:", sum(overlapind), "out of", len(testdata))

# do the same for the satellite images
overlapsind = set([tmp[0] for tmp in traindata]).intersection(set([tmp[0] for tmp in testdata]))
overlap = []
for i in range(len(traindata)):
    if traindata[i][0] in overlapsind:
        overlap.append(True)
    else:
        overlap.append(False)
traindata = [traindata[i][:] for i in range(len(traindata)) if not overlap[i]]

# intersection should be zero
inter = len(set([tmp[0] for tmp in traindata]).intersection(set([tmp[0] for tmp in testdata])))
print("intersection of train and test satellite images:", inter)
assert inter==0
# intersection should be zero
inter = len(set([tmp[1] for tmp in traindata]).intersection(set([tmp[1] for tmp in testdata])))
print("intersection of train and test flickr images:", inter)
assert inter==0

print("saving the splits")
with open(join(splitdir, 'train.csv'), 'w') as ofd:
    writer = csv.writer(ofd)
    for row in traindata:
        writer.writerow(row)
with open(join(splitdir, 'test.csv'), 'w') as ofd:
    writer = csv.writer(ofd)
    for row in testdata:
        writer.writerow(row)
        
# create pixel offset files useful for segmentation model training
print("creating pixel offset files for segmentation model training")
with open('image_data.pkl', 'rb') as ifd:
    image_data = pickle.load(ifd)

alldata = []
for tdata in [traindata, testdata]:
    for i, row in tqdm(enumerate(tdata), total=len(tdata)):
        satcenter = np.array(satcenters['PoIs'][int(row[0].split('.')[0].split('/')[-1])])
        imgcenter = np.array([float(tmp) for tmp in image_data['image_data'][int(row[1].split('.')[0].split('/')[-1])][2:4]])
        coordoff = imgcenter-satcenter
        offset = 112*(coordoff/0.01)
        offset[1] = -offset[1]
        alldata.append(row+offset.tolist())

print(len(alldata))
with open(join(splitdir, 'pixel_offset.csv'), 'w') as ofd:
    writer = csv.writer(ofd)
    for row in alldata:
        writer.writerow(row)

print("visualization of the pixel offset")

from PIL import Image, ImageOps
import matplotlib.pyplot as plt

index = random.choice(len(alldata))
satname = alldata[index][0]
imgnames = [row[1] for row in alldata if row[0]==satname]
offsets = [row[-2:] for row in alldata if row[0]==satname]

plt.figure(figsize=(20, 20))
sat = Image.open(join(satdir, satname)).convert('RGB')
plt.imshow(sat, extent=(0, sat.size[1], 0, sat.size[0]))
for imgname, offset in zip(imgnames, offsets):
    img = Image.open(join(imgdir, imgname)).convert('RGB')
    img = ImageOps.expand(img, border=10, fill='white')
    satcenter = np.array(satcenters['PoIs'][int(satname.split('.')[0].split('/')[-1])])
    satdate = np.array(satcenters['Dates'][int(satname.split('.')[0].split('/')[-1])])
    imgcenter = np.array([float(tmp) for tmp in image_data['image_data'][int(imgname.split('.')[0].split('/')[-1])][2:4]])
    location = np.array(offset)+np.array(sat.size[:2])//2
    location[1] = sat.size[1]-location[1]
    plt.imshow(img, extent=(location[0]-25, location[0]+25, location[1]-25, location[1]+25), origin='upper')
plt.xlim(0, sat.size[1])
plt.ylim(0, sat.size[0])
plt.axis('Off')
plt.savefig('offset_visualization.png')
plt.close()
