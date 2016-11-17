# coding: utf-8
import os.path as osp
import dataset.ingest.labeled
h5 = dataset.ingest.labeled.Hdf5FileData('./unsorted.hdf5')
filedir = list(h5.walk())[-1] # directory with everything in it
nitname = osp.join(filedir[0], filedir[2][-1])
tifname = osp.join(filedir[0], filedir[2][-2])
jpgname = osp.join(filedir[0], filedir[2][-3])
colorjpgname = osp.join(filedir[0], filedir[2][-4])
sioname = osp.join(filedir[0], filedir[2][-20])

nitf = h5.readImage(nitname)
jpeg = h5.readImage(jpgname)
colorjpeg = h5.readImage(colorjpgname)
tiff = h5.readImage(tifname)
sio = h5.readImage(sioname)


