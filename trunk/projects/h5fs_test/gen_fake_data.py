# coding: utf-8
import numpy as np
import scipy.misc
import os.path as osp
import argparse
import urllib2

import coda.sio_lite
import PIL
import PIL.Image
import scipy.misc

from contextlib import contextmanager
import os

imagesize = (64, 64)
img = PIL.Image.fromarray(scipy.misc.face())
imgs = np.asarray(img.resize(imagesize, resample=PIL.Image.LANCZOS))

def make_rand(gray=True):
    ''' Create an image with random noise applied to it '''
    if gray:
        base = imgs[:, :, 0]
        bias = np.random.rand(*imagesize).astype(np.float32)
    else:
        base = imgs
        threechan = imagesize + (3,)
        bias = np.random.rand(*threechan).astype(np.float32)
    return base + bias

def write_sio(x, gray=True):
    coda.sio_lite.write(make_rand(gray), '{:06d}.sio'.format(x))

def write_any(x, ext, gray=True):
    PIL.Image.fromarray(make_rand(gray).astype(np.uint8)).save('{:06d}.{}'.format(x, ext))


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def make_directory(path):
    ''' Make a directory with a bunch of test files '''

    if not osp.exists(path):
        os.makedirs(path)

    with cd(path):

        for x in range(0, 1000):
            write_sio(x)
        write_any(1000, 'jpg', False)
        write_any(1001, 'jpg')
        write_any(1002, 'tif')

        nitffilename = '001003.nitf'
        if not osp.exists(nitffilename):
            nitfurl = 'http://www.gwg.nga.mil/ntb/baseline/software/testfile/Nitfv2_0/U_1001A.NTF'
            nitffile = urllib2.urlopen(nitfurl)

            with open(nitffilename,'wb') as output:
                output.write(nitffile.read())



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str,
                        choices=['unsorted', 'test_train'],
                        help='Directory of input imagery.')
    options = parser.parse_args()

    dataset = options.dataset

    if dataset == 'unsorted':
        make_directory('unsorted')
    elif dataset == 'test_train':
        make_directory('test_train/train/{}'.format(imagesize[0]))
        traindir = 'test_train/test/{}'.format(imagesize[0])
        if osp.exists(traindir):
            os.makedirs(traindir)
    else:
        raise ValueError('Unknown test set type')


