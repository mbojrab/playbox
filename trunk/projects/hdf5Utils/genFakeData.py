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


def make_directory(path, nsios=1000):
    ''' Make a directory with a bunch of test files '''

    if not osp.exists(path):
        os.makedirs(path)

    with cd(path):

        for x in range(0, nsios):
            write_sio(x)
        write_any(nsios, 'jpg', False)
        write_any(nsios+1, 'jpg')
        write_any(nsios+2, 'tif')

        nitffilename = '{:06d}.nitf'.format(nsios+3)
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
    parser.add_argument('--count', type=int,
                        default=1000,
                        help='Number of sios to create.')
    parser.add_argument('--holdout', type=float,
                        default=.05,
                        help='holdout percentage for test data.')
    options = parser.parse_args()

    dataset = options.dataset
    count = options.count
    holdout = options.holdout

    if dataset == 'unsorted':
        make_directory('unsorted', count)
    elif dataset == 'test_train':
        make_directory('test_train/train/{}'.format(imagesize[0]), count)
        make_directory('test_train/test/{}'.format(imagesize[0]),
                       int(count * holdout))
    else:
        raise ValueError('Unknown test set type')


