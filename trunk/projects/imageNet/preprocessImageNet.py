import argparse
import os
from PIL import Image

def createReMap(filepath) :
    hash = {}
    with open(filepath, 'r') as f :
        for line in f.readlines() :
            lineSplit = line.split(' ')
            hash[lineSplit[0]] = lineSplit[2].strip()
    return hash

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', dest='size', type=int, default=256,
                        help='Specify the size of to resize the imagery.')
    parser.add_argument('--map', dest='map', type=str, 
                        help='File that maps the synset names to the ' + 
                             'actual naming.')
    parser.add_argument('train', help='Directory containing the folders ' + 
                                      'named as ImageNet synsets')
    options = parser.parse_args()

    # rename all the directories
    remap = createReMap(options.map)
    for synset in os.listdir(options.train) :
        if synset in remap.keys() :
            os.rename(os.path.join(options.train, synset),
                      os.path.join(options.train, remap[synset]))

    # walk each directory and resize any files that don't
    # match the requested size
    resizeDims = (options.size, options.size)
    for root, dirs, files in os.walk(options.train) :
        for image in files :
            image = os.path.join(root, image)
            Image.open(image).resize(resizeDims).save(image)
