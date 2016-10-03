from flask import Flask, render_template

import os
import fnmatch
import random
import PIL.Image as Image
from inferEngine import classifyImage, parseImage

app = Flask(__name__)

# mutable globals
network = 0
batchSet = 0
opts = 0
threads = 0
labelColors = 0
cssStyle = 0

# static globals
colors = [[234,96,53],
          [98,70,107],
          [145, 0, 232],
          [67,255,20],
          [202,186,200],
          [249,160,63],
          [66,245, 235],
          [147, 232, 157],
          [33, 37, 99],
          [255, 0, 12],
          [255, 223, 15],
          [249, 67, 94],
          [31, 26, 49],
          [255,10,240],
          [28, 60, 99],
          [0, 95, 203],
          [218, 76, 57],
          [85, 61, 0],
          [255, 229, 170],
          [28, 74, 0],
          [128, 186, 93],
          [177, 223, 149],
          [38, 115, 86],
          [72, 144, 117],
          [115, 172, 150],
          [19, 52, 83],
          [42, 78, 110],
          [74, 107, 138],
          [114, 141, 165],
          [60, 49, 118],
          [93, 83, 147],
          [134, 125, 176],
          [255,36,36]]

sicdDir = './demoImagery/'
staticDir = './static/'
imExt = '.jpeg'

def initializeGlobals(options) :
    import theano
    import numpy as np
    from sarSearch import preProcessing
    from nn.net import LabeledClassifierNetwork as Net

    global network, batchSet, opts, threads, labelColors, cssStyle

    network = Net(filepath=options.synapse)

    # allocate memory for our batch size --
    # this will allow fast classification
    batchSet = np.zeros(network.getNetworkInputSize(), 
                        dtype=theano.config.floatX)

    # setup for processing
    opts, _, threads = preProcessing(options.confDir)

    labelColors = {}
    cssStyle = ''
    for ii, label in enumerate(network.getNetworkLabels()) :
        labelColors[label] = colors[ii]
        hexCode = ('%02x%02x%02x' % tuple(labelColors[label]))
        labelStr = label.replace('-', ' ').replace(' ', '_')
        cssStyle += 'img.' + labelStr + '{\nborder: 4px #' + hexCode + ' solid;\n}\n'
        cssStyle += 'img.' + labelStr + ' + p.banner {\nbackground-color: #' + hexCode + ';\n}\n'

def getSICDs(sicdDir) :
    return [os.path.join(sicdDir, f) for f in os.listdir(sicdDir) \
            if f.endswith('SICD.nitf')]

def scaleOverview(image, width=500) :
    factor = float(width) / image.size[0]
    image.thumbnail((width, int(image.size[1] * factor)),
                    Image.ANTIALIAS)
    return image.convert('RGB'), factor

def sicdBase(sicd) :
    return os.path.basename(sicd)[:7]
def getBase(sicd) :
    return os.path.join(staticDir, sicdBase(sicd))

def processSICD(sicd) :
    import os
    from PIL import ImageDraw

    # check if this image has been processed before
    if len(fnmatch.filter(os.listdir(staticDir), sicdBase(sicd) + '*')) < 2 :

        # process SICD
        wbData, objects = parseImage(sicd)
        over, chips, locs, results = classifyImage(
            network, batchSet, wbData, objects, opts, threads, None)

        # extract info
        labels = network.convertToLabels(results[0])
        confidence = []
        for ii, index in enumerate(results[0]) :
            confidence.append(results[1][ii][index])

        # get a base locations for naming
        base = getBase(sicd)

        # write the chips
        for ii, chip in enumerate(chips) :
            labelStr = labels[ii].replace('-', ' ').replace(' ', '_')
            Image.fromarray(chip).save(base + '-' + labelStr + 
                                              '-' + str(confidence[ii]) + 
                                              '-__' + str(ii) + imExt)

        # scale the overview image
        over, factor = scaleOverview(over)

        # apply the locations to the overview image
        draw = ImageDraw.Draw(over)
        for label, loc in zip(labels, locs) :
            # draw a box around the object
            if label.upper() != "MISC" :
                draw.rectangle([int(loc[0]*factor), int(loc[1]*factor),
                                int(loc[2]*factor), int(loc[3]*factor)],
                               fill=tuple(labelColors[label] + [5]))

        # write the overview image
        over.save(base + '_class' + imExt)

    return getBase(sicd) + '_class' + imExt

def detectedFull(sicd) :
    from sarSearch import detectImage
    filename = getBase(sicd) + '_full' + imExt

    # check if this product has a detected counterpart        
    if not os.path.exists(filename) :
        wbData, _ = parseImage(sicd)
        over = Image.fromarray(detectImage(wbData, opts, threads))
        over, _ = scaleOverview(over)
        over.save(filename)
    return filename

def genImageSlider(images) :
    # generate the text block for these images
    rand = str(random.random())
    textBlock = ''
    for im in images :
        name = im.split('-')
        textBlock += '<div class="swiper-slide blowup"><img class="{0}" ' \
                     'src="/static/{3}?{4}" />'\
                     '<p class="banner">{1} : {2}</p></div>'.format(
                         name[1], name[1].replace('_', ' '), name[2], im, rand)
    return textBlock

def getFullOverviews() :
    return [detectedFull(sicd) for sicd in getSICDs(sicdDir)]

def genOverview() :
    rand = str(random.random())
    textBlock = ''
    for sicd in getSICDs(sicdDir) :
        baseSicd = os.path.basename(sicd)
        
        filename = getBase(sicd) + '_class' + imExt
        if not os.path.exists(filename) :
            filename = getBase(sicd) + '_full' + imExt
        filename += '?' + rand
        textBlock += '<div class="swiper-slide" style="background-image:url(../{0})">' \
                         '<button id="{1}" ><img src="/static/icon_red_search.png" /></button>' \
                         '<iframe scrolling="no" src="/slider/{1}"></iframe>' \
                     '</div>\n'.format(filename, baseSicd)

    return textBlock

def wipeAll() :
    
    imageList = [os.path.join(staticDir, im) for im in os.listdir(staticDir)]
    deleted = []
    for sicd in getSICDs(sicdDir) :
        base = getBase(sicd)
        images = fnmatch.filter(imageList, base + '*__*' + imExt) + \
                 fnmatch.filter(imageList, base + '_class*' + imExt)
        deleted.extend(images)
        for image in images :
            os.remove(image)
    return str(deleted)

@app.route('/process/<sicd>')
def processImage(sicd) :
    return str(processSICD(os.path.join(sicdDir, sicd)))

@app.route('/slider/<sicd>')
def getSlider(sicd) :
    images = fnmatch.filter(os.listdir(staticDir), sicd[:7] + '*__*' + imExt)
    return render_template(
        'image.html', sicd=sicd,
        datetime=str(random.random()), 
        chipBlock=genImageSlider(images),
        cssStyle=cssStyle)

@app.route('/cnn/')
def getOveriew() :
    getFullOverviews()
    return render_template(
        'index.html', 
        datetime=str(random.random()),
        overviewBlock=genOverview())

@app.route('/reset/')
def reset() :
    return wipeAll()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', dest='confDir', type=str, default=None,
                        help='Specify the conf/ directory.')
    parser.add_argument('--log', dest='logfile', type=str, default=None,
                        help='Specify log output file.')
    parser.add_argument('--level', dest='level', default='INFO', type=str, 
                        help='Log Level.')
    parser.add_argument('--syn', dest='synapse', type=str, required=True,
                        help='Load from a previously saved network.')
    parser.add_argument('--color', dest='color', type=int, default=0,
                        help='Load a particular color palette.')
    options = parser.parse_args()

    initializeGlobals(options)
    app.run(host='0.0.0.0', port=5020)
