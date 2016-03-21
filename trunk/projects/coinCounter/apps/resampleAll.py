import os, sys

resizeDims = (30,30)

def resize(image) :
    try :
        from PIL import Image
        im = Image.open(image)
        im.load()
        imageDims = im.size
        print image + ": " + str(imageDims)
        if imageDims[0] < resizeDims[0] or imageDims[1] < resizeDims[1] :
            im = im.resize(resizeDims, Image.BICUBIC)
        elif imageDims[0] > resizeDims[0] or imageDims[1] > resizeDims[1] :
            im.thumbnail(resizeDims, Image.ANTIALIAS)
        im.save(image)
    except WindowsError as ex :
        os.remove(image)
if __name__ == '__main__' :
    print sys.argv

    if os.path.isdir(sys.argv[1]) :
        for image in os.listdir(sys.argv[1]) :
            image = os.path.join(sys.argv[1], image)
            resize(image)
    else :
        resize(sys.argv[1])