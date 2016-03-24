import os, sys

resizeDims = (30,30)

def resize(image) :
    from PIL import Image
    im = Image.open(image)
    im.load()
    imageDims = im.size
    if imageDims != resizeDims :
        print image
if __name__ == '__main__' :
    print sys.argv

    if os.path.isdir(sys.argv[1]) :
        for image in os.listdir(sys.argv[1]) :
            image = os.path.join(sys.argv[1], image)
            resize(image)
    else :
        resize(sys.argv[1])