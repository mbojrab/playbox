import os, sys

resizeDims = (40,40)

def rotate(image) :
    from PIL import Image
    im = Image.open(image)
    im.load()

    # rotate and save the images
    for deg in [90, 180, 270] :
        imRot = im.rotate(deg)
        imRot.save(image.replace('.tiff', '_' + str(deg) + '.tiff'))
if __name__ == '__main__' :
    print sys.argv

    if os.path.isdir(sys.argv[1]) :
        for image in os.listdir(sys.argv[1]) :
            image = os.path.join(sys.argv[1], image)
            rotate(image)
    else :
        rotate(sys.argv[1])