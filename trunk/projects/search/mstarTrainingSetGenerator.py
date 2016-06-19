'''Generates a series of chips using an MSTAR SICD and the research XML. The
   chips are placed into a directories of the same name as the label. The 
   filter argument allows the user to filter the chips based on different
   label types.   
'''
import os
from lxml import etree
from pysix import six_sicd
from coda import sio_lite
from six import text_type

def parseXML(filePath) : 
    '''Parse the XML using etree'''
    from io import StringIO
    with open(filePath, 'r') as f: 
        return etree.parse(StringIO(text_type(f.read())))

def parseMSTAR (options) :
    '''Walks the directory structure and chips the imagery into directories.'''
    from pysix.six_base import VectorString
    import fnmatch
    
    if not options.input.endswith(os.path.sep) :
        options.input += os.path.sep
    
    schemaPaths = VectorString()
    if options.schema is not None :
        schemaPaths.push_back(options.schema)

    # search for imagery
    sicds = []
    for root, dirs, files in os.walk(options.input) :
        if options.dirFilter is None :
            sicds.extend([os.path.join(root, f) \
                          for f in fnmatch.filter(files, '*_SICD.nitf')])
        else :
            if len([d for d in options.dirFilter.split(',') if d in root]) > 0 : 
                sicds.extend([os.path.join(root, f) \
                              for f in fnmatch.filter(files, '*_SICD.nitf')])

    # start processing
    halfChip = int(options.chipSize / 2)
    labels = {}
    for sicd in sicds :
        
        print ("Processing [" + sicd + "]")

        # read the SICD
        sioData, cmplx = six_sicd.read(sicd, schemaPaths)

        # the research XML explains the locations of all vehicles
        xmlFile = sicd.replace('_SICD.nitf', '-research.xml')
        xmlETree = parseXML(xmlFile)
        for observation in xmlETree.getroot().iter("Object") :

            # check if this label has been seen before
            label = observation.find(options.filter).text
            labelDir = os.path.join(options.outputDir, label)
            if label not in labels.keys() :
                labels[label] = 0
                if not os.path.exists(labelDir) : os.makedirs(labelDir)

            Row = observation.find('.//Row')
            Col = observation.find('.//Col')
            if Row != None and Col != None :
                
                outFile = sicd.replace(options.input, '')
                outFile = outFile.replace(os.path.sep, '_')
                outFile = outFile.replace('SICD.nitf', str(labels[label]))
                outFile = os.path.join(labelDir, outFile)
                labels[label] += 1

                # write the XML if it was requested
                if options.writeXML :

                    newXmlETree = etree.ElementTree(observation)
                    newRoot = newXmlETree.getroot()
                    
                    def addElement(root, elem, data) :
                        newElem = etree.SubElement(root, elem)
                        newElem.text = data
                    addElement(newRoot, 'researchXmlPath', xmlFile)
                    addElement(newRoot, 'sicdXmlPath', sicd)
                    newXmlETree.write(outFile + '.xml')

                # write the chip
                rowStart = max(int(Row.text) - halfChip, 0)
                colStart = max(int(Col.text) - halfChip, 0)
                sio_lite.write(sioData[rowStart : rowStart + options.chipSize, 
                                       colStart : colStart + options.chipSize],
                               outFile + '.sio')

if __name__ == "__main__" :

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--chip', dest='chipSize', type=int, default=67,
                        help='Pixels dimensions for each chip.')
    parser.add_argument('--filter', dest='filter', default='SystemName',
                        help='Name of the xml field to filter imagery.')
    parser.add_argument('--schema', dest='schema', type=str, default=None,
                        help='Path to SIX_SCHEMA_PATH.')
    parser.add_argument('--dirs', dest='dirFilter', type=str, default=None,
                        help='Specify a subset of directories to process.')
    parser.add_argument('--xml', dest='writeXML',
                        default=False, action='store_true', 
                        help='Write XML descriptions for the chips.')
    parser.add_argument('--outDir', dest='outputDir', default='./output',
                        help='Path to output the training set.')
    parser.add_argument('input', help='Path to MSTAR directory.')

    parseMSTAR(parser.parse_args())
