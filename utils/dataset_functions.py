import numpy as np
import PIL
from PIL import Image
import tensorflow as tf

from utils.image_processing import *


def create_dataset(clientIdentifierDict, imagePath0, imagePath1, npyPath, tileSize = 128):
    '''
    Creates a dataset of clients using the post-spreading and post-fusion images
    based on clientIdentifierDict.
    
    Returns:
    - datasetImageDict: Dictionary of tiled images keyed by clientID with dimension (nTiles, tileSize, tileSize, 2)
    - datasetMaskDict : Dictionary of tiled masks keyed by clientID with dimension (nTiles, tileSize, tileSize)
    
    '''
    datasetImageDict, datasetMaskDict = {}, {}
    
    for clientID in clientIdentifierDict:
        fileNum = 0
        for fileName in clientIdentifierDict[clientID]:
            im0 = Image.open(imagePath0 + fileName + '.jpg')
            im1 = Image.open(imagePath1 + fileName + '.jpg')
            segmentationMask = np.load(npyPath + fileName + '.npy')
            
            splitImages, splitSegmentationMask = preprocess_image(im0, im1,
                                                                  segmentationMask,
                                                                  tileSize)
            if fileName == '0000216':
                # Don't keep tiles from the bottom half of the figure (unlabeled)
                idxToKeep = [i + j for i in np.arange(0, 324, 18) for j in range(9)]
                splitImages = tf.gather(splitImages, idxToKeep)
                splitSegmentationMask = tf.gather(splitSegmentationMask, idxToKeep)
                
            if fileNum == 0:
                clientImages, clientMasks = splitImages, splitSegmentationMask
            else:
                clientImages = tf.concat([clientImages, splitImages], 0)
                clientMasks = tf.concat([clientMasks, splitSegmentationMask], 0)
            fileNum += 1
        
        print('\n'+clientID+'...')
        print('Contains ' + str(fileNum) + ' images...')
        print('Tiled Image Tensor Shape: ', clientImages.shape)
        print('Tiled Mask Shape: ', clientMasks.shape)
        datasetImageDict[clientID] = clientImages
        datasetMaskDict[clientID] =  clientMasks
    
    return datasetImageDict, datasetMaskDict


def unwrap_client_data(imageDict, maskDict, clientList):
    '''
    Takes all clients in clientList and combines their data into a single tensor.
    
    '''
    unwrappedImages = imageDict[clientList[0]]
    unwrappedMasks = maskDict[clientList[0]]
    for i in range(1,len(clientList)):
        unwrappedImages = tf.concat([unwrappedImages, imageDict[clientList[i]]], 0)
        unwrappedMasks = tf.concat([unwrappedMasks, maskDict[clientList[i]]], 0)
    return unwrappedImages, unwrappedMasks