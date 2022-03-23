import numpy as np
import tensorflow as tf

def split_image(image, tileSize):
    '''
    Takes an image and returns a tensor of tiled images.
    
    '''
    image_shape = tf.shape(image)
    tile_rows = tf.reshape(image, [image_shape[0], -1, tileSize, image_shape[2]])
    serial_tiles = tf.transpose(tile_rows, [1, 0, 2, 3])
    
    return tf.reshape(serial_tiles, [-1, tileSize, tileSize, image_shape[2]])


def unsplit_image(tiles, height, width):
    '''
    Takes a tiled image and puts it back together as a single image.
    
    '''
    tileSize = tf.shape(tiles)[1]
    serializedTiles = tf.reshape(tiles, [-1, height, tileSize, 2])
    rowwiseTiles = tf.transpose(serializedTiles, [1, 0, 2, 3])
    return tf.reshape(rowwiseTiles, [1, height, width, 2])



def unsplit_image_mask(tiles, height, width):
    '''
    Takes a tiled segmask and puts it back together as a single mask.
    
    '''
    tileSize = tf.shape(tiles)[1]
    serializedTiles = tf.reshape(tiles, [-1, height, tileSize, 1])
    rowwiseTiles = tf.transpose(serializedTiles, [1, 0, 2, 3])
    return tf.reshape(rowwiseTiles, [1, height, width, 1])


def preprocess_image(im0, im1, segmentationMask, tileSize):
    '''
    Image preprocessing (see supplementary info in the paper)
    
    '''
    # Crop image to a multiple of tileSize
    rightcrop, bottomcrop = im0.size[0]//tileSize * tileSize, im0.size[1]//tileSize * tileSize  
    im0, im1 = im0.crop((0,0,rightcrop,bottomcrop)), im1.crop((0,0,rightcrop,bottomcrop))
    imarray0, imarray1 = np.array(im0), np.array(im1)
    segmentationMask = np.copy(segmentationMask[:bottomcrop, :rightcrop])
    
    # Label the very few unlabeled pixels as powder (label 0)
    segmentationMask[segmentationMask == -1] = 0
    
    # Combine all defects into a single class 'defect' with label 2
    segmentationMask[(segmentationMask!=0) & (segmentationMask!=1)] = 2
    
    # Normalize pixel values to [0,1] range
    imtensor0 = tf.cast(imarray0.reshape(im0.size[0], im0.size[1], 1), tf.float32)/255.0
    imtensor1 = tf.cast(imarray1.reshape(im1.size[0], im1.size[1], 1), tf.float32)/255.0
    
    # Tile the image and mask
    splitImages0 = split_image(imtensor0, tileSize)
    splitImages1 = split_image(imtensor1, tileSize)
    splitImages = tf.concat([splitImages0, splitImages1], -1)
    splitSegmentationMask = tf.reshape(split_image(np.expand_dims(segmentationMask, -1), tileSize),(-1,tileSize, tileSize,))
    splitSegmentationMask = tf.cast(splitSegmentationMask, tf.int32)
    
    return splitImages, splitSegmentationMask
