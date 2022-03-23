import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
plt.rcParams.update({'font.size': 16})

from utils.dataset_functions import unwrap_client_data
from utils.image_processing import *

def visualize_results_testset(model, imageDictTest, segMaskDictTest, 
                              testClients, clientIdentifierDict):
    '''
    Result visualization for a model evaluated on testClients
    
    '''
    testImages, testMasks = unwrap_client_data(imageDictTest, segMaskDictTest, testClients)
    curr_idx = 0
    for clientID in testClients:
        print('For '+clientID+'...')
        for build in clientIdentifierDict[clientID]:
            print(build)
            prev_idx = curr_idx
            if build == '0000216':
                curr_idx = prev_idx + 162
                imageheight, imagewidth = 9*128, 18*128
            elif build == '0000161':
                curr_idx = prev_idx + 256
                imageheight, imagewidth = 16*128, 16*128
            else:
                curr_idx = prev_idx + 324
                imageheight, imagewidth = 18*128, 18*128
            
            predictedImages = model.predict(testImages[prev_idx:curr_idx])
            predictedMask = tf.argmax(predictedImages, axis=-1)

            fullImage = unsplit_image(testImages[prev_idx:curr_idx], imageheight, imagewidth)
            fullPredictedMask = unsplit_image_mask(predictedMask, imageheight, imagewidth)
            fullTrueMask = unsplit_image_mask(testMasks[prev_idx:curr_idx], imageheight, imagewidth)

            metric = tf.keras.metrics.MeanIoU(num_classes = 3)
            metric.update_state(fullTrueMask, fullPredictedMask)
            print("MeanIoU=", np.round(metric.result().numpy(), 3))

            fig, arr = plt.subplots(1, 3, figsize=(15, 15))
            arr[0].imshow(fullImage[0,:,:,0], cmap = 'gray')
            arr[0].set_title('Original Image')
            arr[1].imshow(fullTrueMask[0,:,:,0], cmap = 'viridis')
            arr[1].set_title('Actual Segmentation Mask')
            arr[2].imshow(fullPredictedMask[0,:,:,0], cmap = 'viridis')
            arr[2].set_title('Predicted Segmentation Mask')
            plt.show()


def compare_results_testset(cl_model, fl_model, imageDictTest, segMaskDictTest, 
                              testClients, clientIdentifierDict):
    '''
    Result comparison between cl_model and fl_model on testClients
    Generates the plots seen in the paper
    
    '''
    testImages, testMasks = unwrap_client_data(imageDictTest, segMaskDictTest, testClients)
    curr_idx = 0
    for clientID in testClients:
        print('For '+clientID+'...')
        for build in clientIdentifierDict[clientID]:
            print(build)
            prev_idx = curr_idx
            if build == '0000216':
                curr_idx = prev_idx + 162
                imageheight, imagewidth = 9*128, 18*128
            elif build == '0000161':
                curr_idx = prev_idx + 256
                imageheight, imagewidth = 16*128, 16*128
            else:
                curr_idx = prev_idx + 324
                imageheight, imagewidth = 18*128, 18*128
    
            cl_predictedImages = cl_model.predict(testImages[prev_idx:curr_idx])
            cl_predictedMask = tf.argmax(cl_predictedImages, axis=-1)
            fl_predictedImages = fl_model.predict(testImages[prev_idx:curr_idx])
            fl_predictedMask = tf.argmax(fl_predictedImages, axis=-1)

            fullImage = unsplit_image(testImages[prev_idx:curr_idx], imageheight, imagewidth)
            fullTrueMask = unsplit_image_mask(testMasks[prev_idx:curr_idx], imageheight, imagewidth)
            cl_fullPredictedMask = unsplit_image_mask(cl_predictedMask, imageheight, imagewidth)
            fl_fullPredictedMask = unsplit_image_mask(fl_predictedMask, imageheight, imagewidth)

            cl_metric = tf.keras.metrics.MeanIoU(num_classes = 3)
            cl_metric.update_state(fullTrueMask, cl_fullPredictedMask)
            fl_metric = tf.keras.metrics.MeanIoU(num_classes = 3)
            fl_metric.update_state(fullTrueMask, fl_fullPredictedMask)
            cl_values = np.array(cl_metric.get_weights()).reshape(3,3)
            fl_values = np.array(fl_metric.get_weights()).reshape(3,3)

            cl_iou_powder = cl_values[0,0]/(cl_values[0,0] + cl_values[0,1] + cl_values[0,2] + cl_values[1,0] + cl_values[2,0])
            cl_iou_part = cl_values[1,1]/(cl_values[1,1] + cl_values[1,0] + cl_values[1,2] + cl_values[0,1] + cl_values[2,1])
            cl_iou_defect = cl_values[2,2]/(cl_values[2,2] + cl_values[2,0] + cl_values[2,1] + cl_values[0,2] + cl_values[1,2])
            fl_iou_powder = fl_values[0,0]/(fl_values[0,0] + fl_values[0,1] + fl_values[0,2] + fl_values[1,0] + fl_values[2,0])
            fl_iou_part = fl_values[1,1]/(fl_values[1,1] + fl_values[1,0] + fl_values[1,2] + fl_values[0,1] + fl_values[2,1])
            fl_iou_defect = fl_values[2,2]/(fl_values[2,2] + fl_values[2,0] + fl_values[2,1] + fl_values[0,2] + fl_values[1,2])

            print("CL / FL :")
            print("Powder IoU = ", np.round(cl_iou_powder,3), " / ", np.round(fl_iou_powder,3))
            print("Part IoU = ", np.round(cl_iou_part,3), " / ", np.round(fl_iou_part,3))
            print("Defect IoU = ", np.round(cl_iou_defect,3), " / ", np.round(fl_iou_defect,3))
            print("MeanIoU = ", np.round(cl_metric.result().numpy(),3), " / ", np.round(fl_metric.result().numpy(),3))
            
            fig, arr = plt.subplots(1, 4, figsize=(20, 5))
            arr[0].imshow(fullImage[0,:,:,0], cmap = 'gray')
            arr[0].set_title('Input Image\nL-'+build[-3:])
            arr[1].imshow(fullTrueMask[0,:,:,0], cmap = 'viridis')
            arr[1].set_title('Ground Truth\nSegmentation Mask')
            arr[1].axis('off')
            arr[2].imshow(cl_fullPredictedMask[0,:,:,0], cmap = 'viridis')
            arr[2].set_title('Predicted Mask with CL\nMeanIoU = {:.3f}'.format(cl_metric.result().numpy()))
            arr[2].axis('off')
            arr[3].imshow(fl_fullPredictedMask[0,:,:,0], cmap = 'viridis')
            arr[3].set_title('Predicted Mask with FL\nMeanIoU = {:.3f}'.format(fl_metric.result().numpy()))
            arr[3].axis('off')

            plt.show()

