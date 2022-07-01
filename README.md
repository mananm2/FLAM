# Federated Learning for Additive Manufacturing (FLAM)

This repository contains the code and data for the paper:

#### [Federated learning-based semantic segmentation for pixel-wise defect detection in additive manufacturing](https://www.sciencedirect.com/science/article/pii/S0278612522001054)

Before getting started, download dependencies using
```
pip install -r requirements.txt
```

The repository contains three main folders along with jupyter notebooks to demonstrate the case studies in the paper. The notebooks and functions have sufficient comments to make the code self-explanatory. It is recommended to use [Google Colab](https://colab.research.google.com/?utm_source=scs-index) with [GPU acceleration](https://colab.research.google.com/notebooks/gpu.ipynb) if models need to be trained from scratch.

### Data
The data used in the notebooks is in the `data` folder. The L-PBF images are in `data/Laser Powder Bed Fusion` where folders `0`, `1`, and `annotations` are for post-spreading images (.jpg and .tif), post-fusion images (.jpg and .tif), and the segmentation masks (.npy) respectively. The code only uses .jpg file extensions.

**Note that this data was originally collected and compiled at Oak Ridge National Laboratory and is available [here](https://www.osti.gov/dataexplorer/biblio/dataset/1779073).** Please consider citing the [dataset](https://www.osti.gov/dataexplorer/biblio/dataset/1779073) (doi:10.13139/ORNLNCCS/1779073) and the [related work](https://www.sciencedirect.com/science/article/pii/S2214860420308253) appropriately if the data is used.

### Functions
The functions used throughout the notebooks are provided in the `utils` folder. Function descriptions are provided below each declaration. Functions are grouped together into files based on usage, with 5 main files. The file and function names are self-explanatory.

### Pre-trained Models
The pre-trained models are provided in the `saved_models` folder for the user to load pre-trained federated and centralized models. The naming convention is as follows:

For centralized learning, file names follow 

`CL_NumberOfEpochs_MiniBatchSize_LearningRate_TestSetNumber.h5`

So the file name `CL_100_32_8e05_HoldoutPart0708.h5` means the CL model is trained for 100 epochs with a mini-batch size of 32, learning rate of 8e-05, and Parts 07 and 08 (file ID 0000007 and 0000008) are used as test set (excluded from training). These parts can be viewed in the `data` folder and correspond to Client 6 in the paper.

For federated learning, file names follow 

`FL_NumberOfServerRounds_NumberOfLocalEpochs_LocalMiniBatchSize_LocalLearningRate_TestSetNumber.h5`

So the file name `FL_30_10_32_8e05_HoldoutPart1213.h5` means the FL model is trained for 30 server rounds with 10 local epochs (effectively 300 epochs), with local mini-batche size of 32, local learning rate of 8e-05, and Parts 12 and 13 (Client 7 in the paper) used as test set (excluded from training).

To load any model, use
```
new_model = tf.keras.models.load_model('saved_models/model_name.h5')
```
### Jupyter Notebooks
Three notebooks are provided for demonstration purposes. The code is structured in a manner that a user can change the client composition before training.

`Federated_learning.ipynb`: Demonstrates the learning procedure and results for an FL model.

`Centralized_learning.ipynb`: Demonstrates the learning prcedure and results for a CL model.

`Comparisons.ipynb`: Reproduces the case study results from the paper using pre-trained models.
