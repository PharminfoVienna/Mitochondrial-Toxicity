# This repository contains supporting information, as well as the files for the paper "Using machine learning methods and structural alerts for prediction of mitochondrial toxicity"

The repository contains the neural network (deep learning) model as well as the gradient boosting model which were presented in our paper. In addition it also contains the datasets as well as the supporting information. 

The folders for the Models have their own README to explain how models can be used to predict your own data. 

In the near future we try to integrate the models in the Vienna Livertox Workspace at https://livertox.univie.ac.at/


## Support:
--------
In case of any problems, please open an issue here in the repository or contact us via email.

Jennifer Hemmerich, jennifer.hemmerich[at]univie.ac.at

## Dependencies
--------------------

### Gradient Boosting:
 The gradient boosting model depends on 
 - KNIME (https://www.knime.com/) >= 3.7.0
 - RDKit Nodes for KNIME
 - Machine learning extensions
 
 ### Neural Network
 The Neural network was trained with the script developed in our COVER paper (https://github.com/PharminfoVienna/COVER-Conformational-Oversampling). Please have a look at the dependencies there. 
 To run predictions you need:
 
 Python 3.6 >=, with
 - scikit learn >= 20.X
 - tensorflow >= 1.12.X but < 2.X
 - keras
 - pandas 
 - numpy
 - RDKit >= 2019.X
 
 
 ## Information
 
 The folder data contains the training and test dataset for the models. The label activity contains
 the activity values, 1 denotes that a compound s toxic, 0 is non toxic.
 
 The supplemental information contains the used descriptors (Supplement A), the Drugbank Molecules with positive predictions or structural alerts and the respective literature indicationg a possible mitochondrial toxicity (Supplement B). Supplement C contains the SMILES patterns of the alerts as well as their predictivty in terms of positive predictive value.
 
 For explanations of the models see:
 
 [Gradient Boosting](GradientBoosting Model/README.md)
 
 [Neural networks](NeuralNetwork Models/README.md)
 

