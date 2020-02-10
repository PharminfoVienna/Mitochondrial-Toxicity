# Neural network models

To predict your data you need your dataset as an sdf file and follow these steps:

1. Calculate the descriptors using the Calculate_descriptors.knwf KNIME workflow. The resulting csv file can be used as an input for the model. To try it out if it works you can use the example.sdf file which contains a few molecules.
 NOTE: You can also use your own script or workflow for the calculation of descriptors, however the order of the descriptors has to be the same as in the example_predictions.py file.

2. run the predictions.py file the options are as follows:

  predictions.py [-h] --input-file INPUT_FILE
                      [--activity-column ACTIVITY_COLUMN]
                      [--model-dir MODEL_DIR] [--output-dir OUTPUT_DIR]
                      [--output-file-predictions OUTPUT_FILE_PREDICTIONS]
                      [--output-file-performance OUTPUT_FILE_PERFORMANCE]
                      [--compound-ids COMPOUND_IDS]

  optional arguments:
    -h, --help            show this help message and exit
    --input-file INPUT_FILE
                          Specifies the input file which should be predicted.
                          This argument is required. The file needs to be an sdf
                          file!
    --activity-column ACTIVITY_COLUMN
                          Specifies the sdf tag which contains the compounds
                          activities. Defaults to <activity>
    --model-dir MODEL_DIR
                          Specifies the directory where the model files are
                          located. Defaults to current working directory.
    --output-dir OUTPUT_DIR
                          Specifies the directory for the output files. Defaults
                          to current working directory.
    --output-file-predictions OUTPUT_FILE_PREDICTIONS
                          Specifies the filename for the output files. Defaults
                          to predictions_<current_date>.
    --output-file-performance OUTPUT_FILE_PERFORMANCE
                          Specifies the filename for the output files. Defaults
                          to performance_<current_date>.
    --compound-ids COMPOUND_IDS
                          Specifies the column which contains the compound IDs

If you use the same folder structure as in the GitHub project the script should run as follows

``
python predictions.py --input-file example_predictions.csv
``

The script will create two files: Performance_DDMMYYY_HHMMSS.csv and Predctions_DDMMYYY_HHMMSS.csv. The Performance file contains various performance metrics for each model, whereas the predictions file contains the predctions as well as the activities for each molecule for each model. The ID can be used to match the initial structures back to the predictions for further analyis.


### Important note:
Please do not modify the structure of the Final_models folder. The script uses the provided structure to find the models and the scaler for the data, any modifications will break the script. For further help in terms of modifications please create an issue or contact the support.