from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import keras
from keras.models import model_from_json
import keras.backend as K
import tensorflow as tf
import os
import json
from sklearn.metrics import mean_squared_error, matthews_corrcoef, roc_auc_score, hamming_loss, cohen_kappa_score
from math import sqrt
from pickle import load


def evaluate(y, pred):
    """Evaluates  the performance of a model

    Args:
    y: true values (as pd.Series)
    y_pred: predicted values of the model (as np.array)

    Returns:
    dictionary: dictionary with all calculated values
    """
    y = np.asarray(y.to_frame())
    classes = np.greater(pred, 0.5).astype(int)
    tp = np.count_nonzero(classes * y)
    tn = np.count_nonzero((classes - 1) * (y - 1))
    fp = np.count_nonzero(classes * (y - 1))
    fn = np.count_nonzero((classes - 1) * y)

    # Calculate accuracy, precision, recall and F1 score.
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = np.NaN
    try:
        sensitivity = tp / (tp + fn)
    except ZeroDivisionError:
        sensitivity = np.NaN
    try:
        specificity = tn / (tn + fp)
    except ZeroDivisionError:
        specificity = np.NaN

    bal_acc = (sensitivity + specificity) / 2
    try:
        fpr = fp / (fp + tn)
    except ZeroDivisionError:
        fpr = np.NaN
    try:
        fnr = fn / (tp + fn)
    except ZeroDivisionError:
        fnr = np.NaN
    try:
        fmeasure = (2 * precision * sensitivity) / (precision + sensitivity)
    except ZeroDivisionError:
        fmeasure = np.nan
    mse = mean_squared_error(classes, y)
    mcc = matthews_corrcoef(y, classes)
    youden = sensitivity + specificity - 1
    try:
        AUC = roc_auc_score(y, pred)
    except ValueError:
        AUC = np.nan
    hamming = hamming_loss(y, classes)
    kappa = cohen_kappa_score(y, classes)
    gmean = sqrt(sensitivity * specificity)

    ret = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
           'acc': accuracy, 'bal_acc': bal_acc, 'sens': sensitivity, 'spec': specificity, 'fnr': fnr, 'fpr': fpr,
           'fmeas': fmeasure, 'mse': mse, 'youden': youden, 'mcc': mcc, 'auc': AUC,
           'hamming': hamming, 'cohen_kappa': kappa, 'gmean': gmean}

    return ret


def load_mymodel(path, jsonfile, weightsfile):
    modelpath = os.path.join(path, jsonfile)
    weightpath = os.path.join(path, weightsfile)

    json_file = open(modelpath, 'r')
    loaded_model_json = json_file.read()
    p = json.loads(loaded_model_json)

    model = model_from_json(p)
    model.load_weights(weightpath)

    return (model)


def build_masked_loss(loss_function, mask_value):
    """Builds a loss function that masks based on targets
    taken from: https://github.com/bioinf-jku/FCD/blob/master/fcd/FCD.py

    Args:
    loss_function: The loss function to mask
    mask_value: The value to mask in the targets

    Returns:
    function: a loss function that acts like loss_function with
    masked inputs
    """

    def masked_loss_function(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())

        return loss_function(y_true * mask, y_pred * mask)

    return 0


if __name__ == "__main__":
    # load json and create model
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # order devices
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    now = datetime.now().strftime("%d%m%Y_%H%M%S")
    performance_file = "_".join(["Performance", now])
    predictions_file = "_".join(["Predictions", now])

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default=os.getcwd(), type=str, required=True,
                        help="Specifies the input file which should be predicted. This argument is REQUIRED. The file needs to be an sdf file!")
    parser.add_argument("--activity-column", type=str, default="activity",
                        help="Specifies the sdf tag which contains the compounds activities. Defaults to <activity>")
    parser.add_argument("--model-dir", default="./Final_Models", type=str,
                        help="Specifies the directory where the model files are located. Defaults to directory Final_models, found in the working directory.")
    parser.add_argument("--output-dir", default=os.getcwd(), type=str,
                        help="Specifies the directory for the output files. Defaults to current working directory.")
    parser.add_argument("--output-file-predictions", default=performance_file, type=str,
                        help="Specifies the filename for the output files. Defaults to predictions_<current_date>.")
    parser.add_argument("--output-file-performance", default=predictions_file, type=str,
                        help="Specifies the filename for the output files. Defaults to performance_<current_date>.")
    parser.add_argument("--compound-ids", default="ID", type=str,
                        help="Specifies the column which contains the compound IDs (if present in the dataset).Defaults to <ID>.")
    parser.add_argument("--scaler", default="./Final_Models/scaler.pkl", type=str,
                        help="Specifies the file which contains the scaler for the input descriptors  Defaults to scaler in ./Final_models/scaler.pkl.")

    keras.losses.masked_loss_function = build_masked_loss
    input_args = parser.parse_args()

    model_dir = input_args.model_dir

    paths = filter(os.path.isdir, os.listdir(model_dir))

    metr = {}
    eval_test = {}

    data = pd.read_csv(input_args.input_file)

    data = data.dropna(axis=0)
    id_column = input_args.compound_ids
    activity_column = input_args.activity_column
    ids = data[id_column]
    data = data.set_index(id_column)

    preds = pd.DataFrame(data[activity_column])
    y = data[activity_column]
    model_input = data.drop([activity_column], 1)
    scaler = load(open(input_args.scaler, 'rb'))
    model_input = scaler.transform(model_input)
    loops = sorted(filter(lambda x: x.startswith("Loop"), os.listdir(model_dir)))

    for i, loop in enumerate(loops):
        loopdir = os.path.join(model_dir, loop)
        w = pd.DataFrame(sorted(os.walk(loopdir, topdown=False)))
        dirs = w.iloc[0][1][0]
        files = w.iloc[1][2]

        modelfile = list(filter(lambda x: x.endswith(".hdf5"), files))[0]
        modeldir = os.path.join(loopdir, dirs, modelfile)
        model = tf.keras.models.load_model(modeldir, custom_objects={'tf': tf}, compile=False)

        pred = model.predict(model_input)

        preds[loop] = pred.reshape(-1)
        eval_test[loop] = evaluate(y, pred)

        print("Done prediction model for", loop)

    eval_test = pd.DataFrame.from_dict(eval_test, orient="index")

    preds.to_csv(os.path.join(input_args.output_dir, input_args.output_file_performance))
    eval_test.to_csv(os.path.join(input_args.output_dir, input_args.output_file_predictions))

    print("Done")
