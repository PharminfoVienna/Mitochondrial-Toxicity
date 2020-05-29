# Gradient Boosting KNIME model

The model was trained with the workflow 

## Dependencies
The prediction workflow requires the following KNIME extensions:

- KNIME H2O Machine Learning Integration - MOJO Extension
- KNIME Base Chemistry Types & Nodes
- RDKit KNIME integration
- KNIME Quick Form Nodes

## Use the workflow

To use the workflow you need your data as an *.sdf file. To load your data into KNIME double klick on the setup node and enter the necessary information. If you so not want to use the obtained predictions in KNIME choose one of the last nodes (csv writer, xls writer or sdf writer) and enter where the results should be stored. Execute the workflow.The resulting file will contain the predictions of the molecules, you can use this further to calculate the important metrics such as the confusion matrix. 

In case H2O nodes give an error please set the H2O local context in the KNIME preferences to version 3.10, this should enable H2O to read the model.