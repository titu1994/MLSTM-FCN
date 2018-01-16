# Multivariate LSTM-FCNs for Time Series Classification
MLSTM FCN models, from the paper [Multivariate LSTM-FCNs for Time Series Classification](https://arxiv.org/abs/1801.04503), 
augment the squeeze and excitation block with the state of the art univariate time series model, LSTM-FCN and ALSTM-FCN.

<img src="https://github.com/titu1994/MLSTM-FCN/blob/master/images/MLSTM-FCN.png?raw=true" height=100% width=100%>
<img src="https://github.com/titu1994/MLSTM-FCN/blob/master/images/MLSTM1-FCN.png?raw=true" height=100% width=100%>

# Installation 
Download the repository and apply `pip install -r requirements.txt` to install the required libraries. 

Keras with the Tensorflow backend has been used for the development of the models, and there is currently no support for Theano or CNTK backends. The weights have not been tested with those backends.

**Note** : The input to the Input layer of all models will be pre-shuffled to be in the shape (Batchsize, 1, Number of timesteps), and the input will be shuffled again before being applied to the CNNs (to obtain the correct shape (Batchsize, Number of timesteps, 1)). This is in contrast to the paper where the input is of the shape (Batchsize, Number of timesteps, 1) and the shuffle operation is applied before the LSTM to obtain the input shape (Batchsize, 1, Number of timesteps). These operations are equivalent.

#Multivariate Benchmark Datasets
<img src="https://github.com/titu1994/MLSTM-FCN/blob/master/images/MLSTM-FCN-benchmark1.png?raw=true" height=100% width=100%>


# Training and Evaluation
Various multivariate benchmark datasets can be evaluated with the provided code and weight files. Refer to the weights directory for clarification.

There is 1 script file for each dataset, and 4 major sections in the code. For each of these code files, please keep the line below uncommented. 

- To use the MLSTM FCN model : `model = generate_model()`
- To use the MALSTM FCN model : `model = generate_model_2()`
- To use the LSTM FCN model : `model = generate_model_3()`
- To use the ALSTM FCN model : `model = generate_model_4()`

## Training
To train the a model, uncomment the line below and execute the script. **Note** that '???????' will already be provided, so there is no need to replace it. It refers to the prefix of the saved weight file. Also, if weights are already provided, this operation will overwrite those weights.

`train_model(model, DATASET_INDEX, dataset_prefix='???????', epochs=2000, batch_size=128)` 

## Evaluate 
To evaluate the performance of the model, simply execute the script with the below line uncommented. 

`evaluate_model(model, DATASET_INDEX, dataset_prefix='???????', batch_size=128)`

# Results
<img src="https://github.com/titu1994/MLSTM-FCN/blob/master/images/MLSTM-FCN-scores1.png?raw=true" height=100% width=100%>
<img src="https://github.com/titu1994/MLSTM-FCN/blob/master/images/MLSTM-FCN-scores2.png?raw=true" height=100% width=100%>
<img src="https://github.com/titu1994/MLSTM-FCN/blob/master/images/MLSTM-FCN-scores3.png?raw=true" height=100% width=100%>

# Citation
