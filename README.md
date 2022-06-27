# simple-neural-network
Simple Neural Network I made based on "Make Your Own Neural Network" Book by Tariq Rashid

## Neural Network Skeleton Code
> Source Code: *neural_network.ipynb*
- initialisation - to set the number of input, hidden and output nodes
- train - refine the weights after being given a training set example to learn from
- query - give an answer from the output nodes after being given an input

## Predicting Number from MNIST Dataset
> Source Code: *mnist.ipynb*
- Import Libraries
- Read Dataset
- Plot The Handwriting
- Pre Processing Input Data
  - Scale the data to range [0.01, 0.99]
- Constructs The Target Matrix
- Training Neural Network
- Testing the Network
- Score our Neural Network Model
- Training using the full dataset
- Some Improvements
  - Tweaking the Learning Rate
  - Doing Multiple Runs (epoch)
  - Change Netwrok Shape (Change the number of hidden layer nodes)
- Final Result
  - My Model scores **0.9737**
- Save Model to .pkl

## Predicting Number from Your Own Handwriting
> Source Code: *handwriting.ipynb*
- Import Libraries
- Load Trained Model .pkl
- Load Own Image Dataset
- Plot the Image Dataset
- Predict the Image Dataset
  - The neural network **recognised all of the images** we created, including the deliberately damaged “3”. Only the **“6” with added noise failed**.
- Additional Testing (Custom Inputs)