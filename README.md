# Description

The library provides a set of classes - NeuralNetwork, LayerBase, ReluLayer, SigmoidLayer, which have high cohesion and designed to serve as an implementation of logistic regression based **DNN** image classifier and provide means to train model on datasets and predict classes of supplied image examples.

# Example of usage

`layers` - provides ability to set a custom configuration of neural network;
`learning_rate` - this parameter gives control how fast gradient descent algorithm corrects weights and bias;
`iterations` - how many iterations are performed to optimize the model;
`train_features` - matrix of shape `(flattened_image, num_of_images)`, represents the training data;
`train_classes` - vector of shape `(1, num_of_examples)`, represents classification of training data to anchor the mode.

```
layers = [ReluLayer(2, sigmoid), SigmoidLayer(1, relu)]
nn = NeuralNetwork(layers, learning_rate=0.09, iterations=1)
nn.train(train_features, train_classes)
predictions = nn.predict(train_features)
```
