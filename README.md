A scalar-valued autograd engine and a neural net library on top of it with PyTorch-like API
The DAG only operates over scalar values. we divide up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes.

Training a neural net:
The notebook demo.ipynb provides a full demo of training an 2-layer neural network (MLP) binary classifier. This is achieved by initializing a neural net from micrograd.nn module, implementing a simple svm "max-margin" binary classification loss and using SGD for optimization. As shown in the notebook, using a 2-layer neural net with two 16-node hidden layers we achieve the following decision boundary on the moon dataset:

For added convenience, the notebook trace_graph.ipynb produces graphviz visualizations. 
