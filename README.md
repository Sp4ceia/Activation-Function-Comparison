This project compares the overall accuracy of 4 TensorFlow based neural nets trained on the  FashionMNIST dataset. These 4 neural networks have the same number of neurons (164) and the same epoch value (5) (Epoch value is the number of passes made by the AI model through the entire dataset). The goal of this project is to produce consistent accuracy values about the results of each activation function by minimizing data noise. In order to minimize data noise, I chose the FashionMNIST data set. 
Compared functions: 
1) ReLU
2) Leaky ReLU
3) Sigmoid
4) Hyperbolic Tangent
Training Setup:
  Optimizer = Adam
  Loss Function =  Cross Categorical Entropy
  Batch Size = 32 (Which is the default size for tensorflow neural nets)
Mean accuracy of all models after 4 test runs:
  ReLU == 86.575%
  Sigmoid == 86.500%
  tanh == 86.125%
  LeakyReLU == 85.975%
 





