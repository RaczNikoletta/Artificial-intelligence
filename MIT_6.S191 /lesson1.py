# Tensorflow
class MyDenseLayer (tf.keras.layers.Layer):
  def __init__(self, input_dim, output_dim):
    super(MyDenseLayer, self) __init__()

    #Initialize weights and bias

    self.W = self.add_weight([input_dim, output_dim])
    self.b = self.add_weight([1, output_dim])

def call(self, inputs):
  #forward propagate the input
  z = tf.matmul(inputs, self.W) + self.b

  #Feed through
  output = tf.math.sigmoid(z)
  return output

#Pytorch
class MyDenseLayer(nn.Module):
  def __init__(self,input_dim, output_dim):
    super(MyDenseLayer, self).__init__()
    #Initialize weights and bias
    self.W = nn.Parameter(torch.randn(input_dim, 
                                      output_dim, requires_grad=True)
    self.b = nn.Parameter(torch.randn(1, output_dim, requires_grad=True)

    def forward(self, inputs):
      #Forward propagate the inputs
      z = torch.matmul(inputs, self.W) + self.b
      # Feed through a non-linear activation
      output = torch.sigmoid(z)
      return output

#Multi output perceptron

#tensorflow
import tensorflow as tf

model = tf.keras.Sequential ([
  tf.keras.layers.Dense(n),
  tf.keras.layers.Dense(2)
])

#pytorch
from torch import nn

model = nn.Sequential (
  nn.Linear(m, n),
  nn.ReLU(),
  nn.Linear(n, 2)
)





  
  
  
