# Tensorflow
class MyDenseLayer (tf.keras.layers.Layer):
  def __init__(self, input_dim, output_dim):
    super(MyDenseLayer, self) __init__()

#Initialize weights and bias

self.W = self.add_weight([input_dime, output_dim])
self.b = self.add_weight([1, output_dime])

def call(self, inputs):
  #forward propagate the input
  z = tf.matmul(inputs, self.W) + self.b

#Feed through
output = tf.math.sigmoid(z)
return output


  
  
  
