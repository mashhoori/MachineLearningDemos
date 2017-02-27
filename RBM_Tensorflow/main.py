
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import RBM
import matplotlib.pyplot as plt


def convert_to_binary(a):
    a[a >= 0.5] = 1
    a[a <  0.5] = 0
    return a
    
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
bTrainImages = convert_to_binary(mnist.train.images)

tf.reset_default_graph()

costTotal = 0
rbm = RBM.RBM(500, 28*28, 15, True)
for i in range(5000):
     batch_x, batch_y = mnist.train.next_batch(100)
     cost = rbm.train_for_batch(batch_x)         
     costTotal += cost
  
     if(i % (784) == 0):
         print(i)            
         print(costTotal/784)
         costTotal = 0
         sample = rbm.generate_sample(batch_x, 15)
         for j in range(2):
             plt.figure()
             plt.imshow(sample[j].reshape([28, 28]))

             
params = rbm.get_parameters()             
rbm.save_model()
rbm.close()        

###########################################################################

batch_x, batch_y = mnist.validation.next_batch(100)
tf.reset_default_graph()
rbm2 = RBM.RBM(500, 28*28, 200, True)
rbm2.load_model()
    
sample = rbm2.generate_sample(batch_x, 5000)
    
for j in range(2):
    plt.figure()
    plt.imshow(sample[j].reshape([28, 28]))
rbm2.close()

