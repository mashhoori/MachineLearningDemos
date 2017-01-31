
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import RBM
import matplotlib.pyplot as plt


def ConvertToBinary(a):
    a[a >= 0.5] = 1
    a[a <  0.5] = 0
    return a
    
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
bTrainImages = ConvertToBinary(mnist.train.images)    

tf.reset_default_graph()

rbm = RBM.RBM(500, 28*28, 1)
with tf.Session() as sess:    
    rbm.SetSession(sess)       
    
    rbm.Initilize()  
    for i in range(10000):
         batch_x, batch_y = mnist.train.next_batch(100)
         cost = rbm.TrainForBatch(batch_x)         
         print(cost)
  
         if(i % 1000 == 0):
             print(i)            
             sample = rbm.GenerateSample(batch_x, 1)
             for j in range(2):
                 plt.figure()
                 plt.imshow(sample[j].reshape([28, 28]))  
             
    rbm.SaveTheModel()

    
    
batch_x, batch_y = mnist.validation.next_batch(100)

tf.reset_default_graph()    
rbm2 = RBM.RBM(500, 28*28)
with tf.Session() as sess:    
    rbm2.SetSession(sess)       
    rbm2.LoadModel()         
    
    sample = rbm2.GenerateSample(batch_x, 5000)
    
    for j in range(2):
        plt.figure()
        plt.imshow(sample[j].reshape([28, 28]))  