import tensorflow as tf
import numpy as np
import os

class RBM:    
    def __init__(self, numHidden, numVisible, numSteps = 1, pcd = False):
        
        self._numHidden = numHidden
        self._numVisible = numVisible
        self._numSteps = numSteps
        self._pcd = pcd
        
        self.graph = tf.Graph()         
        with self.graph.as_default(): 
            self._define_parameters()        
            self._define_computational_graph()
            self._define_learning_parameters()        
            self._saver = tf.train.Saver()
            self._init = tf.global_variables_initializer()                      
   
        self.sess = tf.Session( graph = self.graph )        
        self.sess.run(self._init)
        
    #############################################################################################        
    def _define_parameters(self):        
        
        low = -4 * np.sqrt(6.0/(self._numVisible + self._numHidden))
        high = -low
        
        self._w = tf.Variable(tf.random_uniform(shape=[self._numVisible, self._numHidden], minval=low, maxval=high), dtype=tf.float32, name = 'weights')
        self._b = tf.Variable(tf.zeros([self._numVisible]), name = 'vBias')
        self._c = tf.Variable(tf.zeros([self._numHidden]), name = 'vHidden')
        
    #############################################################################################    
    def _define_computational_graph(self):
        
        self._visible = tf.placeholder(tf.float32, [None, self._numVisible], name='visible')        
        self._visible_random_nums = tf.random_uniform(tf.shape(self._visible), name = 'vRand')
        self._hidden_random_nums  = tf.random_uniform([tf.shape(self._visible)[0], self._numHidden] , name='hRand')                
        self._hidden, self._bhidden = self._sample_hidden_from_visible(self._visible)                
        
        def body(a, bhidden):
            a = tf.add(a, 1)
            bNewVisible = self._sample_visible_from_hidden(bhidden)[-1]
            bhidden = self._sample_hidden_from_visible(bNewVisible)[-1]
            return a, bhidden            
        condition = lambda a, b: tf.less(a, self._numSteps-1)
        
        if(self._pcd):
            self._flip_index = 0
            self._steady = tf.Variable( tf.random_uniform([100, self._numVisible]), name= 'steadyState') 
            bhiddenFromSteady = self._sample_hidden_from_visible(self._steady)[1]            
            bhidden = tf.while_loop(condition, body, [0, bhiddenFromSteady])[1]   
        else:
            bhidden = tf.while_loop(condition, body, [0, self._bhidden])[1]                            
        
        self._new_visible, self._b_new_visible = self._sample_visible_from_hidden(bhidden) 
        self._new_hidden = tf.nn.sigmoid(tf.matmul(self._b_new_visible, self._w) + self._c, name='newHiddenProb')        
            
        if(self._pcd):
            self._save_chain_state = self._steady.assign(self._b_new_visible)  
            self._cost = self._get_pseudo_likelihood_cost()
        else:
            self._cost = -tf.reduce_mean(tf.mul(self._visible, tf.log(self._new_visible)))
    
    #############################################################################################    
    def _define_learning_parameters(self):    
        
        dw = ((tf.matmul(self._visible, self._hidden, transpose_a=True) - tf.matmul(self._b_new_visible, self._new_hidden, transpose_a=True)) / tf.cast(tf.shape(self._visible)[0], dtype= tf.float32))
        db =  tf.reduce_mean(self._visible - self._b_new_visible, axis = 0)
        dc =  tf.reduce_mean(self._hidden - self._new_hidden, axis = 0)
       
        velw = tf.Variable(tf.zeros([self._numVisible, self._numHidden], dtype= tf.float32), name = 'velWeights')
        velb = tf.Variable(tf.zeros([self._numVisible], dtype= tf.float32), name = 'velVBias')
        velc = tf.Variable(tf.zeros([self._numHidden], dtype= tf.float32), name = 'velHBias')
        
        self.learningRate = tf.constant(0.01)
        self.momentum = tf.constant(0.9)
         
        updateVelW = velw.assign(self.momentum * velw + self.learningRate * dw)
        updateVelB = velb.assign(self.momentum * velb + self.learningRate * db)
        updateVelC = velc.assign(self.momentum * velc + self.learningRate * dc)
        
        with tf.control_dependencies([updateVelW, updateVelB, updateVelC]):
            self._updateW = self._w.assign_add(velw)
            self._updateB = self._b.assign_add(velb)
            self._updateC = self._c.assign_add(velc)
        
    #############################################################################################
    def _free_energy(self, v_sample):        
        
        wx_c = tf.matmul(v_sample, self._w) + self._c
        vbias_term = tf.matmul(v_sample, tf.reshape(self._b, [self._numVisible, 1]))
        hidden_term = tf.reduce_sum(tf.log(1 + tf.exp(wx_c)), axis=1)
        return -hidden_term - vbias_term
    
    #############################################################################################    
    def _sample_prob(self, probs, rand):
        
        return tf.nn.relu(tf.sign(probs - rand))
        
    #############################################################################################
    def _get_pseudo_likelihood_cost(self):         
        
        self._flipIndex = tf.placeholder(tf.int32, shape=())        
        xi = self._visible       
        fe_xi = self._free_energy(xi)
               
        part1 = tf.slice(xi, [0, 0], [-1 , self._flipIndex])
        part2 = tf.slice(xi, [0, self._flipIndex], [-1, 1])
        part3 = tf.slice(xi, [0, self._flipIndex+1], [-1, -1])   
           
        xi_flip =  tf.concat(values = [part1, 1 - part2, part3], concat_dim =1)        
        fe_xi_flip = self._free_energy(xi_flip)        
        cost = tf.reduce_mean(self._numVisible * tf.log(tf.nn.sigmoid(fe_xi_flip - fe_xi))) 
        return cost
        
    #############################################################################################    
    def train_for_batch(self, inputData): 
        
        if(self._pcd):
            result = self.sess.run([self._new_visible, self._updateW, self._updateB, self._updateC, self._save_chain_state, self._cost], feed_dict = {self._visible : inputData, self._flipIndex:self._flip_index})
            self._flip_index = (self._flip_index + 1) % self._numVisible        
        else:   
            result = self.sess.run([self._new_visible, self._updateW, self._updateB, self._updateC, self._cost], feed_dict = {self._visible : inputData})           
        return result[-1]

    #############################################################################################    
    def save_model(self, saveFolder = r'.\model', name = 'model'):  
        
        if(not os.path.exists(saveFolder)):
            os.mkdir(saveFolder)        
        path = os.path.join(saveFolder, name)            
        self._saver.save(self.sess, path)
        
    #############################################################################################    
    def load_model(self, saveFolder = r'.\model', name = 'model'):
        
        self.sess.run(self._init)        
        path = os.path.join(saveFolder, name)            
        self._saver.restore(self.sess, path) 
        
    #############################################################################################    
    def __del__(self):
        self.close()
    
    #############################################################################################    
    def close(self):
        self.sess.close()         
        
    #############################################################################################                                                  
    def _sample_hidden_from_visible(self, visible):
        
        hidden = tf.nn.sigmoid(tf.matmul(visible, self._w) + self._c, name='hiddenProb')
        bhidden =  self._sample_prob(hidden, self._hidden_random_nums)
        return hidden, bhidden
        
    #############################################################################################    
    def _sample_visible_from_hidden(self, hidden):
        
        newVisible = tf.nn.sigmoid(tf.matmul(hidden, self._w, transpose_b=True) + self._b, name='newVisibleProb')           
        bNewVisible = self._sample_prob(newVisible, self._visible_random_nums)        
        return newVisible, bNewVisible   
        
    #############################################################################################           
    def generate_sample(self, visibleData, numberOfSteps):    
        
        def body(a, visible):            
            a = tf.add(a, 1)  
            bhidden = self._sample_hidden_from_visible(visible)[-1]
            bNewVisible = self._sample_visible_from_hidden(bhidden)[-1] 
            return a, bNewVisible            
        condition = lambda a, b: tf.less(a, numberOfSteps)        
        
        r = tf.while_loop(condition, body, [0, self._visible])
        visibleData = self.sess.run(r, feed_dict = {self._visible : visibleData})  
        return visibleData[1]

    #############################################################################################           
    def get_parameters(self):
        
        [w, b, c] = self.sess.run([self._w, self._b, self._c])
        return {'w': w,
                'b': b,
                'c': c                
                }

