import tensorflow as tf

class RBM:
    
    def __init__(self, numHidden, numVisible, numSteps = 1):
        
        self._numHidden = numHidden
        self._numVisible = numVisible
        self._numSteps = numSteps
        
        self.w = tf.Variable(tf.truncated_normal([self._numVisible, self._numHidden], stddev=0.01), name = 'weights')
        self.b = tf.Variable(tf.zeros([self._numVisible]), name = 'vBias')
        self.c = tf.Variable(tf.zeros([self._numHidden]), name = 'vHidden')        
        
        self.saver = tf.train.Saver()    
        self.savePath = r'.\model'
        
        self.DefineComputationalGraph()
        self.DefineLearningParameters()          
        
    #############################################################################################    
    def sample_prob(self, probs, rand):
        return tf.nn.relu(tf.sign(probs - rand))
   
    #############################################################################################    
    def DefineComputationalGraph(self):          
        
        self.visible = tf.placeholder(tf.float32, [None, self._numVisible], name='visible')
        
        self.visibleRandomNums = tf.random_uniform(tf.shape(self.visible), name = 'vRand')
        self.hiddenRandomNums  = tf.random_uniform([tf.shape(self.visible)[0], self._numHidden] , name='hRand')     
        
        self.hidden, self.bhidden = self.Sample_Hidden_From_Visible(self.visible)                
       
        def body(a, bhidden):            
            a = tf.add(a, 1)            
            bNewVisible = self.Sample_Visible_From_Hidden(bhidden)[-1]
            bhidden = self.Sample_Hidden_From_Visible(bNewVisible)[-1]     
            return a, bhidden            
        condition = lambda a, b: tf.less(a, self._numSteps-1)
        
        bhidden = tf.while_loop(condition, body, [0, self.bhidden])[1]        
        
        self.newVisible, self.bNewVisible = self.Sample_Visible_From_Hidden(bhidden)
        self.newHidden = tf.nn.sigmoid(tf.matmul(self.bNewVisible, self.w) + self.c, name='newHiddenProb')
        
        self.cost = -tf.reduce_mean(tf.mul(self.visible, tf.log(self.newVisible)))

    #############################################################################################    
    def DefineLearningParameters(self):
        
        dw = ((tf.matmul(self.visible, self.hidden, transpose_a=True) - tf.matmul(self.bNewVisible, self.newHidden, transpose_a=True)) / tf.cast(tf.shape(self.visible)[0], dtype= tf.float32))
        db =  tf.reduce_mean(self.visible - self.bNewVisible, axis = 0)
        dc =  tf.reduce_mean(self.hidden - self.newHidden, axis = 0)
       
        velw = tf.Variable(tf.zeros([self._numVisible, self._numHidden], dtype= tf.float32), name = 'velWeights')
        velb = tf.Variable(tf.zeros([self._numVisible], dtype= tf.float32), name = 'velVBias')
        velc = tf.Variable(tf.zeros([self._numHidden], dtype= tf.float32), name = 'velHBias')
        
        self.learningRate = tf.constant(0.05)
        self.momentum = tf.constant(0.9)
         
        self.updateVelW = velw.assign(self.momentum * velw + self.learningRate * dw)
        self.updateVelB = velb.assign(self.momentum * velb + self.learningRate * db)
        self.updateVelC = velc.assign(self.momentum * velc + self.learningRate * dc)
        
        self.updateW = self.w.assign_add(velw)
        self.updateB = self.b.assign_add(velb)
        self.updateC = self.c.assign_add(velc)

    #############################################################################################    
    def SetSession(self, sess):
        self.sess = sess
    
    #############################################################################################    
    def Initilize(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
    #############################################################################################    
    def TrainForBatch(self, inputData):
        result = self.sess.run([self.newVisible, self.updateVelW, self.updateVelW, self.updateVelW, self.cost], feed_dict = {self.visible : inputData})
        self.sess.run([self.updateW, self.updateB, self.updateC])        
        return result[-1]

    #############################################################################################    
    def SaveTheModel(self):
        self.saver.save(self.sess, self.savePath)
        
    #############################################################################################    
    def LoadModel(self):
        self.Initilize()
        self.saver.restore(self.sess, self.savePath)
             
    #############################################################################################                                                  
    def Sample_Hidden_From_Visible(self, visible):
        hidden = tf.nn.sigmoid(tf.matmul(visible, self.w) + self.c, name='hiddenProb')
        bhidden =  self.sample_prob(hidden, self.hiddenRandomNums)
        return hidden, bhidden
        
    #############################################################################################    
    def Sample_Visible_From_Hidden(self, hidden):
        newVisible = tf.nn.sigmoid(tf.matmul(hidden, self.w, transpose_b=True) + self.b, name='newVisibleProb')           
        bNewVisible = self.sample_prob(newVisible, self.visibleRandomNums)        
        return newVisible, bNewVisible   
        
    #############################################################################################           
    def GenerateSample(self, visibleData, numberOfSteps): 
        
        def body(a, visible):            
            a = tf.add(a, 1)  
            bhidden = self.Sample_Hidden_From_Visible(visible)[-1]
            bNewVisible = self.Sample_Visible_From_Hidden(bhidden)[-1] 
            return a, bNewVisible
            
        condition = lambda a, b: tf.less(a, numberOfSteps)
        
        r = tf.while_loop(condition, body, [0, self.visible])
        visibleData = self.sess.run(r, feed_dict = {self.visible : visibleData})  
        return visibleData[1]
        
        
    