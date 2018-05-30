import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

#Check if GPU loaded
a= tf.test.gpu_device_name()
print(a)


#Hyper-parameters

batchSize = 128            #128
epochs = 1000              #1000
margin = 0.1               #0.1
learningRate = 0.001       #0.001
displaySteps = 50           #100
testEpochs = 5              #2
KNN_k = 5                  #10

class TripletNet:
    
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, [None, 784])        
        self.x2 = tf.placeholder(tf.float32, [None, 784])
        self.x3 = tf.placeholder(tf.float32, [None, 784]) 
        self.y_ = tf.placeholder(tf.float32, [None,10],)
                
        with tf.variable_scope("triplet") as scope:
            self.output1 = self.network(tf.reshape(self.x1,[batchSize,28,28,1])) 
            scope.reuse_variables() 
            self.output2 = self.network(tf.reshape(self.x2,[batchSize,28,28,1])) 
            scope.reuse_variables() 
            self.output3 = self.network(tf.reshape(self.x3,[batchSize,28,28,1]))                         
                
        self.loss = self.TripletLoss() 
        self.Accuracy = self.GetAccuracy()  

        
    def network(self, input, reuse = tf.AUTO_REUSE) :
        
        if (reuse):
          tf.get_variable_scope().reuse_variables()          
        
        with tf.name_scope("network") :          
          with tf.variable_scope("conv1") as scope:
            net = tf.contrib.layers.conv2d(input, 32, [7, 7],biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)            
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=True,  scope='conv1', reuse= reuse)
            
          with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, 64, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=True,  scope='conv2', reuse= reuse)

          with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=True,  scope='conv3', reuse= reuse)

          with tf.variable_scope("conv4") as scope:
            net = tf.contrib.layers.conv2d(net, 256, [1, 1], activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=True,  scope='conv4', reuse= reuse)

          with tf.variable_scope("conv5") as scope:
            net = tf.contrib.layers.conv2d(net, 2, [1, 1], activation_fn=None, padding='SAME',
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')            
            net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=True,  scope='conv5', reuse= reuse)
          
        net = tf.contrib.layers.flatten(net)        #embedding
                
        return net

    def TripletLoss(self) :
      
        anchor_feature = self.output3
        positive_feature = self.output2
        negative_feature = self.output1          
        
        with tf.name_scope("triplet_loss"):               
          pos_dis = tf.reduce_mean(tf.square(tf.subtract(anchor_feature, positive_feature)),1, keepdims=True)
          neg_dis = tf.reduce_mean(tf.square(tf.subtract(anchor_feature, negative_feature)),1, keepdims=True)
          
          res = tf.maximum(0., pos_dis + margin - neg_dis) 
          loss = tf.reduce_mean(res)          
        
        return loss

    def Predict(self, input, k = KNN_k):              #Prediction using KNN
      
        neg_one = tf.constant(-1.0, dtype=tf.float32)    
        distances =  tf.reduce_sum(tf.abs(tf.subtract(self.output3, input)), 1)
        neg_distances = tf.multiply(distances, neg_one)    
        vals, indx = tf.nn.top_k(neg_distances, k)    
        prediction = tf.gather(self.y_, indx)
        
        res = tf.reduce_sum(prediction,0)        
        index = tf.argmax(res)
        
        return index
    
    def GetAccuracy(self) :  #TripletAccuracy
      
        anchor_feature = self.output3
        positive_feature = self.output2
        negative_feature = self.output1    
      
        pos_dis = tf.reduce_mean(tf.square(tf.subtract(anchor_feature, positive_feature)),1, keepdims=True)
        neg_dis = tf.reduce_mean(tf.square(tf.subtract(anchor_feature, negative_feature)),1, keepdims=True)
                
        correct = tf.less_equal(pos_dis +margin, neg_dis)
        acc = tf.reduce_sum(tf.cast(correct, tf.float32))/batchSize
                
        return acc       


def GetDifferentIdentityImage(mnist, lab) :
  
  label = lab
  ran = -1
  
  label2 = lab
  ran2 = -1
    
  while(label == lab) :
    ran = np.random.randint(0,mnist.train.labels.shape[0], 1)  
    label = mnist.train.labels[ran]    
  
    while(label2 != label and ran2!=ran) :
      ran2 = np.random.randint(0,mnist.train.labels.shape[0], 1)  
      label2 = mnist.train.labels[ran2]    
  
  #print("     ", lab, label, label2)
    
  return ran, ran2, label

# Create random Triplet and assign binary Label      
def GetTriplet(mnist) :
  
  ran = np.random.randint(0,mnist.train.labels.shape[0], 1)  
  lab = mnist.train.labels[ran]  
  
  sames = GetDifferentIdentityImage(mnist, lab)
    
  #print(mnist.train.labels[ran],mnist.train.labels[sames[0]],mnist.train.labels[sames[1]])
    
  return np.array([ran, sames[0], sames[1], sames[2]]) 
  
#Creates batch of shape (128,4) where as 128 is batch size and 4 stands for a triplet plus binary label
def CreateTripletBatch(mnist) :  
  Triplet_Set = []
  for i in range(batchSize) : 
    Triplet_Set.append(GetTriplet(mnist))
  
  return np.array(Triplet_Set)

#Fetch image data from index
def FetchImages(mnist, indexes, training) : 
  
  imgList = []  
  for i in indexes : 
    if(training) :
      imgList.append(mnist.train.images[i])
    else:
      imgList.append(mnist.test.images[i])
    
  res = np.asarray(imgList)  
  
  return np.reshape(res, (batchSize,784))

#Main

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

g = tf.Graph() #reset graph
sess = tf.InteractiveSession(graph=g)

model = TripletNet();

optimizer = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(model.loss)

tf.initialize_all_variables().run()

lossList = []
accList = []

for step in range(epochs):  
  
    TripletBatch = CreateTripletBatch(mnist)
    
    batch_x1 = FetchImages(mnist, TripletBatch[:,0], True)
    batch_x2 = FetchImages(mnist, TripletBatch[:,1], True)
    batch_x3 = FetchImages(mnist, TripletBatch[:,2], True)
    batch_y = np.reshape(TripletBatch[:,3], (batchSize,))
    
    y_list = []
    
    for i in range(batchSize) :      
      y_list.append(np.eye(10)[batch_y[i]])           
       
    _, loss_v, Accuracy, clas, la = sess.run([optimizer, model.loss, model.Accuracy, model.output3, model.y_], feed_dict={
                        model.x1: batch_x1,
                        model.x2: batch_x2,
                        model.x3: batch_x3,
                        model.y_: y_list,                        
                        })    
    
        
    lossList.append(loss_v)
    accList.append(Accuracy)
    if step % displaySteps == 0:
        print ('step %3d:  loss: %.6f   triplet-accuracy: %.3f ' % (step, loss_v, Accuracy)) 
     
    colors=["red", "blue", "green", "orange", "black", "yellow", "pink", "cyan", "magenta", "white"]
    if step== epochs-1 :        
      for i in range(batchSize) :                  
          plt.plot(clas[i,0], clas[i,1], "o", c = colors[np.argmax(la[i])])                  
          plt.text(clas[i,0] * (1 + 0.01), clas[i,1] * (1 + 0.01) , np.argmax(la[i]), fontsize=12)
     
       
        
      plt.title('Embeddings (for last Batch)')
      plt.ylabel('y')
      plt.xlabel('x')
      plt.show()      
      
# plot Loss Graph
plt.plot(lossList)
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# plot Loss Graph
plt.plot(accList)
plt.title('Training Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

