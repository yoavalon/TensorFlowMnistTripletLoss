import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

#Check if GPU loaded
a= tf.test.gpu_device_name()
print(a)


#Hyper-parameters

batchSize = 128            #128
epochs = 100                #1000
margin = 0.1               #0.1
learningRate = 0.001       #0.001
displaySteps = 10           #100
testEpochs = 5             #2

class TripletNet:
    
    def __init__(self):
      
        self.images = tf.placeholder(tf.float32, [None,784])
        self.y = tf.placeholder(tf.float32, [None])     
             
        with tf.variable_scope("triplet") as scope:
            self.embeddings = self.network(tf.reshape(self.images,[batchSize,28,28,1]))         
            scope.reuse_variables() 
        
        #self.loss = self.TripletLoss(self.y, self.embeddings)        
        self.loss = self.batch_all_triplet_loss(self.y, self.embeddings, margin)
        
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

    def _pairwise_distances(self, embeddings, squared=False):
    
        dot_product = tf.matmul(embeddings, tf.transpose(embeddings))    
        square_norm = tf.diag_part(dot_product)
    
        distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
        distances = tf.maximum(distances, 0.0)

        if not squared:        
          mask = tf.to_float(tf.equal(distances, 0.0))
          distances = distances + mask * 1e-16
          distances = tf.sqrt(distances)        
          distances = distances * (1.0 - mask)

        return distances      

    def _get_triplet_mask(self, labels):
    
        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)
        i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
        i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
        j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

        distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)
    
        label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        i_equal_j = tf.expand_dims(label_equal, 2)
        i_equal_k = tf.expand_dims(label_equal, 1)

        valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))
    
        mask = tf.logical_and(distinct_indices, valid_labels)

        return mask      
      
    def batch_all_triplet_loss(self, labels, embeddings, margin, squared=False):
    
        
        pairwise_dist = self._pairwise_distances(embeddings, squared=squared)
        
        anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)        
        anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)        
    
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        mask = self._get_triplet_mask(labels)
        mask = tf.to_float(mask)
        
        #problem here:
        #tf.reshape(embeddings[indexes[0],:],[2,1])     
        #Dimensions must be equal, but are 10 and 128 for 'Mul_3' (op: 'Mul') with input shapes: [?,?,?,10], [128,128,128].
        
        #triplet_loss = tf.multiply(mask, triplet_loss)
        triplet_loss = tf.matmul(mask, triplet_loss)
        
        triplet_loss = tf.maximum(triplet_loss, 0.0)
        
        valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        num_valid_triplets = tf.reduce_sum(mask)
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
        
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
        
        return triplet_loss
      
      
   
#Main _________________________________________________________________________________________________________

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

g = tf.Graph() 
sess = tf.InteractiveSession(graph=g)

model = TripletNet();
optimizer = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(model.loss)

tf.initialize_all_variables().run()

lossList = []
accList = []

for step in range(epochs):  
  
    batch_images, batch_y = mnist.train.next_batch(batchSize)
    
    _, loss_v, emb = sess.run([optimizer, model.loss, model.embeddings], feed_dict={ model.images: batch_images, model.y: batch_y})
    lossList.append(loss_v)
    
    if step % displaySteps == 0:
        print ('step %3d:  loss: %.6f' % (step, loss_v)) 
        #print('kann', kann)
        
    colors=["red", "blue", "green", "orange", "black", "yellow", "pink", "cyan", "magenta", "white"]
    #if step== epochs-1 :        
    if step % displaySteps == 0:
      for i in range(batchSize) :                  
          plt.plot(emb[i,0], emb[i,1], "o", c = colors[batch_y[i]])                  
          plt.text(emb[i,0] * (1 + 0.01), emb[i,1] * (1 + 0.01) , batch_y[i], fontsize=12)
        
      plt.title('Embeddings for epoch ' + str(step))
      plt.ylabel('y')
      plt.xlabel('x')
      plt.show()      
      
# plot Loss Graph
plt.plot(lossList)
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

