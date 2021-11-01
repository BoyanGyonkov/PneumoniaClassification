import tensorflow as tf
import numpy as np

class Dense(tf.Module):
    def __init__(self,out_features):
        super().__init__(name='dense')
        self.b = tf.Variable(tf.random.normal([out_features]), dtype=tf.float32, trainable = True)
        self.out_features = out_features
        self.is_built = False
        self.next_layer = 0
	
    def addLayer(self,layer):
        self.next_layer = layer

    def nextLayer(self):
        return self.next_layer

    def __call__(self, x,training = True):
        if not self.is_built:
            self.w = tf.Variable(tf.random.normal([x.shape[-1], self.out_features]), dtype=tf.float32, trainable = True)
            self.is_built = True
        return tf.matmul(x,self.w) + self.b

class Conv2D(tf.Module):
    def __init__(self, kernel_size ,in_channels, out_channels, strides_size=1, padding='VALID'):
        super().__init__(name='conv')
        
        self.padding = padding
        self.strides = [1,strides_size, strides_size,1]
        self.conv = tf.Variable(tf.random.normal([kernel_size, kernel_size, in_channels, out_channels]),dtype = tf.float32, trainable=True)
        self.next_layer = 0
	
    def addLayer(self,layer):
        self.next_layer = layer

    def nextLayer(self):
        return self.next_layer

    def __call__(self, x, training = True):
        return tf.nn.conv2d(x, self.conv, self.strides, self.padding)

class MaxPool2D(tf.Module):
    def __init__(self, pool_size, padding='VALID'):
        super().__init__(name = 'max2d')
        self.ksize = pool_size
        self.padding = padding
        self.strides = [1, pool_size, pool_size, 1]
        self.next_layer = 0
	
    def addLayer(self,layer):
        self.next_layer = layer

    def nextLayer(self):
        return self.next_layer

    def __call__(self,x, training = True):
        return tf.nn.max_pool2d(x, ksize=self.ksize, padding = self.padding, strides = self.strides)

class BatchNormalization(tf.Module):
    def __init__(self,in_features,for_conv = False):
        super().__init__(name = 'batchNorm')
        self.decay = 0.99
        self.population_mean = tf.Variable(tf.zeros(in_features) , trainable = False)
        self.population_variance = tf.Variable(tf.ones(in_features), trainable = False)
        self.scale = tf.Variable(tf.ones(in_features), trainable = True )
        self.beta = tf.Variable(tf.zeros(in_features), trainable = True )
        self.for_conv = for_conv
        
        self.next_layer = 0
	
    def addLayer(self,layer):
        self.next_layer = layer

    def nextLayer(self):
        return self.next_layer

    def __call__(self, x, training = True):
        if training:
            if not self.for_conv:
                batch_mean , batch_variance = tf.nn.moments(x, [0], keepdims= False)
            else:
                batch_mean , batch_variance = tf.nn.moments(x, [0 ,1 ,2], keepdims= False)
            self.population_mean = self.population_mean * self.decay + batch_mean* (1-self.decay)
            self.population_variance = self.population_variance * self.decay + batch_variance * (1-self.decay)
            
            return tf.nn.batch_normalization(x, batch_mean, batch_variance, self.beta, self.scale , variance_epsilon=1e-3)
        else:
            return tf.nn.batch_normalization(x, self.population_mean, self.population_variance, self.beta, self.scale , variance_epsilon=1e-3)    
        
class Dropout(tf.Module):
    def __init__(self, dropout_rate):
        super().__init__(name = 'dropOut')
        self.dropout_rate = dropout_rate
        self.next_layer = 0
	
    def addLayer(self,layer):
        self.next_layer = layer

    def nextLayer(self):
        return self.next_layer


    def __call__(self, x, training = True):
        if training:
            return tf.nn.dropout(x, rate= self.dropout_rate)
        else:
            return x

class Flatten(tf.Module):
    def __init__(self):
        super().__init__(name = 'flatten')
        self.next_layer = 0

    def addLayer(self,layer):
        self.next_layer = layer

    def nextLayer(self):
        return self.next_layer

    def __call__(self, x, training = True):
        return tf.reshape(x, [tf.shape(x)[0] , -1])

class ActivationFunction(tf.Module):
    def __init__(self, activation_function):
        super().__init__(name = 'activFunc')
        self.func = activation_function
        self.next_layer = 0

    def addLayer(self,layer):
        self.next_layer = layer

    def nextLayer(self):
        return self.next_layer

    def __call__(self,x, training = True):
        return self.func(x)
    
class SequentialModel(tf.Module):
    def __init__(self, activation_function = tf.nn.relu,out_function = 'softmax', name = None):
        super().__init__(name = name)
        self.activation_func = activation_function
        self.out_function = out_function
        if out_function == 'softmax':
            self.loss_function = tf.nn.sparse_softmax_cross_entropy_with_logits
        if out_function == 'sigmoid':
            self.loss_function = tf.nn.sigmoid_cross_entropy_with_logits
        self.layer_first = 0
        self.layers_list = []
        

    def __call__(self, x , training=True):
        inp = x

        curr_layer = self.layer_first
        while curr_layer != 0:
            inp = curr_layer(inp, training)
            curr_layer = curr_layer.nextLayer()

        return inp
        
            
    def add(self, layer):
        self.layers_list.append(layer)
        
    def build(self):
        self.layer_first = self.layers_list[0]
        curr_layer = self.layer_first
        
        l_iter = iter(range(len(self.layers_list)))
        
        for i in l_iter:
            if self.layers_list[i].name == 'dense' or self.layers_list[i].name == 'conv':
                #if a dense or convolutional layer is followed by batch normalization -> include activation function after batch normalization
                if i != len(self.layers_list) -1:
                    if(self.layers_list[i+1].name != 'batchNorm'):
                        curr_layer.addLayer(self.layers_list[i])
                        curr_layer = curr_layer.nextLayer()
                        curr_layer.addLayer(ActivationFunction(self.activation_func))
                        curr_layer = curr_layer.nextLayer()
     
                    else:
                        curr_layer.addLayer(self.layers_list[i])
                        curr_layer = curr_layer.nextLayer()
                        curr_layer.addLayer(self.layers_list[i+1])
                        curr_layer = curr_layer.nextLayer()
                        curr_layer.addLayer(ActivationFunction(self.activation_func))
                        curr_layer = curr_layer.nextLayer()
                        next(l_iter)
                else:
                    curr_layer.addLayer(self.layers_list[i])
                    curr_layer = curr_layer.nextLayer()
            else:
                curr_layer.addLayer(self.layers_list[i])
                curr_layer = curr_layer.nextLayer()

        self.layers_list = None
                    
    def printLayers(self):
        curr_layer = self.layer_first
        while curr_layer != 0:
            print(curr_layer)
            curr_layer = curr_layer.nextLayer()


    def prediction(self, x):
        y_pred = self(x,training=False)
        if(self.out_function == 'softmax'):
            return np.argmax(y_pred.numpy(), axis=1)
        elif(self.out_function == 'sigmoid'):
            return np.round(tf.math.sigmoid(y_pred).numpy())

    def loss(self, ground_y , logits):
        if self.out_function=='sigmoid':
            logits = tf.reshape(logits, [tf.shape(logits)[0]])
            ground_y = tf.cast(ground_y, tf.float32)
        return tf.reduce_mean(self.loss_function(labels = ground_y , logits = logits))


    def train(self,x,y , learning_rate):
        with tf.GradientTape() as t:
  
            current_loss = self.loss(y, self(x))

            grads = t.gradient(current_loss, self.trainable_variables)
            opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
            opt.apply_gradients(zip(grads,self.trainable_variables))
            
            return current_loss

    def eval(self, dataset, batch_size):
        batches = dataset.batch(batch_size)
        loss = 0
        length = 0
        accuracy = 0
        for batch in batches:
            logits = self(tf.cast(batch[0], tf.float32), training = False)
            loss += self.loss(tf.cast(batch[1],tf.int32), logits)
        
            if(self.out_function == 'softmax'):
                y_pred = tf.math.argmax(logits.numpy(), axis =1)
                y_test =tf.cast(batch[1] , tf.int64)
            elif(self.out_function == 'sigmoid'):
                y_pred =  tf.round(tf.math.sigmoid(logits))
                y_test = tf.reshape(batch[1], [len(batch[1]), 1])
                y_test = tf.cast(y_test, tf.float32)
                
            accuracy += np.sum(y_pred == y_test)
            length += len(batch[1])

        accuracy = accuracy/length
        print(accuracy)

        return (loss, accuracy)
    

    def fit(self, dataset_train ,epochs , lr , dataset_val, batch_size):
        self.lossHistory = []
        self.evalHistory = []

        batches = dataset_train.batch(batch_size)
        
        for i in  range(epochs):
            loss_ep = 0

            for batch in batches:
    
                loss_ep += self.train(tf.cast(batch[0],tf.float32) ,tf.cast(batch[1], tf.int32) , lr)
    
            self.lossHistory.append(loss_ep/len(dataset_train))
            self.evalHistory.append(self.eval(dataset_val, batch_size))
            print("Epoch: ", i+1 )

        return self.evalHistory
