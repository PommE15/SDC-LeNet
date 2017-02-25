# Implement the LeNet-5 neural network architecture
# Reference: https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab.ipynb

from tensorflow.contrib.layers import flatten

def conv2d(x, W, b, strides=1, padding='VALID'):
    conv = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    conv = tf.nn.bias_add(conv, b)
    return tf.nn.relu(conv)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='VALID')

def fullyConnected(x, W, b):
    fc = tf.add(tf.matmul(x, W), b)
    fc = tf.nn.relu(fc)
    fc = tf.nn.dropout(fc, 0.5)
    return fc
    
def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    weights = {
        'wc1': tf.Variable(tf.truncated_normal((5, 5, 1, 6), mu, sigma)),
        'wc2': tf.Variable(tf.truncated_normal((5, 5, 6, 16), mu, sigma)),
        'wf3': tf.Variable(tf.truncated_normal((400, 120), mu, sigma)),
        'wf4': tf.Variable(tf.truncated_normal((120, 84), mu, sigma)),
        'wf5': tf.Variable(tf.truncated_normal((84, 10), mu, sigma))
    }
    biases = {
        'bc1': tf.Variable(tf.zeros(6)),
        'bc2': tf.Variable(tf.zeros(16)),
        'bf3': tf.Variable(tf.zeros(120)),
        'bf4': tf.Variable(tf.zeros(84)),
        'bf5': tf.Variable(tf.zeros(10))
    }
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    # TODO: Activation.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], )    
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = maxpool2d(conv1, k=2)
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    # TODO: Activation.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])   
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = maxpool2d(conv2, k=2)  
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    conv2 = flatten(conv2)   
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    # TODO: Activation.
    fc3 = fullyConnected(conv2, weights['wf3'], biases['bf3']) 
    

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    # TODO: Activation.
    fc4 = fullyConnected(fc3, weights['wf4'], biases['bf4']) 
    
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.add(tf.matmul(fc4, weights['wf5']), biases['bf5']) 

    return logits
