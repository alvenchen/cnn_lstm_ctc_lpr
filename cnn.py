
import tensorflow as tf

# Layer params:   Filts K  Padding  Name     BatchNorm?
layer_params = [ [  64, 3, 'valid', 'conv1', False], 
                 [  64, 3, 'same',  'conv2', True],  # pool
                 [ 128, 3, 'same',  'conv3', False], 
                 [ 128, 3, 'same',  'conv4', True],  # hpool
                 [ 256, 3, 'same',  'conv5', False],
                 [ 256, 3, 'same',  'conv6', True],  # hpool
                 [ 512, 3, 'same',  'conv7', False], 
                 [ 512, 3, 'same',  'conv8', True] ] # hpool 3

rnn_size = 2**9    # Dimensionality of all RNN elements' hidden layers
dropout_rate = 0.5 # For RNN layers (currently not used--uncomment below)

def conv_layer( x, params, training ):
    """Build a convolutional layer using entry from layer_params)"""

    batch_norm = params[4] # Boolean

    if batch_norm:
        activation = None
    else:
        activation = tf.nn.relu

    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer( value=0.0 )

    top = tf.layers.conv2d( x, 
                            filters=params[0],
                            kernel_size=params[1],
                            padding=params[2],
                            activation=activation,
                            kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer,
                            name=params[3] )
    if batch_norm:
        top = norm_layer( top, training, params[3]+'/batch_norm' )
        top = tf.nn.relu( top, name=params[3]+'/relu' )

    return top


def pool_layer( x, wpool, padding, name ):
    """Short function to build a pooling layer with less syntax"""
    top = tf.layers.max_pooling2d( x, 
                                   2, 
                                   [2, wpool], 
                                   padding=padding, 
                                   name=name )
    return top


def norm_layer( x, training, name):
    """Short function to build a batch normalization layer with less syntax"""
    top = tf.layers.batch_normalization( x, 
                                         axis=3, # channels last                                         
                                         training=training,
                                         name=name )
    return top


def convnet_layers( inputs, widths, mode ):
    """
    Build convolutional network layers attached to the given input tensor
    """

    training = (mode == "train")
    
    with tf.variable_scope( "convnet" ): # h,w
        
        #print(inputs.shape)
        x = conv_layer( inputs, layer_params[0], training )        
        #print(x.shape)
        x = conv_layer( x, layer_params[1], training )        
        #print(x.shape)
        x = pool_layer( x, 2, 'valid', 'pool2' )
        #print(x.shape)
        x = conv_layer( x, layer_params[2], training )    
        x = conv_layer( x, layer_params[3], training )
        #print(x.shape)
        x = pool_layer( x, 2, 'valid', 'pool4' )
        #print(x.shape)
        x = conv_layer( x, layer_params[4], training )        
        x = conv_layer( x, layer_params[5], training )
        #print(x.shape)
        x = pool_layer( x, 2, 'valid', 'pool6')        
        #print(x.shape)
        x = conv_layer( x, layer_params[6], training )        
        x = conv_layer( x, layer_params[7], training )
        
        x = tf.layers.max_pooling2d( x, [2, 1], [2, 1], 
                                         padding='valid', 
                                         name='pool8' ) 

        #print(x.shape)

        # squeeze row dim
        x = tf.squeeze( x, axis=1, name='features' )

        #print(x.shape)

        sequence_length = get_sequence_lengths( widths )                

        return x, sequence_length

def get_sequence_lengths( widths ):    
    """Tensor calculating output sequence length from original image widths"""    
    seq_len = (widths - 2) / 8
    return seq_len

