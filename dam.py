# Author: Han FU

"""Dual attention module"""
import tensorflow as tf

def channel_attention(input_feature, name, ratio=8):
  """channel attention brach"""
  kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
  bias_initializer = tf.constant_initializer(value=0.0)
  
  with tf.variable_scope(name):
    
    channel = input_feature.get_shape()[-1]
    avg_pool = tf.reduce_mean(input_feature, axis=[1,2], keep_dims=True)
    avg_pool = tf.layers.dense(inputs=avg_pool,
                                 units=channel//ratio,
                                 activation=tf.nn.relu,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name='fc_0',
                                 reuse=None)   
    avg_pool = tf.layers.dense(inputs=avg_pool,
                                 units=channel,                             
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name='fc_1',
                                 reuse=None)    
    max_pool = tf.reduce_max(input_feature, axis=[1,2], keep_dims=True)    
    max_pool = tf.layers.dense(inputs=max_pool,
                                 units=channel//ratio,
                                 activation=tf.nn.relu,
                                 name='fc_0',
                                 reuse=True)   
    max_pool = tf.layers.dense(inputs=max_pool,
                                 units=channel,                             
                                 name='fc_1',
                                 reuse=True)  
    scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')
    #scale = tf.sigmoid(avg_pool, 'sigmoid') 
  return input_feature * scale

def spatial_attention(input_feature, name):
  """spatial attention brach"""
  kernel_size = 3
  kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
  with tf.variable_scope(name):

    avg_pool = tf.reduce_mean(input_feature, axis=[3], keep_dims=True)
    max_pool = tf.reduce_max(input_feature, axis=[3], keep_dims=True)
    
    concat = tf.concat([avg_pool,max_pool], 3)
  
    concat = tf.layers.conv2d(concat,
                              filters=1,
                              kernel_size=[kernel_size,kernel_size],
                              strides=[1,1],
                              padding="same",
                              activation=None,
                              kernel_initializer=kernel_initializer,
                              use_bias=False,
                              name='conv')
    concat = tf.sigmoid(concat, 'sigmoid')
   
  return input_feature * concat

def dam_block(input_feature, name, ratio=8):

  with tf.variable_scope(name):
    output_channel=input_feature.get_shape().as_list()[3]
    # 方差缩放初始化
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    inputs=tf.layers.conv2d(input_feature,
                            filters=output_channel,
                            kernel_size=[3,3],
                            strides=[1,1],
                            padding="same",
                            activation=tf.nn.relu,
                            kernel_initializer=kernel_initializer,
                            use_bias=False,
                            name='input_conv1')
    inputs=tf.layers.conv2d(inputs,
                            filters=output_channel,
                            kernel_size=[3,3],
                            strides=[1,1],
                            padding="same",
                            activation=None,
                            kernel_initializer=kernel_initializer,
                            use_bias=False,
                            name='input_conv2')
    CA_feature = channel_attention(inputs, 'ch_at', ratio)
    SA_feature = spatial_attention(inputs, 'sp_at')
    
    concat_CS = tf.concat([CA_feature,SA_feature], 3) 
    attention_feature=tf.layers.conv2d(concat_CS,
                              filters=output_channel,
                              kernel_size=[3,3],
                              strides=[1,1],
                              padding="same",
                              activation=None,
                              kernel_initializer=kernel_initializer,
                              use_bias=False,
                              name='CA_PA_conv')
    feature=tf.add(input_feature,attention_feature)     
    print ("Call for DAM")
  return feature