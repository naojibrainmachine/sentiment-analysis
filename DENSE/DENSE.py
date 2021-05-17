#Dense
import tensorflow as tf
import numpy as np
class dense:
    def __init__(self,inputs_size,output_size):
        '''
        构建一层全连接层
        '''
        self.w=tf.Variable(tf.random.truncated_normal([inputs_size,output_size],stddev=np.sqrt(2.0 / (inputs_size+output_size))))
        self.b=tf.Variable(tf.zeros(output_size))
        
    def __call__(self,array):
        '''
        array:列的长度必须为output_size
        '''
        output=tf.matmul(array,self.w)+self.b
        
        return output

    def get_params(self):
        return [self.w,self.b]
