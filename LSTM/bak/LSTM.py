import tensorflow as tf

class lstm:
    def __init__(self,num_inputs,num_hiddens,num_outputs):
        def _one(shape):
            return tf.Variable(tf.random.normal(shape=shape,stddev=0.01,mean=0.0,dtype=tf.float32))
        def _three():
            return (_one((num_inputs,num_hiddens)),_one((num_hiddens,num_hiddens)),tf.Variable(tf.zeros(num_hiddens),dtype=tf.float32))

        self.W_xi,self.W_hi,self.b_i=_three()#输入门参数
        self.W_xf,self.W_hf,self.b_f=_three()#遗忘门参数
        self.W_xo,self.W_ho,self.b_o=_three()#输出门参数

        self.W_xc,self.W_hc,self.b_c=_three()#候选记忆细胞参数
        
        #self.W_hq=_one((num_hiddens,num_outputs))#输出参数
        #self.b_q=tf.Variable(tf.zeros(num_outputs),dtype=tf.float32)
        
    
    def __call__(self,data,state):
        (H,C)=state
       
        #output_i=[]
        output_i_H=[]
        #print(data.shape,"lstm")
        for i in range(data.shape[0]):
            #output_x=[]
            output_x_H=[]
            for X in data[i,:,:]:
                #print(data[i,:,:],"tf.reduce_sum(X)")
                if(tf.reduce_sum(X)==0.0):
                    break
                X=tf.reshape(X,[-1,self.W_xi.shape[0]])
                I=tf.math.sigmoid(tf.matmul(X,self.W_xi)+tf.matmul(H,self.W_hi)+self.b_i)
                F=tf.math.sigmoid(tf.matmul(X,self.W_xf)+tf.matmul(H,self.W_hf)+self.b_f)
                O=tf.math.sigmoid(tf.matmul(X,self.W_xo)+tf.matmul(H,self.W_ho)+self.b_o)

                C_tilda=tf.math.tanh(tf.matmul(X,self.W_xc)+tf.matmul(H,self.W_hc)+self.b_c)#获得候选记忆细胞

                C=F*C+I*C_tilda#更新记忆细胞
                H=O*tf.math.tanh(C)#获得隐藏状态
                #Y=tf.matmul(H,self.W_hq)+self.b_q
                #print(Y.shape)
                #output_x.append(Y)
                output_x_H.append(H)
                #print()
            #tf.concat(output_x,1)
            #output_i.append(output_x)   
            output_i_H.append(output_x_H)
        #return output_i,(H,C),output_i_H
        return output_i_H

    def get_params(self):
        return [self.W_xi, self.W_hi, self.b_i, self.W_xf, self.W_hf, self.b_f, self.W_xo, self.W_ho, self.b_o, self.W_xc, self.W_hc, self.b_c]#, self.W_hq, self.b_q]
    def init_lstm_state(self,batch_size,num_hiddens):
        return tf.zeros(shape=(batch_size,num_hiddens)),tf.zeros(shape=(batch_size,num_hiddens))
