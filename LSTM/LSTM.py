import tensorflow as tf

class lstm:
    def __init__(self,num_inputs,num_hiddens,num_outputs):
        '''
        定义lstm参数
        num_inputs：如果是以分布式词向量（embedding）作为输入，则num_inputs为分布式词向量大小。如果是以one_hot作为输入，则num_inputs为词库大小
        num_hiddens：隐藏状态和记忆细胞长度
        num_outputs：输出Y的列长度
        '''
        def _one(shape):
            return tf.Variable(tf.random.normal(shape=shape,stddev=0.01,mean=0.0,dtype=tf.float32))
        def _three():
            return (_one((num_inputs,num_hiddens)),_one((num_hiddens,num_hiddens)),tf.Variable(tf.zeros(num_hiddens),dtype=tf.float32))

        self.W_xi,self.W_hi,self.b_i=_three()#输入门参数
        self.W_xf,self.W_hf,self.b_f=_three()#遗忘门参数
        self.W_xo,self.W_ho,self.b_o=_three()#输出门参数
        self.W_xc,self.W_hc,self.b_c=_three()#候选记忆细胞参数
        
        self.W_hq=_one((num_hiddens,num_outputs))#最终输出Y的参数
        self.b_q=tf.Variable(tf.zeros(num_outputs),dtype=tf.float32)#最终输出Y的参数
        
    
    def __call__(self,data,state):
        '''
        这里是lstm核心部分，完成对输入门、遗忘门、输出门、记忆细胞、隐藏状态的更新，并输出结果Y
        '''
        (H,C)=state
        output_i=[]
        #这个循环是外层循环，是遍历一个批次所有语句的循环
        for i in range(data.shape[0]):
            output_x=[]
            #这是遍历语句所有词汇数据的循环
            for X in data[i,:,:]:
                if(tf.reduce_sum(X)==0.0):#遇到-1填充的位置，跳出内层循环
                    break
                X=tf.reshape(X,[-1,self.W_xi.shape[0]])
                I=tf.math.sigmoid(tf.matmul(X,self.W_xi)+tf.matmul(H,self.W_hi)+self.b_i)#输入门
                F=tf.math.sigmoid(tf.matmul(X,self.W_xf)+tf.matmul(H,self.W_hf)+self.b_f)#遗忘门
                O=tf.math.sigmoid(tf.matmul(X,self.W_xo)+tf.matmul(H,self.W_ho)+self.b_o)#输出门

                C_tilda=tf.math.tanh(tf.matmul(X,self.W_xc)+tf.matmul(H,self.W_hc)+self.b_c)#获得候选记忆细胞

                C=F*C+I*C_tilda#更新记忆细胞
                H=O*tf.math.tanh(C)#获得隐藏状态
                Y=tf.matmul(H,self.W_hq)+self.b_q
                
                output_x.append(Y)
            output_i.append(output_x)   
            
        return output_i,(H,C)
        

    def get_params(self):
        '''
        获取lstm的参数
        '''
        return [self.W_xi, self.W_hi, self.b_i, self.W_xf, self.W_hf, self.b_f, self.W_xo, self.W_ho, self.b_o, self.W_xc, self.W_hc, self.b_c, self.W_hq, self.b_q]
    def init_lstm_state(self,batch_size,num_hiddens):
        '''
        初始化记忆细胞和隐藏状态
        '''
        return tf.zeros(shape=(batch_size,num_hiddens)),tf.zeros(shape=(batch_size,num_hiddens))
