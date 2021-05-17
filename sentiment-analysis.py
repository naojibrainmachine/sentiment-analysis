import tensorflow as tf
import random
import numpy as np
from LOADDATA import build_vocab,get_corpus_indices,data_format,get_data
from LSTM.LSTM import lstm
from DENSE.DENSE import dense
from EMBEDDING.EMBEDDING import embedding

random.seed(1)

class stentimentAnalysis:
    def __init__(self,batch_size,lr,num_input,num_outputs,num_hiddens,vocabulary_size,embedding_size):
        '''
        定义所有的模块，包括lstm、embedding、dense
        '''
        self.batch_size=batch_size
        self.lr=lr
        self.vocab_size=vocabulary_size
        self.lstm=lstm(num_input,num_hiddens,num_outputs)
        self.embedding=embedding(self.vocab_size,embedding_size)

        #全连接层的参数
        self.dense1=dense(num_hiddens,num_outputs)
        self.dense2=dense(num_outputs,1)#1为固定参数
        self.dense3=dense(2*num_outputs,256)#256可以随意换成其他的
        self.dense4=dense(256,2)#2是这里情感分析只有正面和负面之分

        #随机梯度下降优化器
        self.opt=tf.keras.optimizers.SGD(learning_rate=lr)
        
        
    def __call__(self,data,state):
        '''
        data:为one_hot编码数据
        '''
        inputs=self.embedding.embedding_lookup(data)
        outputs,state=self.lstm(inputs,state)
        (H,_)=state
        outputs1=tf.concat(outputs,0)

        #重新获取最后一个lstm cell的输出
        outputs3=self.dense1(H)

        #全连接层
        outputs5=self.dense2(outputs1)
        
        output_tran5=[]
        output_tran1=[]
        #把outputs1和outputs5对应的第一维度和第二维度互换
        for i in range(outputs5.shape[1]): 
            output_5=[]
            output_1=[]
            for j in range(outputs5.shape[0]):
                output_5.append(outputs5[j][i][:])
                output_1.append(outputs1[j][i][:])
            output_tran5.append(output_5)
            output_tran1.append(output_1)
        outputs5=tf.reshape(output_tran5,[outputs5.shape[1],outputs5.shape[0],outputs5.shape[2]])
        outputs1=tf.reshape(output_tran1,[outputs1.shape[1],outputs1.shape[0],outputs1.shape[2]])

        #对权重outputs5进行归一化
        outputs6=tf.nn.softmax(outputs5,1)
        outputs7=outputs1*outputs6

        #把所有lstm cell的输出按权重outputs6加到一起
        outputs8=tf.reduce_sum(outputs7,1)

        #把lstm cell最后的输出outputs3，与所有lstm cell输出相关的outputs8，拼接在一起
        outputs9=tf.concat([outputs8,outputs3],1)

        #全连接层
        outputs=self.dense3(outputs9)
        #全连接层
        outputs=self.dense4(outputs)
        
        return tf.nn.softmax(outputs)
        
    def loss(self,logist,label):
        '''
        交叉熵损失函数
        '''
        return -1*tf.reduce_mean(tf.multiply(tf.math.log(logist+1e-10),label))

    def get_params(self):
        '''
        返回模型所有参数
        '''
        params=[]
        params.extend(self.lstm.get_params())
        params.extend(self.embedding.get_params())
        params.extend(self.dense1.get_params())
        params.extend(self.dense2.get_params())
        params.extend(self.dense3.get_params())
        params.extend(self.dense4.get_params())
        
        return params
    def update_params(self,grads,params):
        self.opt.apply_gradients(grads_and_vars=zip(grads,params))

def return_accuracy(temp_predict,temp_batch_label,batch_size):
    '''
    计算准确率
    '''
    rowMaxSoft=np.argmax(temp_predict, axis=1)+1
    rowMax=np.argmax(temp_batch_label, axis=1)+1
    rowMaxSoft=rowMaxSoft.reshape([1,-1])
    
    rowMax=rowMax.reshape([1,-1])
    nonO=rowMaxSoft-rowMax
    exist = (nonO != 0) * 1.0
    factor = np.ones([nonO.shape[1],1])
    res = np.dot(exist, factor)
    accuracy=(float(batch_size)-res[0][0])/float(batch_size)
    
    return accuracy

def train(model,params,vocabulary,labels,chars_to_idx,label_chars_to_idx,batch_size):
    '''
    训练函数
    '''
    acc=[]
    iter_data=get_data(data=vocabulary,labels=labels,chars_to_idx=chars_to_idx,label_chars_to_idx=label_chars_to_idx,batch_size=batch_size)
    outputs=[]
    Ys=[]
    for x,y in iter_data:
        state_lstm=model.lstm.init_lstm_state(len(y),num_hiddens)#初始化lstm的C和H          
        X,Y=data_format(x,y)#格式化数据         
        X,Y=tf.concat(X,0),tf.concat(Y,0)#把格式化后的组合到一个tensor里       
        X=tf.one_hot(X,model.vocab_size)  #one_hot编码          
        Y=tf.one_hot(Y,len(label_idx_to_chars))#one_hot编码
        Y=tf.reshape(Y,[Y.shape[0],Y.shape[-1]])#转化成2维度
        with tf.GradientTape() as tape:
            tape.watch(params)
            output=model(X,state_lstm)
            loss=model.loss(output,Y)#获取交叉熵结果
            print("loss %f"%loss.numpy())
            grads=tape.gradient(loss, params)#求梯度
            grads,globalNorm=tf.clip_by_global_norm(grads, clipNorm)#梯度裁剪
            model.update_params(grads,params)#参数更新
        Ys.append(Y)#记录所有标签
        outputs.append(output)#记录所有输出    
    outputs=tf.concat(outputs,0)
    Ys=tf.concat(Ys,0)
    
    #把准确率存到当前目录
    filepath="acc.txt"
    flie=open(filepath,"a+")
    flie.write(str(tf.math.reduce_mean(return_accuracy(outputs,Ys,Ys.shape[0])).numpy())+"\n")
    flie.close()

    '''
    for k in range(len(params)):
        filepath="p"+str(k)+".txt"
        np.savetxt(filepath,(params[k].numpy()).reshape(1,-1))
    '''

def predict(model,params,test_vovab,test_labels,chars_to_idx,label_chars_to_idx,batch_size):
    '''
    预测函数
    '''
    test_acc=[]
    iter_data=get_data(data=test_vovab,labels=test_labels,chars_to_idx=chars_to_idx,label_chars_to_idx=label_chars_to_idx,batch_size=batch_size)
    outputs=[]
    Ys=[]
    for x,y in iter_data:
        state_lstm=model.lstm.init_lstm_state(len(y),num_hiddens)#初始化lstm的C和H
        X,Y=data_format(x,y)#格式化数据
        X,Y=tf.concat(X,0),tf.concat(Y,0)#把格式化后的组合到一个tensor里
        X=tf.one_hot(X,model.vocab_size)#one_hot编码
        Y=tf.one_hot(Y,len(label_idx_to_chars))#one_hot编码
        Y=tf.reshape(Y,[Y.shape[0],Y.shape[-1]])#转化成2维度
        output=model(X,state_lstm)#
        Ys.append(Y)
        outputs.append(output)
    outputs=tf.concat(outputs,0)
    Ys=tf.concat(Ys,0)
    accT=return_accuracy(outputs,Ys,Ys.shape[0])

    #把准确率存到当前目录
    test_acc.append(accT)
    filepath="testacc.txt"
    flie=open(filepath,"a+")
    flie.write(str(tf.math.reduce_mean(test_acc).numpy())+"\n")
    flie.close()
        
if __name__ == "__main__":
    
    embedding_size=20
    num_hiddens=128
    clipNorm=1.0
    batch_size=1
    
    vocabulary,labels ,test_labels,test_vovab,chars_to_idx,idx_to_chars,vocab_size,label_idx_to_chars,label_chars_to_idx,label_size=build_vocab('data//data_single.csv')
    sta=stentimentAnalysis(batch_size,1e-3,num_input=embedding_size,num_outputs=embedding_size,num_hiddens=num_hiddens,vocabulary_size=vocab_size,embedding_size=embedding_size)
    params=sta.get_params()
    epochs=30#轮次

    '''
    isContinue=True
    if isContinue==True:
        for k in range(len(params)):
            filepath="p"+str(k)+".txt"
            params[k].assign((np.loadtxt(filepath,dtype=np.float32)).reshape(params[k].shape))
    '''
    #训练
    for i in range(epochs):
        train(sta,params,vocabulary,labels,chars_to_idx,label_chars_to_idx,batch_size)
    #测试
    predict(sta,params,test_vovab,test_labels,chars_to_idx,label_chars_to_idx,batch_size)
        
    
