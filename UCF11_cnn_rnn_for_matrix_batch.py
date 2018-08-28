from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import pdb
import math
import time
from sklearn.model_selection import KFold

from tensorflow.python.ops.distributions.util import fill_triangular
from matrixcell import MatrixRNNCell,CNNRNNCell,Chol_de

from readdata import read_data


def get_a_cell():
    return MatrixRNNCell(alpha = a , batch_size = batch_size , matrix_size = matrix_size , eps = eps)



batch_size = 40
height = 120
width = 160
in_channel = 3
out_channel = 7
tot_time_points = 50
class_num = 11
matrix_size = out_channel+1
epoch_num = 1000
depth = 5

# CNN_num_layer = 2
CNN_kernel_shape = [[7,7,5],[7,7,out_channel]]
CNN_num_layer = len(CNN_kernel_shape)


reduced_spatial_dim = height * width / (4**CNN_num_layer) #4800 # height * width / (4**CNN_num_layer)
beta = 0.3

eps = 1e-10
n = matrix_size
a = [0.01, 0.25, 0.5, 0.9, 0.99]
a_num = len(a)

sample_rate = 3

lr = 0.9
decay_steps = 1000
decay_rate = 0.99

matrix_length = tot_time_points
global_steps = tf.Variable(0,trainable = False)
learning_rate = tf.train.exponential_decay(lr, global_step = global_steps, decay_steps = decay_steps, decay_rate = decay_rate)
add_global = global_steps.assign_add(1)



X = tf.placeholder(np.float32,shape = (batch_size,matrix_length,height,width,in_channel)) 
y = tf.placeholder(np.float32,shape = (batch_size,class_num)) 

keep_prob = tf.placeholder(tf.float32)

W2_1 = tf.Variable(tf.random_normal([n*(n+1)//2, class_num],stddev=np.sqrt(2./(class_num*n*(n+1)//2))))
b2_1 = tf.Variable(tf.random_normal([1, class_num],stddev=np.sqrt(2./class_num)))


initMt = tf.placeholder(np.float32,[batch_size,a_num*n*n])

Mt_1 = initMt

tf.keras.backend.set_learning_phase(True)


CNNRNNcell = [CNNRNNCell(alpha = a , num_layer = CNN_num_layer, kernel_shape = CNN_kernel_shape , batch_size = batch_size ,
                        matrix_size = matrix_size ,in_channel= in_channel, out_channel=out_channel ,
                        reduced_spatial_dim=reduced_spatial_dim , beta = beta, keep_prob = keep_prob , eps = eps)]

for i in range(depth):
    CNNRNNcell.append(get_a_cell())


cells = tf.nn.rnn_cell.MultiRNNCell(CNNRNNcell)




initial_state=tuple([initMt for _ in range(depth+1)])
# initial_state = cells.zero_state(batch_size, np.float32)



outputs, state = tf.nn.dynamic_rnn(cells,X,initial_state=initial_state , dtype = np.float32)

outputs = tf.slice(outputs,[0,matrix_length-1,0],[-1,1,-1])
outputs = tf.reshape(outputs,[batch_size,n,n])
# pdb.set_trace()

output_series =  Chol_de ( outputs, n,batch_size )


    
# output_series = tf.slice(output_series,[(matrix_length-1)*n*(n+1)//2,0],[n*(n+1)//2,-1])
output_series = tf.keras.layers.BatchNormalization()(output_series)

output_series = tf.nn.dropout(output_series, keep_prob)

#output_series_1 = tf.nn.relu( tf.add( tf.matmul ( output_series, W2_1 ), b2_1 ) )
predict_label = tf.add( tf.matmul ( output_series, W2_1 ), b2_1 ) 
#predict_label = tf.nn.softmax( tf.clip_by_value(tf.add( tf.matmul ( output_series_1, W2_2 ), b2_2 ), -50, 50) )
#predict_label = tf.nn.softmax( tf.nn.tanh(tf.add( tf.matmul ( output_series_1, W2_2 ), b2_2 )) )

#loss = tf.reduce_mean( tf.reduce_sum( -y * tf.log(predict_label+eps),1)) 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
     logits = predict_label,
     labels = y
))

correct_prediction = tf.equal(tf.argmax(predict_label, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#loss = tf.reduce_mean(tf.reduce_sum( tf.pow( y - predict_label , 2 ),1 ))

with tf.control_dependencies([add_global]):
    #opt = tf.train.AdagradOptimizer(learning_rate)
    #opt = tf.train.RMSPropOptimizer(learning_rate,momentum=0.9)
    opt = tf.train.AdadeltaOptimizer(learning_rate)
    train_step = opt.minimize(loss)
#grad = tf.gradients(loss,[Weights_rnn[0]['WR_root'],Weights_rnn[0]['Wt_root'],Weights_rnn[0]['Ws_root'], Weights_rnn[0]['Wphi_root'],Weights_cnn['W1'],Weights_cnn['W2']])

'''
load data and label here
and make them randomly and group into batch
'''

batch_num = 40

init_state = np.reshape( np.tile(np.eye(n)*1e-5,[batch_size,a_num,1,1]) , [batch_size,a_num*n*n] )
#init_state =  np.tile(1e-5 * np.random.uniform(low=0.0,high=1.0,size=(n,n)),[batch_size,a_num,1,1]) 
loss_p = 0

batch_num_idx = range(batch_num)
k_fold = KFold(n_splits=10)
final_acc_fold = np.zeros((10,1))
#CL = Chol_de(current_X,n)
#CC = Chol_com(CL,n,eps)

data = []
label = []

for idx in range(batch_num):
    print (idx)
    data_batch_in,label_batch_in = read_data(idx,'../../TTRNN/UCF11_updated_mpg/processed_data/',matrix_length,sample_rate)
    data.append(data_batch_in)
    label.append(label_batch_in)


with tf.Session() as sess:
    final_acc = 0.
    co = 0
    for tr_indices, ts_indices in k_fold.split(batch_num_idx):
        sess.run(tf.global_variables_initializer())
        print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        #start_time = time.time()
        for epoch in range(epoch_num):
            start_time = time.time()
            train_acc = 0.
            train_loss = 0.
            for batch_idx in tr_indices:
                data_batch_in = data[batch_idx]
                label_batch_in = label[batch_idx]
                # data_batch_in,label_batch_in = read_data(batch_idx,'../../TTRNN/UCF11_updated_mpg/processed_data/',matrix_length,sample_rate)
                #
                #print batch_idx
                #CL_,CC_ ,current_X_= sess.run([CL,CC,current_X],
                #_, loss_ , predict_label_,Weights_,Rt_,Yt_,tt_,Phit_,Mt_,St_,yt_,ot_,Bias_,W2_,b2_,grad_= sess.run([train_step,loss,predict_label,Weights,Rt,Yt,tt,Phit,Mt,St,yt,ot,Bias,W2,b2,grad],
                _, loss_,  predict_, W2_1_,y_, acc_ = sess.run([train_step,loss,predict_label,W2_1,y,accuracy],
                #Yt_,Rt_,tt_,Phit_= sess.run([Yt,Rt,tt,Phit],
                         feed_dict={
                               X:data_batch_in,
                               y:label_batch_in,
                               initMt:init_state,
                               keep_prob:0.75,
                                })
                #
                # if not batch_idx%100:
                #
                if math.isnan(loss_):
                    pdb.set_trace()
                else:
                    train_acc = train_acc + acc_
                    train_loss = train_loss + loss_
                    # print(loss_,acc_,epoch)
            train_acc = train_acc / len(tr_indices)
            train_loss = train_loss/len(tr_indices)
            print ('Train Accuracy is : ' , train_acc , ' in Epoch : ' , epoch)
            print ('Train Loss is : ' , train_loss)
                # print predict_label_
            print ('Time per epoch : ' , time.time()-start_time)
            
            test_acc = 0
            for batch_idx in ts_indices:
                data_batch_in = data[batch_idx]
                label_batch_in = label[batch_idx]
                # data_batch_in,label_batch_in = read_data(batch_idx,'../../TTRNN/UCF11_updated_mpg/processed_data/',matrix_length,sample_rate)
                loss_, acc_ = sess.run([loss,accuracy],
                    #Yt_,Rt_,tt_,Phit_= sess.run([Yt,Rt,tt,Phit],
                             feed_dict={
                                   X:data_batch_in,
                                   y:label_batch_in,
                                   initMt:init_state,
                                   keep_prob:1.,
                                    })
                test_acc = test_acc + acc_
            test_acc = test_acc / len(ts_indices)
            print ('Test Accuracy is : ' , test_acc)
            print (' ')

        final_acc_fold[co] = 0.
        for batch_idx in ts_indices:
            data_batch_in = data[batch_idx]
            label_batch_in = label[batch_idx]
            # data_batch_in,label_batch_in = read_data(batch_idx,'../../TTRNN/UCF11_updated_mpg/processed_data/',matrix_length,sample_rate)
            loss_, acc_ = sess.run([loss,accuracy],
                #Yt_,Rt_,tt_,Phit_= sess.run([Yt,Rt,tt,Phit],
                         feed_dict={
                               X:data_batch_in,
                               y:label_batch_in,
                               initMt:init_state,
                               keep_prob:1.,
                                })
            final_acc_fold[co] = final_acc_fold[co] + 1.0*acc_/len(ts_indices)
            print(loss_,acc_)
        print('After kth fold' , final_acc_fold[co])
        final_acc = final_acc + final_acc_fold[co]*1.0/10
        co += 1
    print(final_acc)
    np.save('UCF11_5sample20length.npy',final_acc_fold)
