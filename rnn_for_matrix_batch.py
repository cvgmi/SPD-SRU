import tensorflow as tf
import numpy as np
import random
import pdb

from tensorflow.python.ops.distributions.util import fill_triangular
'''
All the data matrix should be batch_size * n * n
All the matrix W should be n * n
'''

def f(x):
    return x
    return tf.nn.relu(x)

def myeigvalue(x):
    return list(np.linalg.eig(x))[0]

def myeigvector(x):
    return list(np.linalg.eig(x))[1]



def Eig(x):
    #return [tf.py_func(myeigvalue,[x],tf.float32),tf.py_func(myeigvector,[x],tf.float32)]
    return tf.self_adjoint_eig(x)

def NUS(W_root,A):
    '''
    input the matrix A and matrix W_root
    A is the one to be compute
    W_root is the square-root of W
    which will make W positive
    '''
    A = tf.add (A , 1e-5 * tf.diag(tf.random_uniform([n])) )
    S,U = Eig(A)
    W = tf.pow(W_root,2)
    # W = tf.nn.relu(W_root)
    #W = tf.tile ( tf.expand_dims(W,0),[A.shape[0],1,1] )
    S = tf.multiply(S,W)
    Sigma = tf.matrix_diag(S)
    #NUSresult = tf.matmul(U,W)                            # U * W
    NUSresult = tf.matmul(U,Sigma)                # U * W * Sigma
    NUSresult = tf.matmul(NUSresult,U,transpose_b = True) # U * W * Sigma * U.T
    return NUSresult

def MatrixExp(B,l,n):
    '''
    input a matrix B, and the total length to be calculated, n is the size of B
    output the somehow exp(B) = I + B + B^2 / 2! + B^3 / 3! + ... + B^l / l!
    '''
    
    Result = tf.eye(n)
    # temp_result = tf.eye(n)
    # factorial = 1.
    # for i in range(1,l+1):
    #     temp_result = tf.matmul(temp_result,B)
    #     factorial = factorial * (i)
    #     Result = tf.add(Result,temp_result/factorial)
    # return tf.matrix_inverse ( tf.matrix_inverse ( Result ) )

    return tf.matmul( tf.matrix_inverse(tf.subtract(Result , B)) , tf.add( Result , B) )

def Translation(A,B,n):

    '''

    input the matrix A and vector B

    change B to be SO 

    like [[0 ,  1, 2]

          [-1,  0, 3]

          [-2, -3, 0]]

    return B * A * B.T

    '''
    power_matrix = 5
    B = tf.reshape(B,[1,-1])

    #lower_triangel = fill_triangular(B)
    line_B = [tf.zeros([1,n])]
    for i in range (n-1):
        temp_line = tf.concat([ tf.slice(B,[0,i],[1,i+1]) , tf.zeros([1,n-i-1]) ] ,axis = 1)
        line_B.append(temp_line)

    lower_triangel = tf.concat(line_B,axis = 0)

    B_matrix = tf.subtract(lower_triangel, tf.transpose(lower_triangel))
    
    B_matrix = MatrixExp(B_matrix,power_matrix,n)

    B_matrix = tf.tile ( tf.expand_dims(B_matrix,0),[A.shape[0],1,1] )

 

    Tresult = tf.matmul(B_matrix,A)                              # B * A

    Tresult = tf.matmul(Tresult,tf.transpose(B_matrix,[0,2,1]))      # B * A * B.T
    return Tresult

'''
Do not delete the codes under. The problem is the self_adjoint_eig which can only calculate the symetric matrix.
However, the before_root is not symetric.
Also, we want to use the tf.SVD. But till now, it has no gradient. Dead lock.
We will try to fix this kind of deseaster later.
'''
# def FM(A,B,a,n):
#     '''
#     input matrix A and matrix B and scalar a(lpha) and scalar n
#     n is the size of A (and B)
#     return the form of FM(A,B,a) in the paper
#     A * sqrt (A^-1 * B + (2a-1)/4*(I-A^-1 *B)^2 - (2a-1)/2*(I-A^-1 * B))
#     '''

#     AB = tf.matmul(tf.matrix_inverse(A),B) # A^-1 * B
#     IAB =  tf.subtract( tf.tile ( tf.expand_dims( tf.eye(n),0),[A.shape[0],1,1] )  , AB)         # I - A^-1 * B
#     eta = (2 * a - 1)/2                    # (2a-1)/2
#     before_root = tf.add( AB , tf.subtract( eta * eta * tf.matmul(IAB,IAB) , eta * (IAB) ) ) # (A^-1 * B + (2a-1)/4*(I-A^-1 *B)^2 - (2a-1)/2*(I-A^-1 * B))
#     before_root = tf.add (before_root , 1e-5 * tf.diag(tf.random_uniform([n])) )
#     S,U = Eig(before_root)
#     Sigma_root = tf.matrix_diag(tf.sqrt(S))
#     after_root = tf.matmul(tf.matmul(U,Sigma_root),tf.matrix_inverse(U)) # calculate the square root by eig-decomponent
#     after_root = tf.add(after_root,1e-5 * tf.eye(n))
#     #L = tf.cholesky(tf.matmul(A,after_root))
#     #return tf.matmul(L,tf.transpose(L,[0,2,1]))
#     result = tf.add( tf.matmul(A,after_root), 1e-5 * tf.eye(n) )
#     # result = tf.add(result,tf.transpose(result,[0,2,1]))/2
#     return result

def FM(A,B,a,n):
    '''
    input matrix A and matrix B(batch * n * n) and scalar a(lpha) and scalar n
    n is the size of A (and B)
    return the form of FM(A,B,a) in the paper
    A^0.5 * (A^-0.5 * B * A^-0.5)^a * A^0.5
    '''
    A = tf.add (A , 1e-5 * tf.diag(tf.random_uniform([n])) )
    S,U = Eig(A)
    Sigma_root = tf.matrix_diag(tf.sqrt(S))
    A1_2 = tf.matmul(tf.matmul(U,Sigma_root),U,transpose_b = True)
    A1_2 = tf.add (A1_2 , 1e-5 * tf.diag(tf.random_uniform([n])) )
    A_1_2 = tf.matrix_inverse(A1_2)
    before_root = tf.matmul(tf.matmul(A_1_2,B),A_1_2)
    before_root = tf.add (before_root , 1e-5 * tf.diag(tf.random_uniform([n])) )
    S_,U_ = Eig(before_root)
    Sigma_a = tf.matrix_diag(tf.pow(S_,a))
    after_a = tf.matmul(tf.matmul(U_,Sigma_a),U_,transpose_b = True) 
    return tf.matmul(tf.matmul(A1_2,after_a),A1_2)


def Chol_de(A,n):
    '''
    input matrix A and it's size n
    decomponent by Cholesky
    return a vector with size n*(n+1)/2
    '''
    #A = tf.add (A , 1e-10 * tf.diag(tf.random_uniform([n])) )
    # A = tf.cond( 
    #     tf.greater( tf.matrix_determinant(A),tf.constant(0.0) ) , 
    #     lambda: A, 
    #     lambda: tf.add (A , 1e-10 * tf.eye(n) ) )
    #L = tf.cholesky(A)
    L = A
    result = tf.slice(L,[0,0,0],[-1,1,1])
    for i in range(1,n):
        j = i
        result = tf.concat( [result , tf.slice(L,[0,i,0],[-1,1,j+1])],axis = 2 )

    result = tf.reshape(result,[-1,n*(n+1)/2])
    return result

def Chol_com(l,n,eps):
    '''
    input vector l and target shape n and eps to be the smallest value
    return lower trangle matrix
    '''
    batch_size = l.shape[0]
    lower_triangle_ = []
    for i in range(n):
        lower_triangle_.append( tf.expand_dims ( tf.concat( [tf.slice(l,[0,i*(i+1)/2],[-1,i+1]) , tf.zeros([batch_size,n-i-1]) ] , axis = 1 ) , -1 ) )

    lower_triangle = tf.concat (lower_triangle_ , axis = 2)
    result = []
    for i in range(batch_size):
        diag = tf.diag_part(tf.reshape(tf.slice(lower_triangle,[i,0,0],[1,n,n]),[n,n]))
        #diag = tf.clip_by_value(diag,-np.inf,-eps)
        diag = tf.diag(diag)
        result.append( tf.subtract ( tf.slice(lower_triangle,[i,0,0],[1,n,n]),diag ))
    return  tf.add(  tf.add(tf.concat(result,axis = 0) , tf.transpose(lower_triangle,[0,2,1]) )  , 0 * tf.eye(n) ) # make diag element to be eps or positive

def Readdata(file_address,matrix_length,n,true_label,class_num):
    data = np.load(file_address)
    s = data.shape
    assert s[0] == matrix_length and s[2]==s[3] and s[3]==n
    data = data.swapaxes(0,1)
    label = np.zeros([s[1],class_num])
    label[:,true_label] = 1
    return data,label

def shuffle_to_batch(data,label,batch_size):
    batch_data = []
    batch_label = []
    total_num = (label.shape)[0]
    list_ = random.sample(range(total_num),total_num)  # shuffle the data and label with the same index
    data = data[list_,:,:,:]
    label = label[list_,:]
    for i in range(0,total_num,batch_size):
        temp_data = data[i:i+batch_size,:,:,:]
        temp_label = label[i:i+batch_size,:]
        batch_data.append(temp_data)
        batch_label.append(temp_label)
    return batch_data,batch_label




batch_size = 10
matrix_length = 20
class_num = 2
matrix_size = 3
epoch_num = 1000

eps = 1e-10
n = matrix_size
a = [0.01, 0.25, 0.5, 0.9, 0.99]
a_num = len(a)

lr = 1e-3
decay_steps = 1000
decay_rate = 0.99
global_steps = tf.Variable(0,trainable = False)
learning_rate = tf.train.exponential_decay(lr, global_step = global_steps, decay_steps = decay_steps, decay_rate = decay_rate)
add_global = global_steps.assign_add(1)



X = tf.placeholder(np.float32,shape = (batch_size,matrix_length,n,n)) # M is batch * (n * n)
y = tf.placeholder(np.float32,shape = (batch_size,class_num)) # y is batch * classnumber
# temp_M = [  tf.placeholder(np.float32,shape=(  n, n)),
#             tf.placeholder(np.float32,shape=(  n, n)),
#             tf.placeholder(np.float32,shape=(  n, n)),
#             tf.placeholder(np.float32,shape=(  n, n)),
#             tf.placeholder(np.float32,shape=(  n, n))] # length is the same with len(a)
# temp_M = tf.placeholder(np.float32,shape=( a_num, n, n))


Weights = {
            'WR_root':tf.Variable(tf.ones([a_num,n]),trainable=True),
            'Wr_root':tf.Variable(tf.ones([n])),
            'Wt_root':tf.Variable(tf.ones([n])),
            'Wphi_root':tf.Variable(tf.ones([n])),
            'Ws_root':tf.Variable(tf.ones([a_num,n])),
            'wo':tf.Variable(tf.ones([1, n*(n+1)/2])),
          }

Bias =    {
            'Br':tf.Variable(tf.zeros([n*(n-1)/2,1]),trainable=True),
            'Bt':tf.Variable(tf.zeros([n*(n-1)/2,1])),
            'By':tf.Variable(tf.zeros([n*(n-1)/2,1])), # should be n*(n-1)/2, but when implementing it, I found that low - low.T will automatically give us n*(n-1)/2
            'bo':tf.Variable(tf.zeros([1])),
          }

W2 = tf.Variable(tf.random_normal([matrix_length, class_num]))
b2 = tf.Variable(tf.random_normal([      1      , class_num]))

initMt = tf.placeholder(np.float32,[batch_size,a_num,n,n])

Mt_1 = initMt

output_series = None

inputs_series = tf.unstack(tf.transpose(X,[1,0,2,3]))


for current_X in inputs_series:
    current_X = tf.reshape(current_X,[batch_size,n,n])

    Yt = Chol_de ( NUS( tf.reshape ( tf.slice(Weights['WR_root'],[0,0],[1,n]) ,[n]), tf.reshape ( tf.slice(Mt_1,[0,0,0,0],[-1,1,n,n] ) ,[batch_size,n,n]) ) , n )
    #xiaoxiang = NUS( tf.reshape ( tf.slice(Weights['WR_root'],[0,0,0],[1,n,n]) ,[n,n]), tf.reshape ( tf.slice(Mt,[0,0,0,0],[-1,1,n,n] ) ,[batch_size,n,n]) )
    for i in range(1,a_num):
        Yt = tf.add( Yt, Chol_de ( NUS( tf.reshape ( tf.slice(Weights['WR_root'],[i,0],[1,n]) ,[n]), tf.reshape ( tf.slice(Mt_1,[0,i,0,0],[-1,1,n,n] ) ,[batch_size,n,n]) ) , n ) )

    #print Yt.shape

    Yt = Chol_com (Yt, n, eps)

    Yt = Yt / a_num
    #print Yt.shape
    Rt = Chol_com( f ( Chol_de( Translation( Yt, Bias['Br'] , n ), n ) ), n, eps )

    tt = Chol_de ( Translation ( NUS ( Weights['Wt_root'], current_X ), Bias['Bt'] , n ),n )

    Phit = Chol_com ( f ( tf.add( tt, Chol_de ( NUS ( Weights['Wphi_root'] , Rt ),n ) ) ), n, eps )

    next_state = []
    for i in range(a_num):
        next_state.append (  tf.expand_dims ( FM ( tf.reshape ( tf.slice(Mt_1,[0,i,0,0],[-1,1,n,n] ) ,[batch_size,n,n]) , Phit, a[i] , n ) , 1 ) )
    Mt = tf.concat(next_state,axis = 1)

    St = Chol_de ( NUS ( tf.reshape ( tf.slice( Weights['Ws_root'], [0,0],[1,n] ),[n] ) , tf.reshape ( tf.slice(Mt,[0,0,0,0],[-1,1,n,n] ) ,[batch_size,n,n]) ),n )
    for i in range(1,a_num):
        St = tf.add ( St, Chol_de ( NUS ( tf.reshape ( tf.slice( Weights['Ws_root'], [i,0],[1,n] ), [n] ) , tf.reshape ( tf.slice(Mt,[0,i,0,0],[-1,1,n,n] ) ,[batch_size,n,n]) ) , n ) )
    St = Chol_com ( St ,n ,eps )
    
    St = St / a_num

    yt = tf.transpose( f( Chol_de ( Translation ( St, Bias['By'] , n ) , n ) ) ,[1,0] )

    ot = f( tf.add ( tf.matmul( Weights['wo'], yt ) , Bias['bo'] ) )
    #print ot.shape
    if output_series is None:
        output_series = ot
        #print output_series.shape
    else:
        output_series = tf.concat([output_series,ot],axis = 0)
    Mt_1 = Mt
output_series = tf.transpose(output_series,[1,0])
predict_label = tf.nn.softmax( tf.add( tf.matmul ( output_series, W2 ), b2 ) )

loss = tf.reduce_mean( tf.reduce_sum( -y * tf.log(predict_label+eps))) 



# loss = tf.reduce_mean(tf.reduce_mean( tf.pow( y - predict_label , 2 ) ))

with tf.control_dependencies([add_global]):
    opt = tf.train.AdagradOptimizer(learning_rate)
    train_step = opt.minimize(loss)
grad = tf.gradients(loss,[Weights['WR_root'],Weights['Wt_root'],Weights['Ws_root']])

'''
load data and label here
and make them randomly and group into batch
'''

data0,label0 = Readdata(file_address='c1_toy_covariance.npy',matrix_length=matrix_length,n=n,true_label=0,class_num=class_num)
data1,label1 = Readdata(file_address='c2_toy_covariance.npy',matrix_length=matrix_length,n=n,true_label=1,class_num=class_num)
data = np.append(data0,data1,axis = 0)
label = np.append(label0,label1,axis = 0)

batch_data,batch_label = shuffle_to_batch(data,label,batch_size)

batch_num = len(batch_data)

init_state = np.tile(np.eye(n) * 1e-5,[batch_size,a_num,1,1]) 

loss_p = 0

#CL = Chol_de(current_X,n)
#CC = Chol_com(CL,n,eps)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epoch_num):
        for batch_idx in range(batch_num):
            data_batch_in = np.reshape(batch_data[batch_idx],[batch_size,matrix_length,n,n])
            label_batch_in = np.reshape(batch_label[batch_idx],[batch_size,class_num])
            #pdb.set_trace()
            #print batch_idx
            #CL_,CC_ ,current_X_= sess.run([CL,CC,current_X],
            _, loss_ , predict_label_,Weights_,Rt_,Yt_,tt_,Phit_,Mt_,St_,yt_,ot_,Bias_,W2_,b2_,grad_= sess.run([train_step,loss,predict_label,Weights,Rt,Yt,tt,Phit,Mt,St,yt,ot,Bias,W2,b2,grad],
            #Yt_,Rt_,tt_,Phit_= sess.run([Yt,Rt,tt,Phit],
                     feed_dict={
                           X:data_batch_in,
                           y:label_batch_in,
                           initMt:init_state,
                            })
            # pdb.set_trace()
            # if not batch_idx%100:
            print loss_
            # print predict_label_
















