import tensorflow as tf
import numpy as np
import random
import pdb

from tensorflow.python.ops.distributions.util import fill_triangular
'''
All the data matrix should be batch_size * n * n
All the matrix W should be n * n
'''


def NUS(W_root,A):
    '''
    input the matrix A and matrix W_root
    A is the one to be compute
    W_root is the square-root of W
    which will make W positive
    '''
    S,U = tf.self_adjoint_eig(A)
    W = tf.multiply(W_root,W_root)

    Sigma = tf.matrix_diag(S)
    NUSresult = tf.matmul(U,W)                            # U * W
    NUSresult = tf.matmul(NUSresult,Sigma)                # U * W * Sigma
    NUSresult = tf.matmul(NUSresult,U,transpose_b = True) # U * W * Sigma * U.T
    return NUSresult

def Translation(A,B):
    '''
    input the matrix A and vector B
    change B to be SO 
    like [[0 ,  1, 2]
          [-1,  0, 3]
          [-2, -3, 0]]
    return B * A * B.T
    '''
    B = tf.reshape(B,[-1,])
    lower_triangel = fill_triangular(B)
    
    B_matrix = tf.subtract(lower_triangel, tf.transpose(lower_triangel))
    Tresult = tf.matmul(B_matrix,A)                              # B * A
    Tresult = tf.matmul(Tresult,B_matrix,transpose_b = True)      # B * A * B.T
    return Tresult

def FM(A,B,a,n):
    '''
    input matrix A and matrix B and scalar a(lpha) and scalar n
    n is the size of A (and B)
    return the form of FM(A,B,a) in the paper
    A * sqrt (A^-1 * B + (2a-1)/4*(I-A^-1 *B)^2 - (2a-1)/2*(I-A^-1 * B))
    '''
    AB = tf.matmul(tf.matrix_inverse(A),B) # A^-1 * B
    IAB =  tf.subtract( tf.eye(n) , AB)         # I - A^-1 * B
    eta = (2 * a - 1)/2                    # (2a-1)/2
    before_root = AB + eta * eta * tf.matmul(IAB,IAB) - eta * (IAB) # (A^-1 * B + (2a-1)/4*(I-A^-1 *B)^2 - (2a-1)/2*(I-A^-1 * B))
    S,U = tf.self_adjoint_eig(before_root)
    Sigma_root = tf.matrix_diag(tf.sqrt(S))
    after_root = tf.matmul(tf.matmul(U,Sigma_root),U,transpose_b = True) # calculate the square root by eig-decomponent
    return tf.matmul(A,after_root)

# def nearestPD(A,eps):
#     A3 = A
#     if isPD(A3):
#         return A3

#     I = tf.eye(A.shape[0])
#     k = 1
#     while not isSP(A3):
#         eig,_ = tf.self_adjoint_eig(A3)
#         mineig = tf.min(eig)
#         tf.assign(A3, tf.add(A3, I * ((-mineig) * k**2 + eps)))
#         k += 1
#     return A3

# def isPD(A):
#     if tf.greater(tf.matrix_determinant(A), tf.constant(0.)) is not None:
#       return True
#     else:
#       return False
#     # try:
#     #     _ = tf.cholesky(A)
#     #     return True
#     # except:
#     #     return False


def Chol_de(A,n):
    '''
    input matrix A and it's size n
    decomponent by Cholesky
    return a vector with size n*(n-1)/2
    '''
    #A = nearestPD(A,1e-10)
    A = tf.cond( tf.greater( tf.matrix_determinant(A),tf.constant(0.0) ) , lambda: A, lambda: tf.add (A , 1e-10 * tf.eye(n) ) )
    L = tf.cholesky(A)
    
    result = tf.slice(L,[0,0],[1,1])
    for i in range(1,n):
        j = i
        result = tf.concat( [result , tf.slice(L,[i,0],[1,j+1])],axis = 1 )

    result = tf.reshape(result,[-1,])
    return result

def Chol_com(l,n,eps):
    '''
    input vector l and target shape n and eps to be the smallest value
    return lower trangle matrix
    '''
    lower_triangle = tf.Variable(tf.zeros([n,n]),trainable = False)
    for i in range(n):
        for j in range(i+1):
            tf.assign(lower_triangle [i,j] , l[ i*(i+1)/2 + j ] )



    diag = tf.diag_part(lower_triangle)
    diag = tf.clip_by_value(diag,-np.inf,-eps)
    diag = tf.diag(diag)
    return tf.subtract(lower_triangle,diag) # make diag element to be eps or positive

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




batch_size = 1
matrix_length = 20
class_num = 2
matrix_size = 6
epoch_num = 1000

eps = 1e-10
n = matrix_size
a = [0, 0.25, 0.5, 0.9, 0.99]
a_num = len(a)

lr = 1e-2
decay_steps = 1000
decay_rate = 0.99
global_steps = tf.Variable(0,trainable = False)
learning_rate = tf.train.exponential_decay(lr, global_step = global_steps, decay_steps = decay_steps, decay_rate = decay_rate)
add_global = global_steps.assign_add(1)



X = tf.placeholder(np.float32,shape = (matrix_length,n,n)) # M is batch * (n * n)
y = tf.placeholder(np.float32,shape = (class_num)) # y is batch * classnumber
# temp_M = [  tf.placeholder(np.float32,shape=(  n, n)),
#             tf.placeholder(np.float32,shape=(  n, n)),
#             tf.placeholder(np.float32,shape=(  n, n)),
#             tf.placeholder(np.float32,shape=(  n, n)),
#             tf.placeholder(np.float32,shape=(  n, n))] # length is the same with len(a)
# temp_M = tf.placeholder(np.float32,shape=( a_num, n, n))

Weights = {
            'WR_root':tf.Variable(tf.random_normal([a_num,n,n])),
            'Wr_root':tf.Variable(tf.random_normal([n,n])),
            'Wt_root':tf.Variable(tf.random_normal([n,n])),
            'Wphi_root':tf.Variable(tf.random_normal([n,n])),
            'Ws_root':tf.Variable(tf.random_normal([a_num,n,n])),
            'wo':tf.Variable(tf.random_normal([1, n*(n+1)/2])),
          }

Bias =    {
            'Br':tf.Variable(tf.random_normal([n*(n+1)/2,1])),
            'Bt':tf.Variable(tf.random_normal([n*(n+1)/2,1])),
            'By':tf.Variable(tf.random_normal([n*(n+1)/2,1])), # should be n*(n-1)/2, but when implementing it, I found that low - low.T will automatically give us n*(n-1)/2
            'bo':tf.Variable(tf.random_normal([1])),
          }

W2 = tf.Variable(tf.random_normal([matrix_length, class_num]))
b2 = tf.Variable(tf.random_normal([      1      , class_num]))

Mt = tf.Variable(tf.random_normal([a_num,n,n]),trainable = False)

output_series = None

inputs_series = tf.unstack(X)


for current_X in inputs_series:
    current_X = tf.reshape(current_X,[n,n])

    Yt = Chol_de ( NUS( tf.reshape ( tf.slice(Weights['WR_root'],[0,0,0],[1,n,n]) ,[n,n]), tf.reshape ( tf.slice(Mt,[0,0,0],[1,n,n] ) ,[n,n]) ) , n )
    for i in range(1,a_num):
        Yt = tf.add( Yt, Chol_de ( NUS( tf.reshape ( tf.slice(Weights['WR_root'],[i,0,0],[1,n,n]) ,[n,n]), tf.reshape ( tf.slice(Mt,[i,0,0],[1,n,n] ) ,[n,n]) ) , n ) )

    #print Yt.shape

    Yt = Chol_com (Yt, n, eps)
    #print Yt.shape
    Rt = Chol_com( tf.nn.relu( Chol_de( Translation( Yt, Bias['Br'] ), n ) ), n, eps )
    
    tt = Chol_de ( Translation ( NUS ( Weights['Wt_root'], current_X ), Bias['Bt'] ),n )

    Phit = Chol_com ( tf.nn.relu( tf.add( tt, Chol_de ( NUS ( Weights['Wphi_root'] , Rt ),n ) ) ), n, eps )

    for i in range(a_num):
        tf.assign(Mt[i,:,:] , FM ( tf.reshape ( tf.slice(Mt,[i,0,0],[1,n,n] ) ,[n,n]) , Phit, a[i] , n ))

    St = Chol_de ( NUS ( tf.reshape ( tf.slice( Weights['Ws_root'], [0,0,0],[1,n,n] ),[n,n] ) , tf.reshape ( tf.slice(Mt,[0,0,0],[1,n,n] ) ,[n,n]) ),n )
    for i in range(1,a_num):
        St = tf.add ( St, Chol_de ( NUS ( tf.reshape ( tf.slice( Weights['Ws_root'], [i,0,0],[1,n,n] ), [n,n] ) , tf.reshape ( tf.slice(Mt,[i,0,0],[1,n,n] ) ,[n,n]) ) , n ) )
    St = Chol_com ( St ,n ,eps )


    #xingjian,_ = tf.self_adjoint_eig(nearestPD(Translation ( St, Bias['By'] ),1e-10))
    #flag = tf.convert_to_tensor(isPD(Translation ( St, Bias['By'] )))

    yt = tf.reshape( tf.nn.relu( Chol_de ( Translation ( St, Bias['By'] ) , n ) ) ,[n*(n+1)/2,1] )

    ot = tf.nn.relu( tf.add ( tf.matmul( Weights['wo'], yt ) , Bias['bo'] ) )
    #print ot.shape
    if output_series is None:
        output_series = ot
        #print output_series.shape
    else:
        output_series = tf.concat([output_series,ot],axis = 1)

predict_label = tf.nn.softmax( tf.add( tf.matmul ( output_series, W2 ), b2 ) )

#loss = tf.reduce_mean( tf.reduce_sum( -y * tf.log(predict_label)))

loss = tf.reduce_sum( tf.pow( predict_label - y , 2 ) )

with tf.control_dependencies([add_global]):
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss)


'''
load data and label here
and make them randomly and group into batch
'''

data0,label0 = Readdata(file_address='c1_covariance.npy',matrix_length=matrix_length,n=n,true_label=0,class_num=class_num)
data1,label1 = Readdata(file_address='c2_covariance.npy',matrix_length=matrix_length,n=n,true_label=1,class_num=class_num)
data = np.append(data0,data1,axis = 0)
label = np.append(label0,label1,axis = 0)

batch_data,batch_label = shuffle_to_batch(data,label,batch_size)

batch_num = len(batch_data)







loss_p = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epoch_num):
        for batch_idx in range(batch_num):
            data_batch_in = np.reshape(batch_data[batch_idx],[matrix_length,n,n])
            label_batch_in = np.reshape(batch_label[batch_idx],[class_num])
            #pdb.set_trace()
            #print batch_idx
            _, loss_,predict_label_ = sess.run([train_step,loss,predict_label],
            # xingjian_,flag_ = sess.run([xingjian,flag],
                     feed_dict={
                           X:data_batch_in,
                           y:label_batch_in,
                            })
            #print xingjian_,flag_
            # print predict_label_,label_batch_in

            # pdb.set_trace()
            if batch_idx % 1000:
                loss_p += loss_
            else:
                print loss_p/1000
                print predict_label_,label_batch_in
                loss_p = 0
