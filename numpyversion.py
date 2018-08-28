import autograd.numpy as np
import autograd
from autograd import grad
import pdb

def NUS(W_root,A):
    A = A + 1e-5*np.eye(A.shape[1])
    SU = list(np.linalg.eig(A))
    S = SU[0]
    U = SU[1]
    W = W_root * W_root
    S = S * W
    result = []
    for i in range(A.shape[0]):
        Sigma = np.diag(S[i,:])
        NUSresult = np.matmul(U[i,],Sigma)
        NUSresult = np.matmul(NUSresult,U[i,].T)
        result.append(NUSresult)
    return np.asarray(result)

def Chol_de(A):
    n = A.shape[1]
    batch_size = A.shape[0]
    result = np.zeros([batch_size,n*(n+1)/2],dtype = object)
    for i in range(n):
        result[:,i*(i+1)/2:(i+1)*(i+2)/2] = A[:,i,0:i+1]
    return result


def Lossfunc(ot,yt):
    return np.sum((ot-yt)*(ot-yt))

def training_loss(W_):
    W2 = W_[0]
    W = W_[1]
    tt = Chol_de(NUS(W,A))
    ot = np.dot(tt,W2)
    loss = Lossfunc(ot,yt)
    return loss

W = np.ones([3])
A = np.eye(3)
A = np.tile(A,(5,1,1))

W2 = np.ones([6,1])



yt = np.ones([5])

training_gradient_fun = grad(training_loss)

rrr = training_gradient_fun([W2,W])

pdb.set_trace()