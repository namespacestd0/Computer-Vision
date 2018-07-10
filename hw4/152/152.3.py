from pylab import *
from numpy import *
from scipy import linalg
import util

dino1_color, dino2_color, cor1, cor2 = util.load_dino('dino2.mat')
dino1= np.dot(dino1_color, [0.299, 0.587, 0.114])
dino2= np.dot(dino2_color, [0.299, 0.587, 0.114])

def compute_fundamental(x1,x2):
    
    x1 /= x1[2]
    S1 = sqrt(2) / std(x1[:2])
    print(S1)
    m1 = mean(x1[:2],axis=1)
    T1 = array([[S1,0,-S1*m1[0]],[0,S1,-S1*m1[1]],[0,0,1]])
    x1 = dot(T1,x1)
    print(x1)
    x2 = x2 / x2[2]
    m2 = mean(x2[:2],axis=1)
    S2 = sqrt(2) / std(x2[:2])
    print(S2)
    T2 = array([[S2,0,-S2*m2[0]],[0,S2,-S2*m2[1]],[0,0,1]])
    x2 = dot(T2,x2)
    print(x2)
    arr = zeros((13,9))
    for i in range(13):
        arr[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i]]
            
    U,S,V = linalg.svd(arr)
    F = V[-1].reshape(3,3)        
    U,S,V = linalg.svd(F)
    S[2] = 0
    F = dot(U,dot(diag(S),V))    
    F /= F[2,2]
    F = dot(T1.T,dot(F,T2))
    return F/F[2,2]

h1 = ones((3,13))
h2 = ones((3,13))
for i in range(len(cor1)):
    h1[0,i] = cor1[i][0]
    h1[1,i] = cor1[i][1]
    h2[0,i] = cor2[i][0]
    h2[1,i] = cor2[i][1]
    
F = compute_fundamental(h1,h2)