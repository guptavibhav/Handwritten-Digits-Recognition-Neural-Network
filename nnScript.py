import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import matplotlib.pyplot as plt
import random

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:        
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    
    return s 


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('/Users/vibhavgupta/Downloads/basecode/mnist_all.mat') #loads the MAT object as a Dictionary
    #print mat.keys()
    #Pick a reasonable size for validation data
    #test0 = mat.get('train0')
    #test0_data = 
    #test0_dat = np.array(test0)
    #print test0_dat
    #np.savetxt('testMat.txt', test0_dat)
    #Your code here
    
    train_data  = np.array([])
    train_label = np.array([]) # 
    validation_data  = np.array([])
    validation_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])
     
    # Getting the data from the mat file and loading the training data ,validation data
    # test data
    a = 0
    for i in range(10):
        
        # Process the training data and split into validation and train
        MTR = mat.get('train'+str(i))
        a = range(MTR.shape[0])
        MTT = mat.get('test'+str(i))
        b = range(MTT.shape[0])
        B1 = MTT[0:]
        aperm = np.random.permutation(a)
        A1 = MTR[aperm[0:1000],:]
        A2 = MTR[aperm[1000:],:]
        
        # set the labels for the rows 
        valid_label = np.empty(A1.shape[0])
        valid_label.fill(i)
        validation_label = np.append(validation_label,valid_label)
        
        tr_label = np.empty(A2.shape[0])
        tr_label.fill(i)
        train_label = np.append(train_label,tr_label)
        
        te_label = np.empty(MTT.shape[0])
        te_label.fill(i)
        test_label = np.append(test_label,te_label)
        
        if(i!=0):
            train_data = np.append(train_data,A2,axis = 0)
            validation_data = np.append(validation_data,A1,axis=0)
            test_data = np.append(test_data,B1,axis = 0)
        if(i==0):
            train_data = np.append(train_data,A2)
            train_data = np.reshape(train_data,(A2.shape[0],A2.shape[1]))
            
            validation_data = np.append(validation_data,A1)
            validation_data = np.reshape(validation_data,(A1.shape[0],A1.shape[1]))
           
            test_data = np.append(test_data,B1)
            test_data = np.reshape(test_data,(B1.shape[0],B1.shape[1])) 
            
    ##print "Train data is "+str(train_data.shape[0])
   ## print "Test label is "+str(test_label.shape[0])
    #print "Validation label is "+str(validation_label.shape[0])
    #print "Train label is "+str(train_label.shape[0]) 
    
    #matdata = np.array([])
    #matdata = np.append(matdata,train_data)
    #matdata = np.reshape(matdata,(train_data.shape[0],train_data.shape[1]))
    #matdata = np.append(matdata,validation_data,axis = 0)
    #matdata = np.append(matdata,test_data,axis = 0)
   
    #print matdata.shape[0]
    # testFunction(validation_data) 
    
    train_data = train_data / 255.0
    validation_data = validation_data /255.0
    test_data = test_data /255.0
    
    #print train_data[0].reshape((28,28)) 
    #print validation_data[0].reshape((28,28))         
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def testFunction():

    training_data = np.array([np.linspace(0,1,num=5),np.linspace(1,0,num=5)])
    training_label = np.array([0,1])
    
    training_data = np.array([np.linspace(0,1,num=5),np.linspace(1,0,num=5)])
    training_label = np.array([0,1])
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    #params = np.linspace(-5,5, num=26)
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    
    t = w2.T
    obj_val = 0  
    hdOp = np.array([])
    
    #appending the bias node 
    training_data = np.append(training_data,np.ones([len(training_data),1]),1)
    hdOp = np.dot(training_data,(w1.T))
       
    # we have to apply the sigmoid over hdOp
    z = sigmoid(hdOp) 
    # adding the bias node
    z = np.append(z,np.ones([len(z),1]),1)  
    np.savetxt('op.txt',z,fmt = '%f')                
    b = np.dot(z,(w2.T))
    ol = sigmoid(b)
       
    y = np.empty(ol.shape)
       
    #create labels for matrices
    lbl_mat= np.array(([1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],
                       [0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],
                       [0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]))
                       
    lbl_mat1 = np.array(([1,0],[0,1]))                       
    
    from collections import Counter   
     
    g = Counter(train_label)    
    
    y = np.array([])
    x = np.array([])
    for i in range(10) :   # change this based on input
        indcnt = g.get(i)
        #print indcnt
        x = np.tile(lbl_mat[i],(indcnt,1))
        if i == 0 :
            y = np.append(y,x)
            y = np.reshape(y,(indcnt,10))  #change to 10
        if i!=0:
            y = np.append(y,x,axis = 0)
    
    #for i in range(training_data.shape[0]):
        #for l in range(n_class):
                 #obj_val += y[i][l]*np.log(ol[i][l]) + (1-y[i][l])*np.log(1-ol[i][l])  
                 
    #neg_log_likelihood = (np.sum(y*np.log(ol)+(1-y)*np.log(1-ol))                       
    
    obj_val =-(np.sum(y*np.log(ol) + (1-y)*np.log(1-ol)))/training_data.shape[0]
    
    #obj_val = - (obj_val / training_data.shape[0]) 
    delta = np.empty(ol.shape)
    delta.fill(0)

    #O = np.amax(ol,axis = 1)
    delta = ol - y
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #grad_w2 = np.dot(delta,z)
    grad_w2 = np.dot(delta.T,z)
    
    #removing the bias from the z
    #calculating the equation  (1-z)z summation(delta.W2)xp for gradw2
    
    #z = np.delete(z,z.shape[1]-1,1)
    diff = 1- z
    sigdiff = z*(1-z)
    summation = np.dot(delta,w2)
    summation = summation[:,:-1]    
    sigdiff_training = summation * sigdiff[:,:-1]
    sigdiff_training = sigdiff_training.T
    grad_w1 = np.dot(sigdiff_training,training_data)
    grad_w1 = grad_w1.T
    grad_w2 = (grad_w2 / training_data.shape[0])
    grad_w1 = (grad_w1 / training_data.shape[0]).T
    
    #np.savetxt('/home/di3/Documents/CSE UB/SPRING 2015/CSE 574 ML/basecode/GRAD_W1.txt',grad_w1,fmt = '%f')
    #np.savetxt('/home/di3/Documents/CSE UB/SPRING 2015/CSE 574 ML/basecode/GRAD_W2.txt',grad_w2,fmt = '%f')
   
    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    w1_sq=w1*w1
    w2_sq=w2*w2
    reg_term=0.0
    reg_term=np.sum(w1_sq)+np.sum(w2_sq)
    obj_val=obj_val+((lambdaval*reg_term)/(2*training_data.shape[0]))
  
  
    return (obj_val,obj_grad)

def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    hdOp = np.array([])
    data = np.append(data,np.ones([len(data),1]),1)
    hdOp = np.dot(data,(w1.T))
    
    z = sigmoid(hdOp) 
    # adding the bias node
    z = np.append(z,np.ones([len(z),1]),1)  
    
    b = np.dot(z,(w2.T))
    ol = sigmoid(b)
    lb = np.argmax(ol,axis = 1)
    
    np.savetxt('outtttttp.txt',ol,fmt = '%f')
    np.savetxt('labels.txt',lb,fmt = '%f')
    
    labels = np.array([])
    labels = lb
    
    return labels

"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();
testFunction();

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 100;
				   
# set the number of nodes in output unit
n_class = 10;	  #change back to 10 - Mithun

################### test function written here 



#n_input = 5
#n_hidden = 3
#n_class = 2
#train_data = np.array([np.linspace(0,1,num=5),np.linspace(1,0,num=5)])
#train_label = np.array([0,1])
#lambdaval = 0
#params = np.linspace(-5,5, num=26)


			   			   
			   			   			   			   
			   			   			   			   			   			   
##############################################################			   			   			   			   			   			   			   			   			   			   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
