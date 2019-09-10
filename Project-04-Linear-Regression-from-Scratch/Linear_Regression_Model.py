#!/usr/bin/env python
# coding: utf-8

# # Linear Regression from scratch

# In this code, we are trying to implement a Linear Regression model from scratch for educationl purposes.

# In[5]:


import numpy as np


# In Linear Regression problems, we have a continuoes variable which we want to predict.   
# The data we will use at the end is USA_Housing.csv data.
# 
# **Notation**   
# 
# 
# | Quantity                     | Notation           | type(in code)     | 
# | -----------                  | -----------        |---------------:   |
# | # features                   | $n$                |  ```int```        |
# | # training examples          | $m$                |  ```int```        |
# | # features matrix            | $X_{n\times m}$    |  ```numpy array```|
# | shape of features matrix     | $(n\times m)$      |  ```tuple```      |
# | # target matrix              | $y_{1\times m}$    |  ```numpy array```|
# | shape of target matrix       | $(1\times m)$      |  ```tuple```      |
# | weight parameters            | $w_{n\times1}$     |  ```numpy array``` and ```dict```|
# | bias parameters              | $b$                |  ```numpy array``` and ```dict```|
# | $i^{\text{th}}$ training example    | $X^{i}\equiv i^{\text{th}}$ column of $X$| ```numpy array```|
# 
# In a Linear Regression model we have: 
# $$ \hat{y}(X^{i}) = w^\dagger X^{i} + b,$$
# and in Python's vectorized implementation, we have:
# $$ \hat{y}(X) = w^\dagger X + b$$
# 

# In[6]:


def scale_info(X,y,axis_val=1):
    '''
    
    '''
    # number of features and training example
    n,m = X.shape
    
    minX = np.min(X,axis=axis_val).reshape(n,1)     # axis =1 because of our notation
    rangeX = ( np.max(X,axis=axis_val)-np.min(X,axis=axis_val) ).reshape(n,1)
    miny = np.min(y,axis = axis_val).reshape(1,1)
    rangey = ( np.max(y,axis=axis_val)-np.min(y,axis=axis_val) ).reshape(1,1)
    
    info = dict()
    info["minX"]   = minX
    info["miny"]   = miny
    info["rangeX"] = rangeX
    info["rangey"]= rangey
    
    return info


# In[7]:


def normalize_data(X,y,scaling_info):
    '''
    a function that normalizes data according to min-max method.
    
    X: feature data (n*m numpy array. n: number of features. m: number of training examples.)
    
    y: target data (1*m numpy array.  m: number of training examples.)
    
    scaling_info: information about how to scale data
    
    '''
    info = scale_info(X,y,axis_val=1)
    
    minX   = info["minX"]
    rangeX = info["rangeX"]
    miny   = info["miny"]
    rangey = info["rangey"]
    
    X_scaled = (X-minX)/rangeX
    y_scaled = (y-miny)/rangey
    return X_scaled,y_scaled


# In[8]:


def initialize_parameters(X):
    '''
    a function to create and intialize dictionary of parameters
    
    input X: features matrix. (a numpy array)
    X.shape = n * m (n: number of features. m: number of training examples)
    
    output: parameters ( a Python dictionary)
        the lenghth of parameters dictionary is n+1, n for w parameters in Linear Regression
        and 1 for bias parameter
    '''
    # number of features and training example
    n,m = X.shape
    
    # parameters dictionary
    parameters = dict()
    for i in xrange(0,n):
            parameters["w[{}]".format(i)] = np.random.randn()*0.01     #intialized with random Gaussian number 
    
    parameters["b"] = 1
    
    assert(type(parameters)==dict)
    
    return parameters


# In[9]:


# test function
X = np.array(range(0,15)).reshape(5,3)
parameters = initialize_parameters(X)
parameters


# In[10]:


def yhat(X,parameters):
    '''
    input X: features matrix. (a numpy array)
    X.shape = n * m (n: number of features. m: number of training examples)
    input parameters: (w[i] and b parameters). (a Python dictionary)
    output: yhat (a linear regression function result): (a numpy array)
    output is yhat value ( a 1*m numpy array) and parameters
    
    '''
    # number of features
    n = len(parameters) - 1
    
    # variables for parameters
    w = np.zeros((n, 1))
    b = np.ones((1,1))

    # assign values for variables from parameters dictionary
    for i in range(0,n):
        w[i:i+1,0:1] = parameters["w[{}]".format(i)]
    
    b[0:1, 0:1] = parameters["b"]
    
    
    yhat_value =  np.matmul(w.T,X) + b 
    
    return yhat_value
    


# In[11]:


# test function
X = np.array(range(0,15)).reshape(5,3)
parameters = initialize_parameters(X)
yhat(X,parameters)


# In[12]:


def cost(yhat,y_true):
    '''
    A loss function. 
    input yhat: a (1*m) numpy array
    input ytrue: a (1*m) numpy array
    '''
    m = yhat.shape[1]
    
    assert(yhat.shape == y_true.shape)
    
    loss = (1.0/m)*np.matmul(yhat - y_true, (yhat - y_true).T )[0][0]
    return loss


# In[13]:


# test function
X = np.array(range(1,10)).reshape(3,3)
y_true = np.array(range(2,5)).reshape(1,3)

parameters = initialize_parameters(X)
yhat_value = yhat(X,parameters)
cost(yhat_value,y_true)


# In[14]:


def derivatives(X, parameters, y_true):
    '''
    '''
    
    yhat_value = yhat(X,parameters)
    
    # number of features
    n = len(parameters) - 1
    m = X.shape[1]
    
    # variables for parameters
    dw = np.zeros((n, 1))
    db = (2.0/m)*np.matmul(np.ones([1,m]), (yhat_value - y_true).T)
    
    grads = dict()
    # assign values for variables from parameters dictionary
    for i in range(0,n):
        grads["dw[{}]".format(i)] = (2.0/m)*np.matmul(X[i:i+1,0:m], (yhat_value - y_true).T)

    grads["db"] = db
    
    return grads
    


# In[15]:


def update_parameters(parameters,grads,learning_rate=0.001):
    '''
    '''
    # number of features
    n = len(parameters) - 1
    m = X.shape[1]
    
    # variables for parameters
    w = np.zeros((n, 1))
    b = np.ones((1,1)) 
    dw = np.zeros((n, 1))
    db = np.ones((1,1))
    
    # assign values for variables from parameters dictionary
    for i in range(0,n):
        w[i:i+1,0:1] = parameters["w[{}]".format(i)]
        dw[i:i+1,0:1] = grads["dw[{}]".format(i)]
    
    b[0:1, 0:1] = parameters["b"]
    db[0:1, 0:1] = grads["db"]
    
    w -= learning_rate*dw
    b -= learning_rate*db
    
    # assign values for variables from parameters dictionary
    for i in range(0,n):
        parameters["w[{}]".format(i)] = w[i:i+1,0:1]
    
    parameters["b"] = b
    
    return parameters


# In[16]:


# test function
X = np.array(range(1,10)).reshape(3,3)
y_true = np.array(range(2,5)).reshape(1,3)

parameters = initialize_parameters(X)
yhat_value = yhat(X,parameters)
cost_value = cost(yhat_value,y_true)
grads = derivatives(X, parameters, y_true,)
update_parameters(parameters,grads)


# In[23]:


def Linear_Regression_Model(X,y_true, learning_rate=0.001, iteration=10000):
    
    # initialize parematers
    parameters = initialize_parameters(X)
    
    #find scaling variables
    info = scale_info(X,y_true,axis_val=1)
    
    minX   = info["minX"]
    rangeX = info["rangeX"]
    miny   = info["miny"]
    rangey = info["rangey"]
    
    #normalized data
    X,y_true = normalize_data(X,y_true,info)
    
    # stores values of cost for each iteration
    cost_list = []
        
    # learn the model
    for i in xrange(1,iteration):
        grads = derivatives(X, parameters, y_true)
        parameters = update_parameters(parameters,grads,learning_rate)
        if i%100==0:
            cost_list.append(cost(yhat(X,parameters),y_true))
            
    def predict(X_test):
        X_test_scaled = (X_test - minX)/(rangeX)
        yhat1 = yhat(X_test_scaled,parameters)
        yhat2 = yhat1*(rangey)+miny
        return yhat2
            
    return predict, cost_list


# In[24]:


#test function
X = np.array(range(1,10)).reshape(3,3)
y_true = np.array(range(2,5)).reshape(1,3)
prediction, cost_list = Linear_Regression_Model(X,y_true)
print("Prediction = "+str(prediction(X)))


# In[25]:


def plot_cost(cost_list,plot_name="cost"):
    
    import matplotlib.pyplot as plt
    get_ipython().magic(u'matplotlib inline')
    fig = plt.figure(figsize=(10,6))
    plt.title("cost function vs iteration for a linear regression model")
    plt.ylabel("cost")
    plt.yscale('log')
    plt.xlabel("iteration/100")
    plt.xscale('log')
    plt.plot(cost_list, c="b", linestyle='--', marker='o', markersize=5, markerfacecolor='r')
    #plt.show()
    plt.savefig(plot_name+".png")
    
    return fig


# In[28]:


# test function
#fig = plot_cost(cost_list,"cost")


# In[27]:

