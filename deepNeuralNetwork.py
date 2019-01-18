import numpy as np
from load_mnist import mnist
import matplotlib.pyplot as plt
import pdb
import math
import sys, ast
from sklearn.utils.extmath import softmax
import random

def relu(Z):
    A = np.maximum(0,Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def relu_der(dA, cache):
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z<0] = 0
    return dZ

def linear(Z):
    A = Z
    cache = {}
    return A, cache

def linear_der(dA, cache):
    dZ = np.array(dA, copy=True)
    return dZ

def softmax_cross_entropy_loss(Z, Y=np.array([])):

    for i in range(Z.shape[1]):
        Z[:,i] = Z[:,i] - Z[:,i].max()

    numerator = np.exp(Z)
    denominator = np.sum(np.exp(Z), axis=0, keepdims=True)
    A = numerator / denominator

    if len(Y) == 0:
        return A, {}, 0

    Y = one_hot(Y)

    m = Y.shape[1]
    epsilon = math.pow(10, -3)
    loss = (-1/m) * np.sum(Y * np.log(A+epsilon))

    cache = {}
    cache['A'] = A

    return A, cache, loss

def softmax_cross_entropy_loss_der(Y, cache):
    Y_bar = one_hot(Y)
    dZ = cache['A'] - Y_bar

    return dZ

def initialize_multilayer_weights(net_dims):

    np.random.seed(0)
    numLayers = len(net_dims)
    parameters = {}
    v = {}
    s = {}
    for l in range(numLayers-1):
        parameters["W"+str(l+1)] = np.random.randn(net_dims[l+1], net_dims[l]) * math.sqrt(2/(net_dims[l+1]+net_dims[l]))
        parameters["b"+str(l+1)] = np.random.randn(net_dims[l+1], 1) * 0.01
        v["dW"+str(l+1)] = 0
        v["db"+str(l+1)] = 0
        s["dW"+str(l+1)] = 0
        s["db"+str(l+1)] = 0
    return parameters, v, s

def linear_forward(A, W, b):

    Z = np.dot(W, A) + b

    cache = {}
    cache["A"] = A

    return Z, cache

def layer_forward(A_prev, W, b, activation):
    
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)
    
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    return A, cache

def multi_layer_forward(X, parameters):
    
    L = len(parameters)//2  
    A = X
    caches = []
    for l in range(1,L):  # since there is no W0 and b0
        A, cache = layer_forward(A, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)

    AL, cache = layer_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "linear")
    caches.append(cache)
    return AL, caches

def linear_backward(dZ, cache, W, b):
    
    A_prev = cache["A"]
   
    dW = np.dot(dZ, A_prev.T) / A_prev.shape[1]
    db = np.sum(dZ, axis=1, keepdims=True) / A_prev.shape[1]
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation):
    
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    elif activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db

def multi_layer_backward(dAL, caches, parameters):
    L = len(caches)  
    gradients = {}
    dA = dAL
    activation = "linear"
    for l in reversed(range(1,L+1)):
        dA, gradients["dW"+str(l)], gradients["db"+str(l)] = layer_backward(dA, caches[l-1], parameters["W"+str(l)],parameters["b"+str(l)], activation)
        activation = "relu"
    return gradients

def classify(X, parameters):
    Z, caches = multi_layer_forward(X, parameters)
    A, cache, loss = softmax_cross_entropy_loss(Z)
    Ypred = np.argmax(A, axis=0)
    return Ypred


def approximate_gradient(train_x, train_y, L, parameter, parameter_value, all_parameters):
    
    delta = math.pow(10, -6)*5
    
    all_parameters[parameter] = parameter_value
    cost_old = 0  
    for i in range(1,L):
        AL, caches = multi_layer_forward(train_x, all_parameters)
        A3, cache, cost_old = softmax_cross_entropy_loss(AL, train_y)

    all_parameters[parameter] = parameter_value + delta
    cost_new = 0 
    for i in range(1,L):
        AL, caches = multi_layer_forward(train_x, all_parameters)
        A3, cache, cost_new = softmax_cross_entropy_loss(AL, train_y)

    gradient = (cost_new - cost_old) / delta

    print("old cost", parameter, cost_old)
    print("new cost", parameter, cost_new)
    print("gradient", parameter, gradient)

    return gradient



def update_parameters(train_x, train_y, parameters, gradients, epoch, iteration, learning_rate, gamma, beta, NAG_coeff, num_dims, v, s, decay_rate=0.0, optimization_method='classical'):

    alpha = learning_rate*(1/(1+decay_rate*epoch))
    L = len(parameters)//2
    epsilon = math.pow(10, -4)

    if optimization_method == 'classical':

        for i in range(1, num_dims):
            
            parameters["W"+str(i)] = parameters["W"+str(i)] - (alpha * gradients["dW"+str(i)])
            parameters["b"+str(i)] = parameters["b"+str(i)] - (alpha * gradients["db"+str(i)])

    elif optimization_method == 'NAG':

        current_gradient = {}
        v_prev = {}

        for i in range(1, num_dims):,
            
            current_gradient["W"+str(i)] = approximate_gradient(train_x, train_y, num_dims, "W"+str(i), parameters["W"+str(i)]+NAG_coeff*v["dW"+str(i)], parameters)
            current_gradient["b"+str(i)] = approximate_gradient(train_x, train_y, num_dims, "b"+str(i), parameters["b"+str(i)]+NAG_coeff*v["db"+str(i)], parameters)

            v["dW"+str(i)] = (NAG_coeff*v["dW"+str(i)]) + ((1-NAG_coeff) * current_gradient["W"+str(i)])
            v["db"+str(i)] = (NAG_coeff*v["db"+str(i)]) + ((1-NAG_coeff) * current_gradient["b"+str(i)])
            
            parameters["W"+str(i)] = parameters["W"+str(i)] - alpha*v["dW"+str(i)]
            parameters["b"+str(i)] = parameters["b"+str(i)] - alpha*v["db"+str(i)]   
    
    elif optimization_method == 'momentum':
        
        for i in range(1, num_dims):
            
            v["dW"+str(i)] = (gamma * v["dW"+str(i)]) + (alpha * gradients["dW"+str(i)])
            v["db"+str(i)] = (gamma * v["db"+str(i)]) + (alpha * gradients["db"+str(i)])
            
            parameters["W"+str(i)] = parameters["W"+str(i)] - v["dW"+str(i)]
            parameters["b"+str(i)] = parameters["b"+str(i)] - v["db"+str(i)]
   
    elif optimization_method == 'rmsprop':
        
        for i in range(1, num_dims):

            s["dW"+str(i)] = (beta * s["dW"+str(i)]) + ((1-beta)*(np.square(gradients["dW"+str(i)])))
            s["db"+str(i)] = (beta * s["db"+str(i)]) + ((1-beta)*(np.square(gradients["db"+str(i)])))
            
            parameters["W"+str(i)] = parameters["W"+str(i)] - ( alpha * (gradients["dW"+str(i)]/(np.sqrt(s["dW"+str(i)]+epsilon))) )
            parameters["b"+str(i)] = parameters["b"+str(i)] - ( alpha * (gradients["db"+str(i)]/(np.sqrt(s["db"+str(i)]+epsilon))) )
    
    elif optimization_method == 'adam':
        
        for i in range(1, num_dims):
            
            v["dW"+str(i)] = (gamma * v["dW"+str(i)]) + ((1-gamma) * gradients["dW"+str(i)])
            v["dW"+str(i)] = (v["dW"+str(i)]) / (1-(math.pow(gamma,iteration)))
            v["db"+str(i)] = (gamma * v["db"+str(i)]) + ((1-gamma) * gradients["db"+str(i)])
            v["db"+str(i)] = (v["db"+str(i)]) / (1-(math.pow(gamma,iteration)))

            s["dW"+str(i)] = (beta * s["dW"+str(i)]) + ((1-beta)*(np.square(gradients["dW"+str(i)])))
            s["dW"+str(i)] = (s["dW"+str(i)]) / (1-(math.pow(beta,iteration)))
            s["db"+str(i)] = (beta * s["db"+str(i)]) + ((1-beta)*(np.square(gradients["db"+str(i)])))
            s["db"+str(i)] = (s["db"+str(i)]) / (1-(math.pow(beta,iteration)))

            parameters["W"+str(i)] = parameters["W"+str(i)] - ( alpha * (v["dW"+str(i)]/(np.sqrt(s["dW"+str(i)]+epsilon))) )
            parameters["b"+str(i)] = parameters["b"+str(i)] - ( alpha * (v["db"+str(i)]/(np.sqrt(s["db"+str(i)]+epsilon))) )

    return parameters, v, s, alpha

def one_hot(Y):
    oh = np.zeros((int(Y.max()+1), Y.shape[1]))
    oh[Y.astype(int), np.arange(Y.shape[1])] = 1
    return oh


def multi_layer_network(train_x, train_y, net_dims, algorithm, num_iterations=500, learning_rate=0.2, gamma=0.9, beta=0.999, NAG_coeff=0.999, mini_batch_size=256, num_of_epochs=10, decay_rate=0.00):
  
    parameters, v, s = initialize_multilayer_weights(net_dims)
    costs = []
    epoch_count = 0

    for ec in range(num_of_epochs):

        indices = [i for i in range(train_x.shape[1])]
        random.shuffle(indices)

        iteration = 0
        i = 0
        while i < train_x.shape[1] - mini_batch_size:
      
            process_indices = indices[i:i+mini_batch_size]
            A0 = train_x[:, process_indices]
            y_hat = train_y[:, process_indices]

            AL, caches = multi_layer_forward(A0, parameters)
            A3, cache, cost = softmax_cross_entropy_loss(AL, y_hat)
          
            dAL = softmax_cross_entropy_loss_der(y_hat, cache)
            gradients = multi_layer_backward(dAL, caches, parameters)
           
            parameters, v, s, alpha = update_parameters(A0, y_hat, parameters, gradients, ec, iteration+1, learning_rate, gamma, beta, NAG_coeff, len(net_dims), v, s, decay_rate, algorithm)
            
            if iteration % 1 == 0:
                costs.append(cost)

            if iteration % 1 == 0:
                print(cost)
                print(alpha)
                print("Cost at iteration %i is: %.05f, learning rate: %.05f" %(iteration, cost, alpha))

            i = i + mini_batch_size
            iteration = iteration + 1

        print("one epoch done", ec)
    
    return costs, parameters

def main():
   
    net_dims = ast.literal_eval( sys.argv[1] )
    net_dims.append(10) 
    print("Network dimensions are:" + str(net_dims))

    train_data, train_label, test_data, test_label = \
            mnist(noTrSamples=50000,noTsSamples=5000,\
            digit_range=[0,1,2,3,4,5,6,7,8,9],\
            noTrPerClass=5000, noTsPerClass=500)

    learning_rate = 0.001
    num_iterations = 500
    mini_batch_size = 512
    gamma = 0.9
    beta = 0.999
    NAG_coeff = 0.999
    algorithm = ['classical', 'momentum', 'rmsprop', 'adam', 'NAG']
    print(algorithm[4])

    train_costs, parameters = multi_layer_network(train_data, train_label, net_dims, algorithm[4], num_iterations, learning_rate, gamma, beta, NAG_coeff, mini_batch_size)

    train_pred = classify(train_data, parameters)
    test_pred = classify(test_data, parameters)

    trAcc = (np.count_nonzero(train_pred == train_label) / train_data.shape[1]) * 100
    teAcc = (np.count_nonzero(test_pred == test_label) / test_data.shape[1]) * 100
    
    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
    print("Cost for training set is {0:0.3f} ".format(train_costs[len(train_costs)-1]))

    plt.title("training costs; learning rate="+str(learning_rate))
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.plot(list(range(len(train_costs))), train_costs, label='training cost')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()