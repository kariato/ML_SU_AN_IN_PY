#GRADIENTDESCENT Performs gradient descent to learn theta
import numpy as np
from computeCost import computeCost
def gradientDescent(X, y, theta, alpha, num_iters):
    #function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

    #   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
    #   taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = len(y);     # number of training examples
    J_history = np.zeros((num_iters, 1))

    for iter in range(1,num_iters):

        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta. 
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        x1 = X.dot(theta)
        x1 = np.subtract(x1,y)
        x1 = x1.T
        x1 = x1.dot(X)
        x1 = (alpha/m) * x1
        x1 = x1.T
        theta = np.subtract(theta,x1)

        #theta = theta - ((alpha/m) * ((X * theta) - y).T * X).T
        J_history[iter] = computeCost(X, y, theta)
        if iter>2 and J_history(iter) >=  J_history(iter-1):
            print("Bang")
            raise Exception('bang')
        #end
    


        # ============================================================

        # Save the cost J in every iteration    
        J_history[iter] = computeCost(X, y, theta);

    #end

    return theta, J_history

#end
