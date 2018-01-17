"""
A radial basis function (RBF) is a real-valued function
whose value depends only on the distance from the origin,
or alternatively on the distance from some other point c, called a center.

1. Choose the centers randomly from the training set.
2. Compute the spread for the RBF functionusing the normalization method.
3. Find the weights using the pseudo-inverse method.

The basis functions are gaussians,
the output layer is linear and
the weights are learned by a simple pseudo-inverse.
Each RBF neuron applies a Gaussian to the input.

norm: A function that returns the 'distance' between two points, with
inputs as arrays of positions (x, y, z, ...), and an output as an
array of distance.

RBFA gets less effective the further you approximate from the bounds of the training set

@TODO
Find a way to autogenerate good multidimensional input matrices!
Tidy up imports. Maybe do that numpy as np thing
Tidy up variable names inconsistent use of capitalization
"""

from scipy import *
import numpy
from scipy.linalg import norm, pinv
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist

class RBF:
    """
    ndin      : The number of input dimensions, implied by the training set and consequently depreciated
    nCenters  : The number of neuronodes, each neurnode has a relative center
    ndout     : The number of output dimensions, implied by the training set and consequently depreciated
    """
    def __init__(self, nCenters):
        # chosen by pop culture reference, a truly random number and a placeholder, it'll probably work with 0 as the placeholder.. probably
        self.sigma = 0.007
        self.nCenters = nCenters
        self.Centers = []
        self.weights = []

    """
    https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
    http://www.di.unito.it/~cancelli/retineu06_07/RBF.pdf
    Calculates the spread parameter sigma for use by the basis function
    Some people reccomend making this a constant value and not bothering to calculate it
    It has surprisingly  little impact on the utility of the RBFA
    @ToDo figure out why print statements cause infinite loops here
    There's totally a slightly more efficient way to do this waiting to be found
    """
    def spreadFactor(self):
        sigma = numpy.max(pdist(self.Centers))
        sigma = self.nCenters / (sigma ** 2)
        return sigma
    """
    http://www.di.unito.it/~cancelli/retineu06_07/RBF.pdf
    The Gaussian Function is the chosen Radial Basis Function (RBF)
    c : the center ~ c, yay!
    d : the distance from the center
    sigma: the spread parameter
    @ToDo figure out why print statements cause infinite loops here
    """
    def basisFunction(self, c, d):
        # beta = 1/(2*sigma**2)
        # Gaussian Radial Basis Function f(x) = aexp(beta* x-b)**2)
        return exp(-self.sigma * norm(c - d) ** 2)

    """
    Calculates the threshold theta values of each neuronode relative to it's center
    X : the Matrix data to be approximated
    """
    def doThresholds(self, X):
        # calculate nCenters activation threshold values (thetas) of the RBF, one for each center
        theta = zeros((X.shape[0], self.nCenters), float)

        #for every center c in Centers calculate the basis function..
        #for every x example in the domain from the x=mgrid
        for ni in range(len(self.Centers)):
            n = self.Centers[ni]
            for xi in range(len(X)):
                x = X[xi]
                theta[xi, ni] = self.basisFunction(n, x)
        #Organize those values into the grand theta
        return theta

    """
    Training optimizes the weights to get the output as close as possible to the desired value.
    The desired output is just the output value associated with the training example.
    Training data needs an input Matrix X full of the input values the data
    Needs an already accurate output Matrix fu
    """
    def train(self, X, Y):
        # choose numcenters random center vectors from training set input set X
        randCenterIDs = random.permutation(X.shape[0])[:self.nCenters]

        self.Centers = [X[i, :] for i in randCenterIDs]
        self.sigma = self.spreadFactor()

        # calculate the threshold activation values (thetas) of RBFs, one for each center
        theta = self.doThresholds(X)

        # Matrix pseudoinverse to calculate weights
        self.weights = dot(pinv(theta), Y)

    """
    X needs to be  an input matrix, even in the 1 dimensionality case, ex: [[ 3.14]]
    This is where the function is approximated from what it learned while training
    """
    def predict(self, X):
        theta = self.doThresholds(X)
        Y = dot(theta, self.weights)
        return Y

    """
    Cost function
    Y   :   Output Results
    H   :   Output Expected
    """
    def cost_function(Y, H):
        J = (1 / len(H)) * (sum(numpy.multiply((numpy.square(Y - H)), 0.5)))
        return J

    """
    a = 1, b=100 is usually seemingly always how this works
    The Rosenbrock function with X as a Matrix Input
    Depreciated:
    Yt = (1-Xt[:,0])**2 + 100*(Xt[:,1] - Xt[:,0]**2)**2  + (1-Xt[:,1])**2 + 100*(Xt[:,2] - Xt[:,1]**2)**2
    Where each '+' is the addition of an input dimensi
    X : a NumPy matrix where each row is an input vector
    d : the n-dimensionality and also the length of each X row
    """
    def rosenbrock(X, d):
        Y = (1-X[:,0])**2 + 100*(X[:,1] - X[:,0]**2)**2
        i = 1
        while(i < d-1):
            Y += (1-X[:,i])**2 + 100*(X[:,i+1] - X[:,i]**2)**2
            i += 1
        return Y  

"""
Xt : Training input matrix
Yt : Training output matrix
Xp : input matrix to be Predicted
Yp : output matrix Prediction
Yy : genuine output matrix for Xp
nt : number of vectors to train with
np : number of vectorrs to predict on
ndin  : input dimensionality  - n dimensional input
ndout : output dimensionality - n dimensional output   
[beg,end] : very convenient for higher dimensional inputs to just stay square/cubic
beg       : beginning of domain for training set centers
end       : ending of domain for training set and centers

Putting center vectors outside the training domain might make the RBFA better at predicting values outside the training domain.. might
Consequently an offset parameter for the center vectors could be an interesting the to play around with
"""
if __name__ == '__main__':
    # Have nt >= ncenters
    nt = 800
    np = 400
    ndin = 2

    # Beginning and end of the domain's range, the domain will be cubic for convenience
    beg = 0.0
    end = 1.0
    # Maybe you want to try predicting outside of the training domain?
    # Get completely out by making offset > abs(beg-end)
    offset = 0.0
    
    # RBFA parameters, more centers -> more accuracy
    ndout = 1
    ncenters = 200
    
    """
    # This is 2D in to 1D out
    # 2D rosenbrock (a=1, b=100), plotting this is kind of pointless, 
    # I hope to find a more eloquent solution than randomized multidimensional vectors so that graphs may attain some sort of meaning
    Xt = beg + (abs(end-beg) * random.rand(nt,ndin))
    Yt = RBF.rosenbrock(Xt, ndin)
    Xp = beg + offset + (abs(end-beg) * random.rand(np,ndin))
    # Only useful for the cost function analysis
    Yy = RBF.rosenbrock(Xp, ndin)
    """
    # This is 1D in to 1D out for a cosine function
    
    # an input array of n evenly distributed values in the range [beg,end] inclusive
    Xt = mgrid[beg:end:complex(0, nt)].reshape(nt, 1)
    # Useful testing function that's all squiggly and symmetric about x=1
    Yt = cos(4 * (Xt - 1) ** 2 + 3) + 1.2
    Xp = mgrid[beg:end + offset:complex(0, np)].reshape(np, 1)
    # Only useful for the cost function
    Yy = cos(4 * (Xp - 1) ** 2 + 3) + 1.2
    

    #RBFA
    rbf = RBF(ncenters)
    rbf.train(Xt, Yt)
    Yyp = rbf.predict(Xp)
    Yyt = rbf.predict(Xt) 
    #print("The value(s) to be predicted (Xp):\n", Xp)
    #print("The predicted value(s) of Xp:\n", Yp)
    #print("The actual value(s) of Xp:\n", Yy)
    print("A measure of accuracy where lower is better. This is of the predicted set Cost(Yy):\n", RBF.cost_function(Yy,Yyp))
    print("A measure of accuracy where lower is better. This is of the training set Cost(Yy):\n", RBF.cost_function(Yt,Yyt))
    #print("A measure of accuracy where lower is better Cost(Yy):\n", cost_function(Yy,Yp))

    
    #plot it partially maybe, Cartesian style
    plt.figure(figsize=(12, 8))

    # plot original training data model (black)
    plt.plot(Xt[:,0], Yt[:,0], 'k-', linewidth=3)

    # plot learned predicted model (red)
    plt.plot(Xp[:,0], Yyp[:,0], 'r-', linewidth=2)

    #x bounds of plot, not necessaary?
    #plt.xlim(beg - 0.2, end + offset + 0.2)

    plt.show()
    
