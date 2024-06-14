import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def GenerateData(shape,F,NoiseAmplitude = 0):
    np.random.seed(42)
    X = np.random.random(shape)
    y = F(X)
    Noise = np.random.normal(scale = NoiseAmplitude,size = y.shape)
    y= y + Noise
    return X,y

def VisualizeData2D(X,y,title=""):
    plt.tricontourf(X[:,0],X[:,1],y)
    plt.colorbar()
    plt.title(title)
    plt.show()

def VisualizeData3D(X,y,title=""):
    ax = plt.axes(projection="3d")
    img = ax.scatter3D(X[:,0],X[:,1],X[:,2],c = y)
    plt.title(title)
    plt.colorbar(img)
    plt.show()

def MSELoss(pred,y):
    n = y.shape[0]
    loss = pred-y
    return np.sqrt(loss@loss.T/n)
    

class RBF():
    """
    Define Radial basis function layer:
        Key points:
        (i) Centres of radial basis function are calculated using K-means clustering algorithm.
        (ii) Normalized radial function is used.
    """
    def __init__(self,nCentres):
        self.centres = None
        self.beta = None
        self.nCentres = nCentres

    def reset_parameters(self,X,beta):
        self.beta = beta
        kmeans = KMeans(self.nCentres,random_state=0).fit(X)
        self.centres = kmeans.cluster_centers_

    def forward(self,X):
        n = X.shape[0]
        m = self.centres.shape[0]
        out = np.zeros((n,m))
        for i in range(n):
            d = np.square(X[i,:]-self.centres)
            out[i,:] = np.exp(-np.sum(d,axis=1)*self.beta)
        
        norm = np.sum(out,axis = 1,keepdims=True)
        return out/norm
    

class Linear():
    """
    Define Linear Layer:
        Key points
        (i) Pseudoinverse solution for the linear weights i.e., 
        w = {(G^TG)}^{-1}G^T*y
        where G is the output of Radial basis function layer
        Note: Weights of linear layer are not updated with backproportion algorithm.
    """
    def __init__(self):
        self.weights = None
    
    def reset_parameter(self,phi,y):
        self.weights = np.linalg.inv(phi.T@phi)@phi.T@y

    def forward(self,phi):
        return phi@self.weights
    
class RBFNN():
    def __init__(self,nCentres):
        self.RBF = RBF(nCentres)
        self.Linear = Linear()

    def reset_parameter(self,X,y,beta):
        self.RBF.reset_parameters(X,beta)
        phi = self.RBF.forward(X)
        self.Linear.reset_parameter(phi,y)

    def forward(self,X):
        X = self.RBF.forward(X)
        X = self.Linear.forward(X)
        return X
        
def GridSearch(betas,nCentres,X,y,X_test,y_test):
    """
    Performing grid search on test data to get best hyperparameter (beta and number of centres)
    for our model.

    Note: The test data is used to achieve a generalized model and reduce overfitting.
    """
    optimal_beta = None
    optimal_nCentres = None
    min_loss = 100
    for beta in betas:
        for nCentre in nCentres:
            if(nCentre > X.shape[0]):
                continue
            model = RBFNN(nCentre)
            model.reset_parameter(X,y,beta)
            pred_test = model.forward(X_test)
            loss = MSELoss(pred_test,y_test)
            if(min_loss > loss):
                min_loss = loss
                optimal_beta = beta
                optimal_nCentres = nCentre  

    return  min_loss,optimal_beta, optimal_nCentres





