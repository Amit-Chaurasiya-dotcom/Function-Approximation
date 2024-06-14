# Function-Approximation
We use a Radial Basis Function (RBF) neural network to predict the hypersurface of analytical functions. Initially, we predict a smooth harmonic function on 2D points, followed by predicting an exponential function on 3D input points. To evaluate the robustness of our model, we add AWGN noise with an amplitude of 0.1 to our data. A good model should be insensitive to this noise.
<br>

### Radial basis function network
<br>

**Architecture**
<br>
An input vector 
𝑥 is used as input to all radial basis functions, each with different parameters i.e, centres and beta. The output of the network is a linear combination of the outputs from radial basis functions. Normalized radial functions are used,as they give better result.
<br><br>

**Training:**
<br>
Training is done in two steps:
1. First, we update the centres of each radial basis function using K-means clustering algorithm.
<br>
2. Then, we update the weight of linear layer.Weights of linear layer are not updated with backproportion algorithm. Instead, Pseudoinverse solution for the linear weights are used to calculate it.

$$ w = {(G^TG)}^{-1}G^Ty$$

*where G is the output of Radial basis function layer*
<br>
<p>

**Grid Search**
<br>
Performing grid search on test data to get best hyperparameter (beta and number of centres) for our model.

*Note: The test data is used to achieve a generalized model and reduce overfitting.*



