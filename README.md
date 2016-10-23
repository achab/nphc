test

# NPHC: Non Parametric Hawkes with Cumulants
## Compute cumulants from a list of point processes

With *N* a list of *d* arrays corresponding to the ticks of the *d* processes, and *H* half the size of the support of the truncated covariance density, one instantiates the *Cumulants* object this way
```python
from nphc.utils.cumulants import Cumulants
cumul = Cumulants(N, hMax=H)
```
Then, the easier way to compute the integrated cumulants *C* and *K^c* is done via the following line.
```python
cumul.set_all()
```
If you want to compute the cumulants for a new value of *H=H_* (to find the one that minimizes the estimation error for instance), run the line
```python
cumul.set_all(H_)
```

## Minimize NPHC objective function

The optimization of our non-convex problem is now done using the library *tensorflow*.
This choice enables us using all their SGD-like solvers (SGD, AdaGrad, AdeDelta, Adam, etc...), and writing our whole matching problem on **second** and **third** integrated cumulants in **four lines**, and **without writing any gradient!**
```python
# Construct model
activation_3 = tf.matmul(R*R,C,transpose_b=True) + tf.matmul(2*R*C,R,transpose_b=True) - tf.matmul(2*R*R,tf.matmul(tf.diag(L),R,transpose_b=True))
activation_2 = tf.matmul(R,tf.matmul(tf.diag(L),R,transpose_b=True))
# Minimize error
cost = (1.-alpha)*tf.reduce_mean(tf.square(activation_3 - K_c)) + alpha*tf.reduce_mean(tf.square(activation_2 - C))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
```


## Good starting point

The choice of the starting point is a paramount issue for the optimization step.
One can prove that the optimal value writes ``R = C^{1/2} M L^{-1/2}``, M being an orthogonal matrix.
Use the function ``nphc.main.starting_point``to select such a starting point where M equals either a random orthogonal matrix, or the identity matrix. 
```python
from nphc.main import starting_point
R0_random = starting_point(cumul,random=True)
R0_deterministic = starting_point(cumul,random=False)
```
This starting point corresponds to the optimal solution of the problem for ```alpha=1.```, up to an orthogonal matrix.

## Example of prediction:

![Example](http://i.imgur.com/44M8qct.png?1)
