from nphc.cumulants import Cumulants
from nphc.utils.loader import load_data
from scipy.linalg import inv, qr, sqrtm, norm
from itertools import product
import tensorflow as tf
import numpy as np


def starting_point(cumulants_list,random=False):
    L_list, C_list, K_c_list = cumulants_list
    d = len(L_list[0])
    sqrt_C = sqrtm(np.mean(C_list,axis=0))
    sqrt_L = np.sqrt(np.mean(L_list,axis=0))
    if random:
        M = random_orthogonal_matrix(d)
    else:
        M = np.eye(d)
    initial = np.dot(np.dot(sqrt_C,M),np.diag(1./sqrt_L))
    return initial

def random_orthogonal_matrix(dim):
    M = np.random.rand(dim**2).reshape(dim, dim)
    Q, _ = qr(M)
    return Q


class NPHC(object):
    """
    A class that implements non-parametric estimation described in th paper
    `Uncovering Causality from Multivariate Hawkes Integrated Cumulants` by
    Achab, Bacry, Gaiffas, Mastromatteo and Muzy (2016, Preprint).


    Methods
    -------

        fit : compute the Cumulants using function from `cumulants.py`

        solve : minimize the objective function


    Attributes
    ----------

        L : list of `np.array` shape=(dim,)
            Estimated means

        C : list of `np.array` shape=(dim,dim)
            Estimated covariance

        K : list of `np.array` shape=(dim,dim)
            Estimated skewness (sliced)

        R : `np.array` shape=(dim,dim)
            Parameter of interest, linked to the integrals of Hawkes kernels
    """

    def __init__(self):

        object.__init__(self)

        # we will store here the optimal cost reached
        self.optcost = None

    def fit(self, realizations=[], half_width=100., filtr='rectangular', method="parallel", mu_true=None, R_true=None):
        """
        Set the corresponding realization(s) of the process.
        Compute the cumulants.

        Parameters
        ----------

            realizations : `list`
                * Either a single realization as a list of np_arrays each representing
                the time stamps of a node of the Hawkes process
                * Or a list of realizations represented as above.

        """
        if all(isinstance(x,list) for x in realizations):
            self.realizations = realizations
        else:
            self.realizations = [realizations]

        cumul = Cumulants(realizations, half_width=half_width)
        cumul.mu_true = mu_true
        cumul.R_true = R_true
        cumul.compute_cumulants(half_width,filtr=filtr,method=method,sigma=half_width/5.)

        self.L = cumul.L.copy()
        self.C = cumul.C.copy()
        self.K_c = cumul.K_c.copy()
        if R_true is not None and mu_true is not None:
            self.L_th = cumul.L_th
            self.C_th = cumul.C_th
            self.K_c_th = cumul.K_c_th
        else:
            self.L_th = None
            self.C_th = None
            self.K_c_th = None


    def solve(self, alpha=-1, l_l1=0., l_l2=0., initial_point=None, training_epochs=1000, learning_rate=1e6, optimizer='momentum', \
         display_step = 100, use_average=False, use_projection=False, projection_stable_G=False, positive_baselines=False, l_mu=0.):
        """

        Parameters
        ----------

            training_epochs : `int`
                The number of training epochs.

            learning_rate : `float`
                The learning rate used by the optimizer.

            optimizer : `str`
                The optimizer used to minimize the objective function. We use optimizers from TensorFlow.
        """

        if use_projection:
            self.alpha = 0.
        elif alpha == -1:
            self.alpha = 1./(1. + (norm(np.mean([C for C in self.C],axis=0))**2) / (norm(np.mean([K_c for K_c in self.K_c],axis=0))**2) )
        else:
            self.alpha = alpha

        self.l_l1 = l_l1
        self.l_l2 = l_l2

        cumulants_list = [self.L, self.C, self.K_c]
        d = len(self.L[0])
        if initial_point is None:
            start_point = starting_point(cumulants_list, random=False)
        else:
            start_point = initial_point.copy()

        R0 = tf.constant(start_point.astype(np.float64), shape=[d,d])
        L = tf.placeholder(tf.float64, d, name='L')
        C = tf.placeholder(tf.float64, (d,d), name='C')
        K_c = tf.placeholder(tf.float64, (d,d), name='K_c')

        R = tf.Variable(R0, name='R', dtype=tf.float64)

        #I = tf.diag(tf.ones(d,dtype=tf.float64))
        I = tf.Variable(initial_value=np.eye(d), dtype=tf.float64)

        # Construct model
        activation_3 = tf.matmul(C,tf.square(R),transpose_b=True) + 2.0*tf.matmul(R,R*C,transpose_b=True) \
                       - 2.0*tf.matmul(R,tf.matmul(tf.diag(L),tf.square(R),transpose_b=True))
        activation_2 = tf.matmul(R,tf.matmul(tf.diag(L),R,transpose_b=True))

        cost =  (1-self.alpha) * tf.reduce_mean( tf.squared_difference( activation_3, K_c ) ) \
        + self.alpha * tf.reduce_mean( tf.squared_difference( activation_2, C ) )

        reg_l1 = tf.contrib.layers.l1_regularizer(self.l_l1)
        reg_l2 = tf.contrib.layers.l2_regularizer(self.l_l2)

        if (self.l_l2 * self.l_l1 > 0):
            cost = tf.cast(cost, tf.float64) + reg_l1((I - tf.matrix_inverse(R))) + reg_l2((I - tf.matrix_inverse(R)))
        elif (self.l_l1 > 0):
            cost = tf.cast(cost, tf.float64) + reg_l1((I - tf.matrix_inverse(R)))
        elif (self.l_l2 > 0):
            cost = tf.cast(cost, tf.float64) + reg_l2((I - tf.matrix_inverse(R)))
        else:
            cost = tf.cast(cost, tf.float64)

        # always use the average cumulants over all realizations
        if use_average or use_projection or projection_stable_G or positive_baselines:
            L_avg = np.mean(self.L, axis=0)
            C_avg = np.mean(self.C, axis=0)
            K_avg = np.mean(self.K_c, axis=0)
        if use_projection:
            L_avg_sqrt = np.sqrt(L_avg)
            L_avg_sqrt_inv = 1./L_avg_sqrt
            from scipy.linalg import inv, sqrtm
            C_avg_sqrt = sqrtm(C_avg)
            C_avg_sqrt_inv = inv(C_avg_sqrt)
        if projection_stable_G or positive_baselines:
            from scipy.linalg import inv
            C_avg_inv = inv(C_avg)

        if positive_baselines:
            #neg_baselines = - tf.matmul(tf.matmul(np.diag(L_avg),R,transpose_b=True),\
            #                            np.dot(C_avg_inv,L_avg.reshape(d,1)))
            neg_baselines = - tf.matmul(tf.matrix_inverse(R), L_avg.reshape(d,1))
            cost += l_mu * tf.reduce_sum(tf.nn.relu(neg_baselines))

        if optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(cost)
        elif optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        elif optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)
        elif optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
        elif optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        # Initialize the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            # Set logs writer into folder /tmp/tf_cumul
            #summary_writer = tf.train.SummaryWriter('/tmp/tf_cumul', graph=sess.graph)

            # Training cycle
            for epoch in range(training_epochs):

                print("neg baselines: ", tf.reduce_sum(tf.nn.relu()))
                    
                if epoch % display_step == 0:
                    avg_cost = np.average([sess.run(cost, feed_dict={L: L_, C: C_, K_c: K_c_})
                                           for (L_, C_, K_c_) in zip(self.L, self.C, self.K_c)])
                    print("Epoch:", '%04d' % (epoch), "log10(cost)=", "{:.9f}".format(np.log10(avg_cost)))

                if use_average:
                    sess.run(optimizer, feed_dict={L: L_avg, C: C_avg, K_c: K_avg})

                elif use_projection:
                    # Fit training using batch data
                    i = np.random.randint(0,len(self.realizations))
                    sess.run(optimizer, feed_dict={L: self.L[i], C: self.C[i], K_c: self.K_c[i]})
                    to_be_projected = np.dot(C_avg_sqrt_inv,np.dot(sess.run(R),np.diag(L_avg_sqrt)))
                    U, S, V = np.linalg.svd(to_be_projected)
                    R_projected = np.dot( C_avg_sqrt, np.dot( np.dot(U,V), np.diag(L_avg_sqrt_inv) ) )
                    assign_op = R.assign(R_projected)
                    sess.run(assign_op)
                else:
                    # Fit training using batch data
                    i = np.random.randint(0,len(self.realizations))
                    sess.run(optimizer, feed_dict={L: self.L[i], C: self.C[i], K_c: self.K_c[i]})

                if projection_stable_G:
                    to_be_projected = np.eye(d) - np.dot( np.dot(np.diag(L_avg), sess.run(tf.transpose(R))), C_avg_inv)
                    U, S, V = np.linalg.svd(to_be_projected)
                    S[S >= .99] = .99
                    G_projected = np.dot( U, np.dot(np.diag(S), V) )
                    R_projected = np.dot(C_avg, np.dot( np.eye(d) - G_projected.T, np.diag(1./L_avg) ) )
                    assign_op = R.assign(R_projected)
                    sess.run(assign_op)

                # Write logs at every iteration
                #summary_str = sess.run(merged_summary_op, feed_dict={L: cumul.L, C: cumul.C, K_c: cumul.K_c})
                #summary_writer.add_summary(summary_str, epoch)

            print("Optimization Finished!")

            return sess.run(R)


'''
Run the command line: tensorboard --logdir=/tmp/tf_cumul
Open http://localhost:6006/ into your web browser
'''
