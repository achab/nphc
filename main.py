from nphc.utils.cumulants import Cumulants
from nphc.utils.loader import load_data
from scipy.linalg import inv, qr, sqrtm
from itertools import product
import tensorflow as tf
import numpy as np


def starting_point(cumulants,random=False):
    d = cumulants.dim
    sqrt_C = sqrtm(cumulants.C)
    sqrt_L = np.sqrt(cumulants.L)
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

def NPHC(list_cumulants, initial_point, alpha=.5, training_epochs=1000, learning_rate=1e6, optimizer='momentum', \
         display_step = 100, l_l1=0., l_l2=0.):

    cumulants = list_cumulants[0]

    d = cumulants.dim

    R0 = tf.constant(initial_point.astype(np.float32), shape=[d,d])

    L = tf.placeholder(tf.float32, d, name='L')
    C = tf.placeholder(tf.float32, (d,d), name='C')
    K_c = tf.placeholder(tf.float32, (d,d), name='K_c')

    R = tf.Variable(R0, name='R')

    # Set weight matrices
    W_2 = np.ones((d,d))
    W_3 = np.ones((d,d))

    # Construct model
    activation_3 = tf.matmul(C,tf.square(R),transpose_b=True) + 2.0*tf.matmul(R,tf.mul(R,C),transpose_b=True) - 2.0*tf.matmul(R,tf.matmul(tf.diag(L),tf.square(R),transpose_b=True))
    activation_2 = tf.matmul(R,tf.matmul(tf.diag(L),R,transpose_b=True))

    #cost_3 = tf.reduce_mean( tf.squared_difference( activation_3, K_c ) )
    #cost_2 = tf.reduce_mean( tf.squared_difference( activation_2, C ) )

    cost =  (1-alpha) * tf.reduce_mean( tf.squared_difference( activation_3, K_c ) ) + alpha * tf.reduce_mean( tf.squared_difference( activation_2, C ) )

    reg_l1 = tf.contrib.layers.l1_regularizer(l_l1)
    reg_l2 = tf.contrib.layers.l2_regularizer(l_l2)
    if l_l1*l_l2 > 0:
        cost = tf.cast(cost, tf.float32) + reg_l1(R) + reg_l2(R)
    else:
        cost = tf.cast(cost, tf.float32)

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
    init = tf.initialize_all_variables()

    # Create a summary to monitor cost function
    #tf.scalar_summary('loss', cost)

    # Merge all summaries to a single operator
    #merged_summary_op = tf.merge_all_summaries()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Set logs writer into folder /tmp/tf_cumul
        #summary_writer = tf.train.SummaryWriter('/tmp/tf_cumul', graph=sess.graph)

        # Training cycle
        for epoch in range(training_epochs):

            cumulants = np.random.choice(list_cumulants)

            if epoch % display_step == 0:
                avg_cost = sess.run(cost, feed_dict={L: cumulants.L, C: cumulants.C, K_c: cumulants.K_c})
                #avg_cost_3 = sess.run(tf.nn.l2_loss(tf.gradients(cost_3, R)), feed_dict={L: cumulants.L, C: cumulants.C, K_c: cumulants.K_c})
                #avg_cost_2 = sess.run(tf.nn.l2_loss(tf.gradients(cost_2, R)), feed_dict={L: cumulants.L, C: cumulants.C})
                print("Epoch:", '%04d' % (epoch), "log10(cost)=", "{:.9f}".format(np.log10(avg_cost)))
                #print("       log10(cost3)=", "{:.9f}".format(np.log10(avg_cost_3)))
                #print("       log10(cost2)=", "{:.9f}".format(np.log10(avg_cost_2)))
            # Fit training using batch data
            sess.run(optimizer, feed_dict={L: cumulants.L, C: cumulants.C, K_c: cumulants.K_c})
            # Write logs at every iteration
            #summary_str = sess.run(merged_summary_op, feed_dict={L: cumul.L, C: cumul.C, K_c: cumul.K_c})
            #summary_writer.add_summary(summary_str, epoch)

        print("Optimization Finished!")

        return sess.run(R)

'''
Run the command line: tensorboard --logdir=/tmp/tf_cumul
Open http://localhost:6006/ into your web browser
'''

if __name__ == '__main__':

    # Load Cumulants object
    kernel = 'exp_d10'
    mode = 'nonsym_1'
    log10T = 10
    url = 'https://s3-eu-west-1.amazonaws.com/nphc-data/{}_{}_log10T{}_with_params_without_N.pkl.gz'.format(kernel, mode, log10T)
    cumul, Alpha, Beta, Gamma = load_data(url)

    # Params
    learning_rate = 1e7
    training_epochs = 1000
    display_step = 100
    d = cumul.dim

    _, s, _ = np.linalg.svd(cumul.C)
    lbd_max = s[0]
    initial = tf.constant(inv(np.eye(d) - cumul.C / (1.1*lbd_max)).astype(np.float32), shape=[d,d])

    out_1 = NPHC(cumul, initial, alpha=.99, learning_rate=learning_rate)
    print("First step is done!")

    out_2 = NPHC(cumul, out_1, alpha=.5, learning_rate=learning_rate)
    print("Second step is done!")
