from utils.cumulants import Cumulants
from itertools import product
from utils.loader import load_data
from scipy.linalg import inv
import tensorflow as tf
import numpy as np


def NPHC(cumulants, starting_point, alpha=.5, training_epochs=1000, learning_rate=1e6, optimizer='momentum', display_step = 10):

    d = cumulants.dim

    L = tf.placeholder('float', d, name='L')
    C = tf.placeholder('float', (d,d), name='C')
    K_c = tf.placeholder('float', (d,d), name='K_c')

    R = tf.Variable(starting_point, name='R')

    # Construct model
    activation_3 = tf.sub(tf.add(tf.matmul(tf.square(R),C,transpose_b=True), tf.matmul(tf.scalar_mul(2.0,tf.mul(R,C)),R,transpose_b=True)), tf.matmul(tf.scalar_mul(2.0,tf.square(R)),tf.matmul(tf.diag(L),R,transpose_b=True)))
    activation_2 = tf.matmul(R,tf.matmul(tf.diag(L),R,transpose_b=True))

    cost = tf.add( tf.scalar_mul( 1-alpha, tf.reduce_mean( tf.squared_difference( activation_3, K_c ) ) ), tf.scalar_mul( alpha, tf.reduce_mean( tf.squared_difference( activation_2, C ) ) ) )
    if optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.95).minimize(cost)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

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
            # Fit training using batch data
            sess.run(optimizer, feed_dict={L: cumulants.L, C: cumulants.C, K_c: cumulants.K_c})
            if epoch % display_step == 0:
                avg_cost = sess.run(cost, feed_dict={L: cumulants.L, C: cumulants.C, K_c: cumulants.K_c})
                print("Epoch:", '%04d' % (epoch), "log10(cost)=", "{:.9f}".format(np.log10(avg_cost)))
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
    kernel = 'exp_d100'
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
