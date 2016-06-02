from utils.cumulants import Cumulants
from itertools import product
from utils.loader import load_data
import tensorflow as tf
import numpy as np

# Load Cumulants object
kernel = 'exp_d100'
mode = 'nonsym_1'
log10T = 10
url = 'https://s3-eu-west-1.amazonaws.com/nphc-data/{}_{}_log10T{}_with_params_without_N.pkl.gz'.format(kernel, mode, log10T)
cumul, Alpha, Beta, Gamma = load_data(url)

# Params
alpha = 1.
learning_rate = 1e5
training_epochs = 10
display_step = 10
d = cumul.dim

# tf Graph Input
L = tf.placeholder('float', d, name='L')
C = tf.placeholder('float', (d, d), name='C')
K_c = tf.placeholder('float', (d, d), name='K_c')

K_c_ij = tf.placeholder(tf.float32)
C_ij = tf.placeholder(tf.float32)
ind_i = tf.placeholder(tf.int32, shape=[2])
ind_j = tf.placeholder(tf.int32, shape=[2])

# Set model weight
#R = tf.Variable(tf.ones([d,d]), name='R')
#initial = tf.constant([[float(i+j*d)/(d**2) for i in range(d)] for j in range(d)], shape=[d,d])
#initial = tf.truncated_normal(shape=[d,d], stddev=0.1)
#R = tf.Variable(initial, name='R')
R = tf.get_variable('R',shape=[d,d],initializer=tf.contrib.layers.xavier_initializer())

# Construct model
C_ind_i = tf.placeholder('float', d, name='C_ind_i')
C_ind_j = tf.placeholder('float', d, name='C_ind_j')
R_ind_i = tf.slice(R, ind_i, [1,d])
R_ind_j = tf.slice(R, ind_j, [1,d])
act_3_ij = tf.reduce_sum(tf.mul(tf.mul(C_ind_j,R_ind_i),R_ind_i) + 2*R_ind_i*C_ind_i*R_ind_j - 2*L*R_ind_j*R_ind_i**2)
act_2_ij = tf.reduce_sum(tf.mul(tf.mul(L,R_ind_i),R_ind_j))

# Minimize error
activation_3 = tf.matmul(R*R,C,transpose_b=True) + tf.matmul(2*R*C,R,transpose_b=True) - tf.matmul(2*R*R,tf.matmul(tf.diag(L),R,transpose_b=True))
activation_2 = tf.matmul(R,tf.matmul(tf.diag(L),R,transpose_b=True))
cost = tf.reduce_mean(tf.square(activation_3 - K_c)) + alpha*tf.reduce_mean(tf.square(activation_2 - C))
sub_cost = tf.reduce_mean(tf.square(act_3_ij - K_c_ij)) + alpha*tf.reduce_mean(tf.square(act_2_ij - C_ij))
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.95).minimize(sub_cost)

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
        for (i, j) in product(range(d), repeat=2):
        # Fit training using batch data
            sess.run(optimizer, feed_dict={L: cumul.L, C_ij: cumul.C[i,j], K_c_ij: cumul.K_c[i,j], ind_i: [i,0], ind_j: [j,0], C_ind_i: cumul.C[i], C_ind_j: cumul.C[j]})
        #if epoch % display_step == 0:
            if j == 0:
                avg_cost = sess.run(cost, feed_dict={L: cumul.L, C: cumul.C, K_c: cumul.K_c})
                print("Epoch:", '%04d' % (epoch), "log10(cost)=", "{:.9f}".format(np.log10(avg_cost)))
        # Write logs at every iteration
        #summary_str = sess.run(merged_summary_op, feed_dict={L: cumul.L, C: cumul.C, K_c: cumul.K_c})
        #summary_writer.add_summary(summary_str, epoch)

    print("Optimization Finished!")

    import gzip, pickle
    f = gzip.open('out_sto.pkl.gz', 'wb')
    pickle.dump(sess.run(R),f,protocol=2)
    f.close()


'''
Run the command line: tensorboard --logdir=/tmp/tf_cumul
Open http://localhost:6006/ into your web browser
'''
