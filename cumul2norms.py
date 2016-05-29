from nphc.utils.cumulants import Cumulants
from nphc.utils.loader import load_data
import tensorflow as tf
import numpy as np

# Load Cumulants object
kernel = 'exp_d100'
mode = 'nonsym_1'
log10T = 9 
url = 'https://s3-eu-west-1.amazonaws.com/nphc-data/{}_{}_log10T{}_with_params_without_N.pkl.gz'.format(kernel, mode, log10T)
cumul, Alpha, Beta, Gamma = load_data(url)

# Params
alpha = 0.01
learning_rate = 1e0
training_epochs = 20000
display_step = 100
d = cumul.dim

# tf Graph Input
L = tf.placeholder('float', d, name='L')
C = tf.placeholder('float', (d, d), name='C')
K_c = tf.placeholder('float', (d, d), name='K_c')

# Set model weight
R = tf.Variable(tf.ones([d,d]), name='R')

# Construct model
activation_3 = tf.matmul(R*R,C,transpose_b=True) + tf.matmul(2*R*C,R,transpose_b=True) - tf.matmul(2*R*R,tf.matmul(tf.diag(L),R,transpose_b=True))
activation_2 = tf.matmul(R,tf.matmul(tf.diag(L),R,transpose_b=True))

# Minimize error
cost = tf.reduce_mean(tf.square(activation_3 - K_c)) + alpha*tf.reduce_mean(tf.square(activation_2 - C))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initialize the variables
init = tf.initialize_all_variables()

# Create a summary to monitor cost function
tf.scalar_summary('loss', cost)

# Merge all summaries to a single operator
merged_summary_op = tf.merge_all_summaries()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Set logs writer into folder /tmp/tf_cumul
    summary_writer = tf.train.SummaryWriter('/tmp/tf_cumul', graph=sess.graph)

    # Training cycle
    for epoch in range(training_epochs):
        # Fit training using batch data
        sess.run(optimizer, feed_dict={L: cumul.L, C: cumul.C, K_c: cumul.K_part})
        avg_cost = sess.run(cost, feed_dict={L: cumul.L, C: cumul.C, K_c: cumul.K_part})
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch), "log10(cost)=", "{:.9f}".format(np.log10(avg_cost)))
        # Write logs at every iteration
        summary_str = sess.run(merged_summary_op, feed_dict={L: cumul.L, C: cumul.C, K_c: cumul.K_part})
        summary_writer.add_summary(summary_str, epoch)

    print("Optimization Finished!")

    import gzip, pickle
    f = gzip.open('out.pkl.gz', 'wb')
    pickle.dump(sess.run(R),f,protocol=2)
    f.close()


'''
Run the command line: tensorboard --logdir=/tmp/tf_cumul
Open http://localhost:6006/ into your web browser
'''
