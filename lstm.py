#--------------------------------------------------------
#lstm.py
#lstm code to check the accuracy on babi dataset.
#-------------------------------------------------------

#---------- import necessary packages------------------
from itertools import chain
from sklearn import cross_validation, metrics

import numpy as np
import tensorflow as tf
from NAM import NAM

# These two modules are used to load the task and vectorize the data
from preprocess import read_task_data, convert_to_vector

sess = tf.InteractiveSession()

# define necessary flags to be used later in program
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/en/", "Directory containing bAbI tasks")
FLAGS = tf.flags.FLAGS

#----------------------------------------------------
#train set examples
#train[0]
#[['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway']], ['where', 'is', 'mary'], ['bathroom']) 
#train[1]
#([['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway'], ['daniel', 'went', 'back', 'to', 'the', 'hallway'], ['sandra', 'moved', 'to', 'the', 'garden']], ['where', 'is', 'daniel'], ['hallway'])

# load test and training data from the files
train, test = read_task_data(FLAGS.data_dir, FLAGS.task_id)
data = train + test

# vocab
#['back', 'bathroom', 'bedroom', 'daniel', 'garden', 'hallway', 'is', 'john', 'journeyed', 'kitchen', 'mary', 'moved', 'office', 'sandra', 'the', 'to', 'travelled', 'went', 'where']
vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))

#word_idx mapping of vocab words
#{'hallway': 6, 'bathroom': 2, 'garden': 5, 'journeyed': 9, 'office': 13, 'is': 7, 'sandra': 14,..}
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

# extract necesssary information from data
max_story_size = max(map(len, (s for s, _, _ in data)))
mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
query_size = max(map(len, (q for _, q, _ in data)))
memory_size = min(FLAGS.memory_size, max_story_size)
vocab_size = len(word_idx) + 1 # +1 for nil word
sentence_size = max(query_size, sentence_size) # for the position

# train/validation/test sets
S, Q, A = convert_to_vector(train, word_idx, sentence_size, memory_size)
trainS, valS, trainQ, valQ, trainA, valA = cross_validation.train_test_split(S, Q, A, test_size=.1, random_state=FLAGS.random_state)
testS, testQ, testA = convert_to_vector(test, word_idx, sentence_size, memory_size)

# ------------------------------------------------------
# network parameters
out_dim = len(trainA[0])
input_size = len(trainS[0,0])
n_inputs = (len(trainS[0,:,0]) + 1)
lstm_size = 200
m = len(trainS)
batch_size = 100
mval = len(valS)
mtest = len(testS)
#------------------------------------------------------
# generate input to match the rnn input layer dimension
X = np.zeros((m, n_inputs, input_size))
for i in range(m):
    X[i] = np.vstack([trainS[i], trainQ[i]])

Xval = np.zeros((mval, n_inputs, input_size))
for i in range(mval):
    Xval[i] = np.vstack([valS[i], valQ[i]])

Xtest = np.zeros((mtest, n_inputs, input_size))
for i in range(mtest):
    Xtest[i] = np.vstack([testS[i], testQ[i]])
#------------------------------------------------------
# data input to lstm network
x = tf.placeholder(tf.float32, \
    [None, n_inputs, input_size])
y = tf.placeholder(tf.float32, [None, out_dim])

#-------------------------------------------------------
# defining lstm cells in tensorflow
LSTMcell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, \
        state_is_tuple = True)

states = LSTMcell.zero_state(batch_size, tf.float32)

outputs, states = tf.nn.dynamic_rnn(LSTMcell, x, \
         initial_state=states)
#-----------------------------------------------------
# define weights in the network from hidden to output layer
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

Who = weight_variable([lstm_size, out_dim])
bho = bias_variable([out_dim])
#---------------------------------------------------
# from final output state of lstm, find output
yo = tf.matmul(states[1], Who) + bho

# find cross entropy loss
cross_entropy = tf.reduce_mean(\
        tf.nn.softmax_cross_entropy_with_logits(yo, y))
# minimize the cross entropy loss
train_step = tf.train.AdamOptimizer(1e-4).\
        minimize(cross_entropy)

# check how much accuracy is obtained
correct_prediction = tf.equal(tf.argmax(yo,1), \
        tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,\
        tf.float32))

# initialize all tensorflow variables
sess.run(tf.initialize_all_variables())

# deine number of Iterations for training
iterations = 10000
j =0

# start training the network
for i in range(iterations):
    for j in range(0,m,batch_size):
    # extract batch from training data
        x_batch = X[j:j+batch_size]
        y_batch = trainA[j:j+batch_size]
    # train on batch
        train_step.run(feed_dict={x: x_batch, y: y_batch})
    # check train and validation set accuracy
    if (0 == i%200):
        train_accuracy = accuracy.eval(feed_dict={\
            x: x_batch, y:y_batch})
        print("iteration: %d, train accuracy: %f"\
            %(i, train_accuracy))
        val_accuracy = accuracy.eval(feed_dict={\
            x: Xval, y:valA})
        print ("val accuracy %f"%val_accuracy)
    # randomly permute the data
    perm = np.arange(m)
    np.random.shuffle(perm)
    X = X[perm]
    trainA = trainA[perm]

# after training is over, find test accuracy
for j in range(0, mtest, batch_size):
    test_accuracy = accuracy.eval(feed_dict={\
            x: Xtest[j:j+batch_size], \
            y:testA[j:j+batch_size]})
    print ("test accuracy %f"%test_accuracy)

