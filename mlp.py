#--------------------------------------------------------
#mlp.py
#multilayer neural network code to check the accuracy on babi dataset.
#-------------------------------------------------------

#---------- import necessary packages------------------
from itertools import chain
from sklearn import cross_validation, metrics
import numpy as np
import tensorflow as tf

# These two modules are used to load the task and vectorize the data
from preprocess import read_task_data, convert_to_vector

sess = tf.InteractiveSession()

# define necessary flags to be used later in program
tf.flags.DEFINE_integer("memory_size", 50, \
        "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, \
        1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, \
        "Random state.")
tf.flags.DEFINE_string("data_dir", \
        "data/en/", \
        "Directory containing bAbI tasks")
PARAMS = tf.flags.FLAGS

# load test and training data from the files
train, test = read_task_data(PARAMS.data_dir, PARAMS.task_id)
data = train + test

# vocab on data
vocab = sorted(reduce(lambda x, y: x | y, (set(list(\
        chain.from_iterable(s)) + q + a) \
        for s, q, a in data)))
#word_idx mapping of vocab words
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

# extract necesssary information from data
max_story_size = max(map(len, (s for s, _, _ in data)))
mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
sentence_size = max(map(len, chain.from_iterable(\
        s for s, _, _ in data)))
query_size = max(map(len, (q for _, q, _ in data)))
memory_size = min(PARAMS.memory_size, max_story_size)
vocab_size = len(word_idx) + 1
sentence_size = max(query_size, sentence_size) 

# train/validation/test sets
S, Q, A = convert_to_vector(train, word_idx, sentence_size,\
        memory_size)
trainS, valS, trainQ, valQ, trainA, valA = cross_validation\
        .train_test_split(S, Q, A, test_size=.1, \
        random_state=PARAMS.random_state)
testS, testQ, testA = convert_to_vector(test, word_idx, \
        sentence_size, memory_size)

# convert input data to feed into input layer of neural network
a = len(trainS)
b = len(trainS[0].flatten()) + len(trainQ[0])
X = np.zeros((a,b))

a = len(valS)
b = len(valS[0].flatten()) + len(valQ[0])
Xval = np.zeros((a,b))

a = len(testS)
b = len(testS[0].flatten()) + len(testQ[0])
Xtest = np.zeros((a,b))

# create new data set for input in ANN by merging S and Q
for i in range(len(trainS)):
    temp = trainS[i].flatten()
    X[i] = np.concatenate([temp, trainQ[i]])
for i in range(len(valS)):
    temp = valS[i].flatten()
    Xval[i] = np.concatenate([temp, valQ[i]])
for i in range(len(testS)):
    temp = testS[i].flatten()
    Xtest[i] = np.concatenate([temp, testQ[i]])

# create feed-forward neural network
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# define network structure
in_dim = len(X[0])
out_dim = len(trainA[0])
hid_dim = 2 * in_dim
train_size = len(X)

# define weights between input-hidden and hidden-output layer
Wh = weight_variable([in_dim, hid_dim])
bh = bias_variable([hid_dim])
Wo = weight_variable([hid_dim, out_dim])
bo = bias_variable([out_dim])

# define placeholder for the inputs
x = tf.placeholder(tf.float32, shape=[None, in_dim])
y = tf.placeholder(tf.float32, shape=[None, out_dim])

keep_prob = tf.placeholder(tf.float32)

# use sigmoid activation at the hidden layer
h1 = tf.sigmoid(tf.matmul(x, Wh) + bh)
# to avoid overfitting, using dropout at the hidden layer
h1_drop = tf.nn.dropout(h1, keep_prob)

# calculate linear output at the output layer
yo = tf.matmul(h1_drop, Wo) + bo

# find cross entropy loss at the output layer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yo, y))
# minimize the cross entropy loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# check how much accuracy is obtained
correct_prediction = tf.equal(tf.argmax(yo,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize all tensorflow variables
sess.run(tf.initialize_all_variables())

# deine number of iterations for training
iterations = 1000
j =0

# start training the network
for i in range(iterations):
    # train on whole input data
    train_step.run(feed_dict={x: X, y: trainA,keep_prob: 0.9})
    # check train and validation set accuracy
    if (0 == j%100):
        train_accuracy = accuracy.eval(feed_dict={\
            x: X, y:trainA, keep_prob: 1.0})
        print("iteration: %d, train accuracy: %f"\
            %(j, train_accuracy))
        val_accuracy = accuracy.eval(feed_dict={\
            x: Xval, y:valA, keep_prob: 1.0})
        print ("val accuracy %f"%val_accuracy)
    j += 1
    # randomly permute the data
    perm = np.arange(train_size)
    np.random.shuffle(perm)
    X = X[perm]
    trainA = trainA[perm]

# after training is over, find test accuracy
test_accuracy = accuracy.eval(feed_dict={x: Xtest, y:testA, keep_prob: 1.0})
print ("test accuracy %f"%test_accuracy)

