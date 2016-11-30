import numpy as np
import tensorflow as tf
from itertools import chain
from sklearn import cross_validation, metrics
from NAM import NAM
from preprocess import read_task_data, convert_to_vector

# ######################################### Environment Setup ###########################
# Tensor Flow flags
tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state= NOne => Initialize weights randomly")
tf.flags.DEFINE_string("data_dir", "data/en/", "Directory containing bAbI tasks")
PARAMS = tf.flags.FLAGS

print("Training on Data : ", PARAMS.task_id)


# ############################## Pre-processing of Data #####################################
# Reading Train and Test data from data directory (for Hindi choose data/hn)
train, test = read_task_data(PARAMS.data_dir, PARAMS.task_id)
data = train + test

# Creating vocabulary/Dictionary for whole data(Train + Test)
vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

# Getting information data
max_story_size = max(map(len, (s for s, _, _ in data)))
mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
query_size = max(map(len, (q for _, q, _ in data)))
memory_size = min(PARAMS.memory_size, max_story_size)
vocab_size = len(word_idx) + 1
sentence_size = max(query_size, sentence_size)
print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)

# Converting Train and Test data sentences into equivalent vector representation based on dictionary
# S - Story, Q - Question, A - Answer, val_ - Validation set
S, Q, A = convert_to_vector(train, word_idx, sentence_size, memory_size)
trainS, valS, trainQ, valQ, trainA, valA = cross_validation.train_test_split(S, Q, A, test_size=.1, random_state=PARAMS.random_state)
testS, testQ, testA = convert_to_vector(test, word_idx, sentence_size, memory_size)
print(testS[0])
print("Training set shape", trainS.shape)

# Shape of vectorized data(Story, Question, Answer)
n_train = trainS.shape[0]
n_test = testS.shape[0]
n_val = valS.shape[0]
print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

# Getting maximum value
train_labels = np.argmax(trainA, axis=1)
test_labels = np.argmax(testA, axis=1)
val_labels = np.argmax(valA, axis=1)

# ########################################## Training Model #################################
# Random initialization of weights and specifying static details like batch size and optimizer to be used
tf.set_random_seed(PARAMS.random_state)
batch_size = PARAMS.batch_size
optimizer = tf.train.AdamOptimizer(learning_rate=PARAMS.learning_rate, epsilon=PARAMS.epsilon)

# Creating batches from data
batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]

# Learning start from here
with tf.Session() as sess:
    # Using soft Neural Attention Model(NAM)
    model = NAM(batch_size, vocab_size, sentence_size, memory_size, PARAMS.embedding_size, session=sess,
                   hops=PARAMS.hops, max_grad_norm=PARAMS.max_grad_norm, optimizer=optimizer)

    # Fitting model by minimizing Cross Entropy error
    for t in range(1, PARAMS.epochs+1):
        total_cost = 0.0
        for start, end in batches:
            s = trainS[start:end]
            q = trainQ[start:end]
            a = trainA[start:end]
            cost_t,summary = model.learn_fit_for_batch(s, q, a)
            total_cost += cost_t
           # train_writer.add_summary(summary, t)

        # Answer prediction for given Story(s) and Question(q) from training data by model
        if t % PARAMS.evaluation_interval == 0:
            train_preds = []
            for start in range(0, n_train, batch_size):
                end = start + batch_size
                s = trainS[start:end]
                q = trainQ[start:end]
                prediction = model.predict(s, q)
                train_preds += list(prediction)

            # Checking accuracy of model
            val_prediction = model.predict(valS, valQ)
            train_accuracy = metrics.accuracy_score(np.array(train_preds), train_labels)
            val_accuracy = metrics.accuracy_score(val_prediction, val_labels)

            print('-----------------------')
            print('Epoch', t)
            print('Total Cost:', total_cost)
            print('Training Accuracy:', train_accuracy)
            print('Validation Accuracy:', val_accuracy)
            print('-----------------------')

    # Saving learned Model
    model.saveModel()

    # Accuracy on test data
    test_prediction = model.predict(testS, testQ)
    test_accuracy = metrics.accuracy_score(test_prediction, test_labels)
    print("Testing Accuracy:", test_prediction)
