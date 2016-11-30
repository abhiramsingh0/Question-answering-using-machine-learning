import tensorflow as tf
import numpy as np


# ############### Functions use by soft Neural Attention Model(NAM) ####################
# Position is encoded, so we can use it later to get answer
def position_encoding(sentence_size, embedding_size):
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    embedding_len = embedding_size+1
    sentence_len = sentence_size+1
    
    for i in range(1, embedding_len):
        for j in range(1, sentence_len):
            encoding[i-1, j-1] = (i - (embedding_len-1)/2) * (j - (sentence_len-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)


# Append zero into tensor to match dimension
def match_dimension(tensor, name=None):
    with tf.op_scope([tensor], name, "zero_nil_slot") as name:
        tensor = tf.convert_to_tensor(tensor, name="t")
        shape = tf.shape(tensor)[1]
        zero_vector = tf.zeros(tf.pack([1, shape]))
        return tf.concat(0, [zero_vector, tf.slice(tensor, [1, 0], [-1, -1])], name=name)


# Added random noise to regularize temporal encoding
def add_gradient_noise(tensor, stddev=1e-3, name=None):
    with tf.op_scope([tensor, stddev], name, "add_gradient_noise") as name:
        tensor = tf.convert_to_tensor(tensor, name="t")
        gen_random = tf.random_normal(tf.shape(tensor), stddev=stddev)
        return tf.add(tensor, gen_random, name=name)


# Main soft Neural Attention Model(NAM) code start here
class NAM(object):
    def __init__(self, batch_size, vocab_size, sentence_size, memory_size, embedding_size,      
            hops=3, max_grad_norm=40.0, nonlin=None, initializer=tf.random_normal_initializer(stddev=0.1),
            optimizer=tf.train.AdamOptimizer(learning_rate=1e-2), encoding=position_encoding,
            session=tf.Session(), name='NAM'):

        # Model parameters
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.sentence_size = sentence_size
        self.memory_size = memory_size
        self.embedding_size = embedding_size
        self.hops = hops
        self.max_grad_norm = max_grad_norm
        self.nonlin = nonlin
        self.init = initializer
        self.opt = optimizer
        self.name = name
        self.buildInputs()
        self.buildVars()
        self._encoding = tf.constant(encoding(self.sentence_size, self.embedding_size), name="encoding")

        # Cross Entropy Error functions
        logits = self._inference(self.stories, self.queries)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(self._answers, tf.float32), name="cross_entropy")
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")
        calculated_loss = cross_entropy_sum
        self.saver = tf.train.Saver()

        gradients_and_variables = self.opt.compute_gradients(calculated_loss)
        gradients_and_variables = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g,v in gradients_and_variables]
        gradients_and_variables = [(add_gradient_noise(g), v) for g,v in gradients_and_variables]
        nil_gradients_and_variables = []
        
        for g, v in gradients_and_variables:
            if v.name in self._nil_vars:
                nil_gradients_and_variables.append((match_dimension(g), v))
            else:
                nil_gradients_and_variables.append((g, v))
        train_output = self.opt.apply_gradients(nil_gradients_and_variables, name="train_output")

        predict_output = tf.argmax(logits, 1, name="predict_output")
        predict_probability_output = tf.nn.softmax(logits, name="predict_probability_output")
        predict_log_probability_output = tf.log(predict_probability_output, name="predict_log_probability_output")

        self.calculated_loss = calculated_loss
        self.predict_output = predict_output
        self.predict_probability_output = predict_probability_output
        self.predict_log_probability_output = predict_log_probability_output
        self.train_output = train_output

        init_variables = tf.initialize_all_variables()

        self.sess = session
        self.sess.run(init_variables)

    # Story, Question and Answer into the tensor flow placeholder
    def buildInputs(self):
        self.stories = tf.placeholder(tf.int32, [None, self.memory_size, self.sentence_size], name="stories")
        self.queries = tf.placeholder(tf.int32, [None, self.sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None, self.vocab_size], name="answers")

    def buildVars(self):
        with tf.variable_scope(self.name):
            nil_word_slot = tf.zeros([1, self.embedding_size])
            A = tf.concat(0, [ nil_word_slot, self.init([self.vocab_size-1, self.embedding_size]) ])
            B = tf.concat(0, [ nil_word_slot, self.init([self.vocab_size-1, self.embedding_size]) ])
            self.A = tf.Variable(A, name="A")
            self.B = tf.Variable(B, name="B")

            self.TA = tf.Variable(self.init([self.memory_size, self.embedding_size]), name='TA')

            self.H = tf.Variable(self.init([self.embedding_size, self.embedding_size]), name="H")
            self.W = tf.Variable(self.init([self.embedding_size, self.vocab_size]), name="W")
        self._nil_vars = set([self.A.name, self.B.name])

    def _inference(self, stories, queries):
        with tf.variable_scope(self.name):
            q_emb = tf.nn.embedding_lookup(self.B, queries)
            u_0 = tf.reduce_sum(q_emb * self._encoding, 1)
            u = [u_0]
            for _ in range(self.hops):
                m_emb = tf.nn.embedding_lookup(self.A, stories)
                m = tf.reduce_sum(m_emb * self._encoding, 2) + self.TA

                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m * u_temp, 2)

                probs = tf.nn.softmax(dotted)
                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                c_temp = tf.transpose(m, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)

                u_k = tf.matmul(u[-1], self.H) + o_k

                # Use this, define a non-linear function
                # if self._nonlin:
                #     u_k = nonlin(u_k)

                u.append(u_k)

            return tf.matmul(u_k, self.W)

    # Fit each batch of data
    def learn_fit_for_batch(self, stories, queries, answers):
        feed_dict = {self.stories: stories, self.queries: queries, self._answers: answers}
        loss, _ = self.sess.run([self.calculated_loss, self.train_output], feed_dict=feed_dict)
        summary, _ = self.sess.run([self.calculated_loss, self.train_output], feed_dict=feed_dict)
        return loss,summary

    # Save the model
    def saveModel(self):
        save_path = self.saver.save(self.sess, "/tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)

    # Restore the model to use on test data
    def restoreModel(self):
        dummy_msg = 5
        #self.saver.restore(self.sess, "/tmp/model.ckpt")
        #print("Model restored.")

    # Predict answer
    def predict(self, stories, queries):
        feed_dict = {self.stories: stories, self.queries: queries}
        self.restoreModel();
        return self.sess.run(self.predict_output, feed_dict=feed_dict)

    # Predict probabilities
    def predict_prob(self, stories, queries):
        feed_dict = {self.stories: stories, self.queries: queries}
        return self.sess.run(self.predict_probability_output, feed_dict=feed_dict)

    # Predict log probabilities
    def predict_log_proba(self, stories, queries):
        feed_dict = {self.stories: stories, self.queries: queries}
        return self.sess.run(self.predict_log_probability_output, feed_dict=feed_dict)