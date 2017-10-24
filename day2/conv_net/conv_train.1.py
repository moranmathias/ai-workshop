import os
import sys
import argparse
import tensorflow as tf
from load_data import load_data
import preprocessing
import numpy as np

FLAGS = None
ROOT_PATH = "../../data/belgium_ts"
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")

def model(learning_rate):
    tf.reset_default_graph()
    session = tf.Session()

    # Setup placeholders, and reshape the data
    x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28], name="images")
    y = tf.placeholder(dtype = tf.int32, shape = [None], name="labels")
    
    flattened = tf.contrib.layers.flatten(x)

    w = tf.Variable(tf.truncated_normal([28 * 28, 62], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[62]), name="B")
    logits = tf.nn.relu(tf.matmul(flattened, w) + b)

    # Define a loss function
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                         logits = logits))
    # Define an optimizer 
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    correct_prediction = tf.argmax(logits, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    session.run(tf.global_variables_initializer())

    images, labels = load_data(train_data_directory)
    
    images28 = preprocessing.rgb2gray(preprocessing.resize(images))
    labels = np.array([l.index(1) for l in labels])
    
    test_images, test_labels = load_data(test_data_directory)
    
    test_images28 = preprocessing.rgb2gray(preprocessing.resize(test_images))
    test_labels = [l.index(1) for l in test_labels]
    for i in range(FLAGS.max_steps):
        if i % 500 == 0:
            [predicted] = session.run([correct_prediction], feed_dict={x: test_images28, y: test_labels})
            # Calculate correct matches 
            match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
            # Calculate the accuracy
            acc2 = match_count / len(test_labels)
            print("Epoch: ", i, " Test acc: ", acc2)
        session.run(train_step, feed_dict={x: images28, y: labels})

def make_hparam_string():
    return "conv_s1"

def main(_):
    for learning_rate in [1E-3]:
        hparam = make_hparam_string()
        print('Starting run for %s' % hparam)
        model(learning_rate)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=2001,
                      help='Number of steps to run trainer.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]])
