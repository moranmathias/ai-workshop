import os
import sys
import argparse
import tensorflow as tf
import numpy as np
from load_data import load_data
import preprocessing
import random
from math import sqrt


FLAGS = None
ROOT_PATH = "../../data/belgium_ts"
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")

def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.matmul(input, w) + b
        tf.summary.histogram("weigths", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

def batch_generator(images, labels, batch_size):
    index = 0
    data = list(zip(images, labels))
    random.shuffle(data)
    while True:
        if index >= len(images):
            index = 0
            random.shuffle(data)

        partial = data[index: index+batch_size]
        yield ([x[0] for x in partial],[x[1] for x in partial])
        index += batch_size


def model(learning_rate, batch_size, hparam, use_two_fc):
    tf.reset_default_graph()
    session = tf.Session()

    # Setup placeholders, and reshape the data
    x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28], name="images")
    flattened = tf.contrib.layers.flatten(x)
    x_image = tf.reshape(flattened, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 3)
    y = tf.placeholder(dtype = tf.int32, shape = [None, 62], name="labels")

    if use_two_fc:
        fc1 = fc_layer(flattened, 28*28, 2520, "fc1")
        relu = tf.nn.relu(fc1)
        tf.summary.histogram("fc1/relu", relu)
        logits = fc_layer(fc1, 2520, 62, "fc2")
    else:
        logits = fc_layer(flattened, 28*28, 62, "fc")
    
    with tf.name_scope("xent"):
        xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=y
            ), name="xent"
        )
        tf.summary.scalar("xent", xent)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)
    
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
    summ = tf.summary.merge_all()

    session.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("logs/conv/"+hparam)
    writer.add_graph(session.graph)

    images, labels = load_data(train_data_directory)
    images28 = preprocessing.rgb2gray(preprocessing.resize(images))
    
    test_images, test_labels = load_data(test_data_directory)
    test_images28 = preprocessing.rgb2gray(preprocessing.resize(test_images))
    
    batches = batch_generator(images28, labels, batch_size)

    for i in range(FLAGS.max_steps):
        batch = next(batches)
        if i % 10 == 0:
            [train_accuracy, s] = session.run([accuracy, summ], feed_dict={x: batch[0], y:  batch[1]})
            writer.add_summary(s, i)
            print("Epoch: ", i, " Train acc: ", train_accuracy)
        session.run(train_step, feed_dict={x: batch[0], y: batch[1]})

def make_hparam_string(learning_rate, batch_size, cycle, use_two_fc):
    return "conv_s4-lr_%.0E__b_%s__c_%s__fc_%s" % (learning_rate, batch_size, cycle, 2 if use_two_fc else 1)

def main(_):
    for learning_rate in [1E-3]:#, 1E-4]:
        for batch_size in [100]:#, 250, 500]:
            for use_two_fc in [False, True]:
                for i in [0]: #range(3):
                    hparam = make_hparam_string(learning_rate, batch_size, i, use_two_fc)
                    print('Starting run for %s' % hparam)
                    model(learning_rate, batch_size, hparam, use_two_fc)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=2001,
                      help='Number of steps to run trainer.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]])
