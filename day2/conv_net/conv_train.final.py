import os
import sys
import argparse
import tensorflow as tf
import numpy as np
from load_data import load_data
import preprocessing
import random
from skimage import io 
from math import sqrt

FLAGS = None
ROOT_PATH = "../../data/belgium_ts"
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")

TEST_BATCH_SIZE = 2500
SPRITE_COLUMN = int(sqrt(TEST_BATCH_SIZE))

def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="w")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

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


def model(learning_rate, batch_size, hparam, use_two_conv, use_two_fc):
    tf.reset_default_graph()
    session = tf.Session()

    # Setup placeholders, and reshape the data
    x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28], name="images")
    images_flat = tf.contrib.layers.flatten(x)
    x_image = tf.reshape(images_flat, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 3)
    y = tf.placeholder(dtype = tf.int32, shape = [None, 62], name="labels")

    if use_two_conv:
        conv1 = conv_layer(x_image, 1, 32, "conv1")
        conv_out = conv_layer(conv1, 32, 64, "conv2")
    else:
        conv1 = conv_layer(x_image, 1, 64, "conv")
        conv_out = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
        
    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])
    if use_two_fc:
        fc1 = fc_layer(flattened, 7*7*64, TEST_BATCH_SIZE, "fc1")
        relu = tf.nn.relu(fc1)
        embedding_input = relu
        tf.summary.histogram("fc1/relu", relu)
        embedding_size = TEST_BATCH_SIZE
        logits = fc_layer(fc1, TEST_BATCH_SIZE, 62, "fc2")
    else:
        embedding_input = flattened
        embedding_size = 7*7*64
        logits = fc_layer(flattened, 7*7*64, 62, "fc")
    
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

    embedding = tf.Variable(tf.zeros([TEST_BATCH_SIZE, embedding_size]), name="test_embedding")
    assignment = embedding.assign(embedding_input)
    saver = tf.train.Saver()

    session.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("logs/conv/"+hparam)
    writer.add_graph(session.graph)
    test_writer = tf.summary.FileWriter("logs/conv/test_"+hparam)
    test_writer.add_graph(session.graph)

    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.image_path = os.path.join(os.getcwd(), "logs/conv/%s/sprites.png" % hparam)
    embedding_config.metadata_path = os.path.join(os.getcwd(), "logs/conv/%s/labels.tsv" % hparam)
    embedding_config.sprite.single_image_dim.extend([28, 28])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    images, labels = load_data(train_data_directory)
    images28 = preprocessing.rgb2gray(preprocessing.resize(images))
    
    test_images, test_labels = load_data(test_data_directory)
    test_images28 = preprocessing.rgb2gray(preprocessing.resize(test_images))

    test_batches = batch_generator(test_images28, test_labels, TEST_BATCH_SIZE)

    #Black magic for embedding
    test_batch = next(test_batches)
    rotated = [np.transpose(i) for i in test_batch[0]]

    chunks = np.array([np.array(rotated[x:x+SPRITE_COLUMN]) for x in range(0, len(rotated), SPRITE_COLUMN)])
    print("Chunks: ",chunks.shape)
    with open("logs/conv/%s/labels.tsv" % hparam, 'w') as f:
        f.write("".join(["%s \n" % l.index(1) for l in test_batch[1]]))
    master_height = 28 * SPRITE_COLUMN

    h_flattened = np.reshape(chunks, [-1, master_height, 28])
    h_flattened = np.array([np.transpose(x) for x in h_flattened])
    print("H flatten", h_flattened.shape)
    w_flattened = np.reshape(np.array(h_flattened), [-1, 28 * SPRITE_COLUMN, 28 * SPRITE_COLUMN])
    print("W flatten", w_flattened.shape)
    io.imsave('logs/conv/%s/sprites.png' % hparam, w_flattened[0])

    batches = batch_generator(images28, labels, batch_size)

    for i in range(FLAGS.max_steps):
        batch = next(batches)
        if i % 10 == 0:
            [train_accuracy, s] = session.run([accuracy, summ], feed_dict={x: batch[0], y:  batch[1]})
            writer.add_summary(s, i)
            print("Epoch: ", i, " Train acc: ", train_accuracy)
        if i % 500 == 0:
            [test_accuracy, _s, s] = session.run([accuracy, assignment, summ], feed_dict={x: test_batch[0], y: test_batch[1]})
            test_writer.add_summary(s, i)
            saver.save(session, os.path.join("logs/conv/%s/" % hparam, "checkpoint.ckpt"), global_step=i)
            print("Epoch: ", i, " Test acc: ", test_accuracy)
        session.run(train_step, feed_dict={x: batch[0], y: batch[1]})

def make_hparam_string(learning_rate, batch_size, cycle, use_two_conv, use_two_fc):
    return "conv_s5-lr_%.0E__b_%s__c_%s__conv_%s__fc_%s" % (learning_rate, batch_size, cycle, 2 if use_two_conv else 1, 2 if use_two_fc else 1)

def main(_):
    for learning_rate in [1E-3, 1E-4]:
        for batch_size in [100, 250]:
            for use_two_fc in [False, True]:
                for use_two_conv in [False, True]:
                    for i in [0]:
                        hparam = make_hparam_string(learning_rate, batch_size, i, use_two_conv, use_two_fc)
                        print('Starting run for %s' % hparam)
                        model(learning_rate, batch_size, hparam, use_two_conv, use_two_fc)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=2001,
                      help='Number of steps to run trainer.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]])
