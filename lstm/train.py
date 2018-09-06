"""
You are encouraged to edit this file during development, however your final
model must be trained using the original version of this file. This file
trains the model defined in implementation.py, performs tensorboard logging,
and saves the model to disk every 10000 iterations. It also prints loss
values to stdout every 50 iterations.
"""


import numpy as np
import tensorflow as tf
from random import randint
import datetime
import os

import implementation as imp

batch_size = imp.batch_size
iterations = 100000
seq_length = 40  # Maximum length of sentence

checkpoints_dir = "./checkpoints"

'''def getTrainBatch():
    labels = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(0, 12499)
            labels.append([1, 0])
        else:
            num = randint(12500, 24999)
            labels.append([0, 1])
        arr[i] = training_data[num]
    return arr, labels'''

num_training = None
num_validation = None
num_tests = 0

def validationSplit(pct_validation):
    global num_training
    global num_validation
    data_size = training_data.shape[0] // 2
    num_validation = int(data_size*pct_validation)
    num_training = data_size - num_validation
    # Split into classes
    positive = training_data[:data_size,]
    negative = training_data[data_size:,]
    print ('positive:', len(positive), 'negative:', len(negative))
    # Shuffle each class
    np.random.shuffle(positive)
    np.random.shuffle(negative)
    # Split classes to training and validation
    pos_training, pos_validation = np.split(positive, [num_training], axis=0)
    neg_training, neg_validation = np.split(negative, [num_training], axis=0)
    # Join the classes
    num_training *= 2
    num_validation *= 2
    training = np.concatenate((pos_training, neg_training), axis=0)
    validation = np.concatenate((pos_validation, neg_validation), axis=0)
    print("Training: {}, Validation: {}".format(training.shape, validation.shape))
    return training, validation

def getTrainBatch():
    num_items = num_training // 2
    labels = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(0, num_items-1)
            labels.append([1, 0])
        else:
            num = randint(num_items, 2*num_items-1)
            labels.append([0, 1])
        arr[i] = training_data[num]
    return arr, labels

def getTestBatch():
    num_items = num_validation // 2
    labels = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(0, num_items-1)
            labels.append([1, 0])
        else:
            num = randint(num_items, 2*num_items-1)
            labels.append([0, 1])
        arr[i] = validation_data[num]
    return arr, labels

# Call implementation
glove_array, glove_dict = imp.load_glove_embeddings()
print ('array length:', len(glove_array), len(glove_dict))
training_data = imp.load_data(glove_dict)
training_data, validation_data = validationSplit(0.15)
input_data, labels, dropout_keep_prob, optimizer, accuracy, loss = \
    imp.define_graph(glove_array)

# tensorboard
train_accuracy_op = tf.summary.scalar("training_accuracy", accuracy)
tf.summary.scalar("loss", loss)
summary_op = tf.summary.merge_all()

# saver
all_saver = tf.train.Saver()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

logdir = "tensorboard/" + datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)
train_writer = tf.summary.FileWriter(logdir + '/train')
test_writer = tf.summary.FileWriter(logdir + '/test')


for i in range(iterations):
    batch_data, batch_labels = getTrainBatch()
    sess.run(optimizer, {input_data: batch_data, labels: batch_labels, dropout_keep_prob: 0.5})
    if (i % 50 == 0):
        loss_value, accuracy_value, summary1 = sess.run(
            [loss, accuracy, summary_op],
            {input_data: batch_data,
             labels: batch_labels})
        train_writer.add_summary(summary1, i)
        print("Iteration: ", i)
        print("loss", loss_value)
        print("acc", accuracy_value)

        # add validation test, doesn't have to be this frequent
        batch_data, batch_labels = getTestBatch()
        loss_value, accuracy_value, summary2 = sess.run(
            [loss, accuracy, summary_op],
            {input_data: batch_data,
             labels: batch_labels})
        test_writer.add_summary(summary2, i)
        print("test loss", loss_value)
        print("test acc", accuracy_value)
        
    if (i % 10000 == 0 and i != 0):
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        save_path = all_saver.save(sess, checkpoints_dir +
                                   "/trained_model.ckpt",
                                   global_step=i)
        print("Saved model to %s" % save_path)
#sess.close()

''' 
print(len(training_data))
epoch = len(training_data) // batch_size
for i in range(iterations // epoch):
    # run training
    labels = [[1,0] for x in range(num_training // 2)]
    labels.extend([[0,1] for x in range(num_training //2)])
    test_set = list(zip(training_data, labels))
    print(training_data[0:2], labels[0:2], test_set[0:2], training_data.shape)
    np.random.shuffle(test_set)
    for i2 in range(num_training // batch_size):
        batch = test_set[i2 * batch_size : (i2+1) * batch_size]
        batch_arr, labels_arr = zip(*(batch))
        arr = np.zeros([batch_size, seq_length])
        print(arr[0])
        labels_arr = []
        for x in range(batch_size):
            arr[x] = batch_arr[x]
            labels_arr.append(batch[x][1])
        print(arr, labels[0:10])
        *batch_arr, b = batch_arr
        batch_arr.append(b)
        print(type(batch_arr))
        *labels_arr, l = labels_arr
        labels_arr.append(l)
        labels_arr = np.array(labels_arr)
        batch_arr = np.array(batch_arr)
        print(type(batch_arr))
        print (labels_arr[0:2])
        print (batch_arr[0:2], batch_arr.shape)
        sess.run(optimizer, {input_data: arr, labels: labels_arr, dropout_keep_prob: 0.75})
        if (i2 * i % 50 == 0):
            loss_value, accuracy_value, summary = sess.run(
            [loss, accuracy, summary_op],
            {input_data: batch_data,
             labels: batch_labels})
            writer.add_summary(summary, i * i2)
            print("Iteration: ", i * i2)
            print("loss", loss_value)
            print("acc", accuracy_value)

            # add validation test
            batch_data, batch_labels = getTestBatch()
            loss_value, accuracy_value, summary = sess.run(
            [loss, accuracy, summary_op],
            {input_data: batch_data,
             labels: batch_labels})
            test_writer.add_summary(summary, i * i2)
            print("test loss", loss_value)
            print("test acc", accuracy_value)
sess.close()
'''
    

# Run validation
print("\nRunning validation ==============")
viterations = iterations // 10 # (num_validation // imp.batch_size) * 2 # No particular logic here, just a small number
#for i in range(iterations, iterations+viterations):
    #batch_data, batch_labels = getTestBatch()
batch_data, batch_labels = getNextTestBatch() # Test everything once
while batch_data is not None:
    i += 1
    sess.run(optimizer, {input_data: batch_data, labels: batch_labels, dropout_keep_prob: 1.0})
    if (i % 50 == 0):
        loss_value, accuracy_value, summary = sess.run(
            [loss, accuracy, summary_op],
            {input_data: batch_data,
             labels: batch_labels})
        writer.add_summary(summary, i)
        print("Iteration: ", i)
        print("loss", loss_value)
        print("acc", accuracy_value)
    batch_data, batch_labels = getNextTestBatch()
    if (i % 10000 == 0 and i != 0):
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        save_path = all_saver.save(sess, checkpoints_dir +
                                   "/trained_model.ckpt",
                                   global_step=i)
        print("Saved model to %s" % save_path)
sess.close()


