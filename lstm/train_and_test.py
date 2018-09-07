import numpy as np
import tensorflow as tf
from random import randint
import datetime
import os

import implementation as imp

batch_size = imp.batch_size
iterations = 1000
seq_length = 40  # Maximum length of sentence

checkpoints_dir = "./checkpoints"

def validation_split(pct_validation):
    data_size = training_data.shape[0]
    num_validation = int(data_size*pct_validation)
    num_training = data_size - num_validation
    # np.random.shuffle(training_data)
    # Split classes to training and validation
    training, validation = np.split(training_data, [num_training], axis=0)
    training_cls, validation_cls = np.split(training_classes, [num_training], axis=0)
    print("Training: {}, Validation: {}".format(training.shape, validation.shape))
    return training, training_cls, validation, validation_cls

def getTrainBatch():
    labels = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        num = randint(0, training_data.shape[0]-1)
        label = [0, 0, 0, 0]
        label[training_classes[num]] = 1
        labels.append(label)
        arr[i] = training_data[num]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        num = randint(0, validation_data.shape[0]-1)
        label = [0, 0, 0, 0]
        label[validation_classes[num]] = 1
        labels.append(label)
        arr[i] = validation_data[num]
    return arr, labels

# Call implementation
word2vec_array, word2vec_dict = imp.load_word2vec_embeddings()
training_data, training_classes = imp.load_data(word2vec_dict)
training_data, training_classes, validation_data, validation_classes = \
    validation_split(0.2)
input_data, labels, dropout_keep_prob, optimizer, accuracy, loss = \
    imp.define_graph(word2vec_array)

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
        loss_value, accuracy_value, summary = sess.run(
            [loss, accuracy, summary_op],
            {input_data: batch_data,
             labels: batch_labels})
        writer.add_summary(summary, i)
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
sess.close()

# Run validation
# print("\nRunning validation ==============")
# viterations = iterations // 10 # (num_validation // imp.batch_size) * 2 # No particular logic here, just a small number
# #for i in range(iterations, iterations+viterations):
#     #batch_data, batch_labels = getTestBatch()
# batch_data, batch_labels = getNextTestBatch() # Test everything once
# while batch_data is not None:
#     i += 1
#     sess.run(optimizer, {input_data: batch_data, labels: batch_labels, dropout_keep_prob: 1.0})
#     if (i % 50 == 0):
#         loss_value, accuracy_value, summary = sess.run(
#             [loss, accuracy, summary_op],
#             {input_data: batch_data,
#              labels: batch_labels})
#         writer.add_summary(summary, i)
#         print("Iteration: ", i)
#         print("loss", loss_value)
#         print("acc", accuracy_value)
#     batch_data, batch_labels = getNextTestBatch()
#     if (i % 10000 == 0 and i != 0):
#         if not os.path.exists(checkpoints_dir):
#             os.makedirs(checkpoints_dir)
#         save_path = all_saver.save(sess, checkpoints_dir +
#                                    "/trained_model.ckpt",
#                                    global_step=i)
#         print("Saved model to %s" % save_path)
# sess.close()
