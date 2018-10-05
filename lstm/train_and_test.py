import numpy as np
import tensorflow as tf
from random import randint
import datetime
import os

import implementation as imp

batch_size = imp.batch_size
iterations = 3000
seq_length = 40  # Maximum length of sentence
reverse_compound_class_mapping = imp.reverse_compound_class_mapping
num_classes = len(reverse_compound_class_mapping)

checkpoints_dir = "./checkpoints"

def validation_split(pct_validation):
    data_size = training_data.shape[0]
    num_validation = int(data_size*pct_validation)
    num_training = data_size - num_validation
    # Shuffle
    rng_state = np.random.get_state()
    np.random.shuffle(training_data)
    np.random.set_state(rng_state)
    np.random.shuffle(training_classes)
    np.random.set_state(rng_state)
    np.random.shuffle(training_fnames)
    # Split classes to training and validation
    training, validation = np.split(training_data, [num_training], axis=0)
    training_cls, validation_cls = np.split(training_classes, [num_training], axis=0)
    training_fs, validation_fs = np.split(training_fnames, [num_training], axis=0)
    print("Training: {}, Validation: {}".format(training.shape, validation.shape))
    return training, training_cls, training_fs, validation, validation_cls, validation_fs

def getTrainBatch():
    labels = []
    fnames = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        num = randint(0, training_data.shape[0]-1)
        label = [0] * num_classes
        label[training_classes[num]] = 1
        labels.append(label)
        arr[i] = training_data[num]
        fnames.append(training_fnames[num])
    return arr, labels, fnames

def getNextOriginalBatch(offset, batch_size):
    labels = []
    fnames = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        num = offset + i
        # print("index:", num)
        label = [0] * num_classes
        label[original_classes[num]] = 1
        labels.append(label)
        arr[i] = original_data[num]
        fnames.append(original_fnames[num])
    return arr, labels, fnames

def getNextTrainBatch(offset, batch_size):
    labels = []
    fnames = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        num = offset + i
        # print("index:", num)
        label = [0] * num_classes
        label[training_classes[num]] = 1
        labels.append(label)
        arr[i] = training_data[num]
        fnames.append(training_fnames[num])
    return arr, labels, fnames

def getTestBatch():
    labels = []
    fnames = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        num = randint(0, validation_data.shape[0]-1)
        label = [0] * num_classes
        label[validation_classes[num]] = 1
        labels.append(label)
        arr[i] = validation_data[num]
        fnames.append(training_fnames[num])
    return arr, labels, fnames

def getNextTestBatch(offset, batch_size):
    labels = []
    fnames = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        num = offset + i
        label = [0] * num_classes
        label[validation_classes[num]] = 1
        labels.append(label)
        arr[i] = validation_data[num]
        fnames.append(validation_fnames[num])
    return arr, labels, fnames

# Call implementation
word2vec_array, word2vec_dict = imp.load_word2vec_embeddings()
training_data, training_classes, training_fnames = imp.load_data(word2vec_dict)
original_data = training_data[:, :]
original_classes = training_classes[:]
original_fnames = training_fnames[:]
training_data, training_classes, training_fnames, \
validation_data, validation_classes, validation_fnames = \
    validation_split(0.2)
input_data, labels, dropout_keep_prob, optimizer, accuracy, loss, \
    prediction, correct_pred, pred_class, pred_prob = \
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
    batch_data, batch_labels, batch_fnames = getTrainBatch()
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
        batch_data, batch_labels, batch_fnames = getTestBatch()
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
# sess.close()

# Run on all test data
print("\n ============ Analyse all validation data")
batch_data, batch_labels, batch_fnames = getNextTestBatch(0, batch_size)
sess.run(optimizer, {input_data: batch_data, labels: batch_labels, dropout_keep_prob: 1.0})
loss_value, accuracy_value, summary, predictions, pred_classes = sess.run(
    [loss, accuracy, summary_op, prediction, pred_class],
    {input_data: batch_data,
     labels: batch_labels})
print('predictions', predictions)
print("pred_class", pred_classes)

results = []
for i in range(len(pred_classes)):
    batch_label = batch_labels[i]
    pred_class_num = pred_classes[i]
    fname = batch_fnames[i]
    batch_class, _ = max(enumerate(batch_label), key=lambda x: x[1])
    results.append((fname, reverse_compound_class_mapping[batch_class], reverse_compound_class_mapping[pred_class_num]))
print("Results", results)

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import glob

files = glob.glob('output/*')
for f in files:
    os.remove(f)

for result in results:
    f = os.path.join('data', '{}.png'.format(result[0]))
    of = os.path.join('output', '{}.png'.format(result[0]))
    img = Image.open(f)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 16)
    color = (255, 0, 0)
    if result[1] == result[2]:
        color = (0, 255, 0)
    draw.text((0, 0), result[2], color, font=font)
    img.save(of)


print("\n ============ Analyse all training data")
num_batches = len(training_data) // batch_size
output_lines = []
for batch_num in range(num_batches):
    offset = batch_num * batch_size
    batch_data, batch_labels, batch_fnames = getNextTrainBatch(offset, batch_size)
    sess.run(optimizer, {input_data: batch_data, labels: batch_labels, dropout_keep_prob: 1.0})
    loss_value, accuracy_value, summary, predictions, pred_classes, pred_probs = sess.run(
        [loss, accuracy, summary_op, prediction, pred_class, pred_prob],
        {input_data: batch_data,
         labels: batch_labels})
    output_line_batch = batch_fnames[:]
    batch_classes = []
    for batch_label in batch_labels:
        batch_class, _ = max(enumerate(batch_label), key=lambda x: x[1])
        batch_classes.append(reverse_compound_class_mapping[batch_class])
    for i in range(batch_size):
        pred_strs = list(map(lambda n: str(n), pred_probs[i].tolist()))
        output_line_batch[i] = [output_line_batch[i]] + [batch_classes[i]] + pred_strs
    output_lines += output_line_batch

with open("pred_vecs.csv", "w") as pred_vecs:
    for line in output_lines:
        line_str = ",".join(line)
        pred_vecs.write(line_str + "\n")

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
