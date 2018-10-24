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
    np.random.set_state(rng_state)
    np.random.shuffle(training_o_comps)
    # Split classes to training and validation
    training, validation = np.split(training_data, [num_training], axis=0)
    training_cls, validation_cls = np.split(training_classes, [num_training], axis=0)
    training_fs, validation_fs = np.split(training_fnames, [num_training], axis=0)
    training_o_cs, validation_o_cs = np.split(training_o_comps, [num_training], axis=0)
    print("Training: {}, Validation: {}".format(training.shape, validation.shape))
    return training, training_cls, training_fs, training_o_cs, \
           validation, validation_cls, validation_fs, validation_o_cs

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
    orig_comps = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        num = offset + i
        # print("index:", num)
        label = [0] * num_classes
        label[training_classes[num]] = 1
        labels.append(label)
        arr[i] = training_data[num]
        fnames.append(training_fnames[num])
        orig_comps.append(training_o_comps[num])
    return arr, labels, fnames, orig_comps

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
training_data, training_classes, training_fnames, training_o_comps = imp.load_data(word2vec_dict)
original_data = training_data[:, :]
original_classes = training_classes[:]
original_fnames = training_fnames[:]
original_o_comps = training_o_comps[:]
training_data, training_classes, training_fnames, training_o_comps, \
validation_data, validation_classes, validation_fnames, validation_o_comps = \
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
all_pred_classes = []
all_batch_labels = []
all_pred_labels = []
all_batch_fnames = []
overall_accuracy = 0.0
num_batches = len(validation_data) // batch_size
for batch_num in range(num_batches):
    offset = batch_num * batch_size
    batch_data, batch_labels, batch_fnames = getNextTestBatch(offset, batch_size)
    sess.run(optimizer, {input_data: batch_data, labels: batch_labels, dropout_keep_prob: 1.0})
    loss_value, accuracy_value, summary, predictions, pred_classes = sess.run(
        [loss, accuracy, summary_op, prediction, pred_class],
        {input_data: batch_data,
        labels: batch_labels})
    overall_accuracy += accuracy_value
    #print('predictions', predictions, type(predictions))
    #print("pred_class", pred_classes)
    all_pred_classes += pred_classes.tolist()
    all_batch_labels += batch_labels
    pred_labels = (predictions == predictions.max(axis=1, keepdims=True)).astype(int)
    all_pred_labels += pred_labels.tolist()
    all_batch_fnames += batch_fnames

print('Overall Accuracy: ', overall_accuracy / (num_batches))
results = []
all_batch_classes = []
for i in range(len(all_pred_classes)):
    batch_label = all_batch_labels[i]
    pred_class_num = all_pred_classes[i]
    fname = all_batch_fnames[i]
    batch_class, _ = max(enumerate(batch_label), key=lambda x: x[1])
    all_batch_classes.append(batch_class)
    results.append((fname, reverse_compound_class_mapping[batch_class], reverse_compound_class_mapping[pred_class_num]))
#print("Results", results)



# # Draw on images
# from PIL import Image
# from PIL import ImageFont
# from PIL import ImageDraw 
# import glob

# files = glob.glob('output/*')
# for f in files:
#     os.remove(f)

# for result in results:
#     f = os.path.join('data', '{}.png'.format(result[0]))
#     of = os.path.join('output', '{}.png'.format(result[0]))
#     img = Image.open(f)
#     draw = ImageDraw.Draw(img)
#     font = ImageFont.truetype("arial.ttf", 16)
#     color = (255, 0, 0)
#     if result[1] == result[2]:
#         color = (0, 255, 0)
#     draw.text((0, 0), result[2], color, font=font)
#     img.save(of)


print("\n ============ Analyse all training data")
num_batches = len(training_data) // batch_size
output_lines = []
for batch_num in range(num_batches):
    offset = batch_num * batch_size
    batch_data, batch_labels, batch_fnames, batch_o_cs = getNextTrainBatch(offset, batch_size)
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
        output_line_batch[i] = [output_line_batch[i]] + [batch_classes[i]] + [batch_o_cs[i]] + pred_strs
    output_lines += output_line_batch

with open("pred_vecs.csv", "w") as pred_vecs:
    for line in output_lines:
        line_str = ",".join(line)
        pred_vecs.write(line_str + "\n")

sess.close()


def tf_f1_score(y_true, y_pred):
    """Computes 3 different f1 scores, micro macro
    weighted.
    micro: f1 score accross the classes, as 1
    macro: mean of f1 scores per class
    weighted: weighted average of f1 scores per class,
            weighted from the support of each class


    Args:
        y_true (Tensor): labels, with shape (batch, num_classes)
        y_pred (Tensor): model's predictions, same shape as y_true

    Returns:
        tuple(Tensor): (micro, macro, weighted)
                    tuple of the computed f1 scores
    """

    f1s = [0, 0, 0]

    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    for i, axis in enumerate([None, 0]):
        TP = tf.count_nonzero(y_pred * y_true, axis=axis)
        FP = tf.count_nonzero(y_pred * (y_true - 1), axis=axis)
        FN = tf.count_nonzero((y_pred - 1) * y_true, axis=axis)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        f1s[i] = tf.reduce_mean(f1)

    weights = tf.reduce_sum(y_true, axis=0)
    weights /= tf.reduce_sum(weights)

    f1s[2] = tf.reduce_sum(f1 * weights)

    micro, macro, weighted = f1s
    return micro, macro, weighted

tf.reset_default_graph()
y_true = tf.Variable(all_batch_labels)
y_pred = tf.Variable(all_pred_labels)
micro, macro, weighted = tf_f1_score(y_true, y_pred)
with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)
    mic, mac, wei = sess.run([micro, macro, weighted])
    print('micro: {:.8f}\nmacro: {:.8f}\nweighted: {:.8f}'.format(
        mic, mac, wei
    ))


from sklearn.metrics import f1_score, precision_score, recall_score

labels = np.array(all_batch_labels, int)
predictions = np.array(all_pred_labels, int)

print("F1 score")
mic = f1_score(labels, predictions, average='micro')
mac = f1_score(labels, predictions, average='macro')
wei = f1_score(labels, predictions, average='weighted')
print('micro: {:.8f}\nmacro: {:.8f}\nweighted: {:.8f}'.format(
    mic, mac, wei
))

print("Precision")
mic = precision_score(labels, predictions, average='micro')
mac = precision_score(labels, predictions, average='macro')
wei = precision_score(labels, predictions, average='weighted')
print('micro: {:.8f}\nmacro: {:.8f}\nweighted: {:.8f}'.format(
    mic, mac, wei
))

print("Recall")
mic = recall_score(labels, predictions, average='micro')
mac = recall_score(labels, predictions, average='macro')
wei = recall_score(labels, predictions, average='weighted')
print('micro: {:.8f}\nmacro: {:.8f}\nweighted: {:.8f}'.format(
    mic, mac, wei
))

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
