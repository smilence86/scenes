from __future__ import division
import os
import numpy as np
from PIL import Image
import datetime
import tensorflow as tf

dir = "./roads_128/";


# # Parameters
batch_size = 10

# # Network Parameters
n_classes = 5 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# # tf Graph input
x = tf.placeholder(tf.float32, [None, 128, 96, 3])
# x = tf.placeholder(tf.float32, [None, 256, 192, 3])
y_ = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, s=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 128, 96, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    print(conv1.shape)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    print(conv1.shape)
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    print(conv2.shape)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    print(conv2.shape)
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 3 input, 24 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 16])),
    # 5x5 conv, 24 inputs, 96 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 16, 48])),
    # fully connected, 32*32*96 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([32*24*48, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([48])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Keep training until reach max iterations
    list = os.listdir(dir)
    # print(list[1190:1120])
    print(len(list))
    total_imgs = len(list);
    print('总图片数：', total_imgs);
    total_page = 1;
    if total_imgs % batch_size == 0:
    	total_page = int(total_imgs / batch_size);
    else:
    	total_page = int(total_imgs / batch_size) + 1;
    print('总页数：', total_page);
    #训练批次
    batch = 0
    total_batch = 40
    while batch < total_batch:
        batch += 1;
        for index in range(0, total_page):
            images = list[index * batch_size:index * batch_size + batch_size]
            batch_xs = []
            batch_ys = []
            for image in images:
                id_tag = image.find("-")
                ext = image.find(".")
                score = image[id_tag+1:ext]
                print(image + '\tscore:' + score)
                # print(type(score))
                img = Image.open(dir + image)
                # row,col =  img.size;
                # print(row,col)
                img_ndarray = np.asarray(img, dtype='float32')
                img_ndarray = np.reshape(img_ndarray, [128, 96, 3])
                # print(img_ndarray.shape)
                batch_x = img_ndarray
                batch_xs.append(batch_x)
                batch_y = np.asarray([0, 0, 0, 0, 0])
                batch_y[int(score)] = 1
                # print(batch_y)
                batch_y = np.reshape(batch_y, [5, ])
                batch_ys.append(batch_y)
            # print(batch_xs)
            # print(batch_ys)
            batch_xs = np.asarray(batch_xs)
            print(batch_xs.shape)
            batch_ys = np.asarray(batch_ys)
            print(batch_ys.shape)

            sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: dropout})
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.});
            ctime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S");
            print(ctime + "\tbatch:" + str(batch) + "/" + str(total_batch) + ", page:" + str(index + 1) + "/" + str(total_page) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
    print("Optimization Finished!")
    saver.save(sess,"./model/model.ckpt")


