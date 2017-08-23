from __future__ import print_function
import os
# import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import urllib.request as ur
import requests
import datetime
import types
import json

# Parameters
learning_rate = 0.001
training_iters = 3000

# Network Parameters
n_classes = 2 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, 128, 96, 3])
y = tf.placeholder(tf.float32, [None, n_classes])
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
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 24])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 24, 96])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([32*24*96, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([24])),
    'bc2': tf.Variable(tf.random_normal([96])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)
pred_result=tf.argmax(pred, 1)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
saver=tf.train.Saver()

dir = "./roads_test/";

# # Launch the graph
# with tf.Session() as sess:
#     saver.restore(sess, "./model/model.ckpt")
#     step = 1
#     # Keep training until reach max iterations
#     list = os.listdir(dir)
#     print(list)
#     print(len(list))
#     # list = ['road_4210-0.jpg', 'road_4220-0.jpg', 'road_4230-0.jpg', 'road_4240-0.jpg', 'road_4250-0.jpg', 'road_4260-0.jpg', 'road_4270-0.jpg', 'road_4285-1.jpg', 'road_4290-1.jpg', 'road_4390-0.jpg'];
#     for batch_id in range(0, 30):
#         batch = list[batch_id * 10:batch_id * 10 + 10]
#         batch_xs = []
#         batch_ys = []
#         for image in batch:
#             id_tag = image.find("-")
#             score = image[0:id_tag]
#             # print(score)
#             img = Image.open(dir + image)
#             img_ndarray = np.asarray(img, dtype='float32')
#             img_ndarray = np.reshape(img_ndarray, [128, 96, 3])
#             # print(img_ndarray.shape)
#             batch_x = img_ndarray
#             batch_xs.append(batch_x)

#         # print(batch_ys)
#         batch_xs = np.asarray(batch_xs)
#         print(batch_xs.shape)

#         # Run optimization op (backprop)
#         pred_result_test=sess.run(pred_result, feed_dict={x: batch_xs,keep_prob: 1.})
#         print(batch);
#         print(pred_result_test)
#         if(0 in pred_result_test):
#             print('检测到特征图片！');
#     print("Test Finished!")
#     saver.save(sess,"./model/model.ckpt")

#上班打卡
isPunch = False;
def punchIn(imgPath):
    url = 'http://xxx.xxx.xxx/xxx/punchin';
    files = {'location_image': ('location_image.jpg', open(imgPath, 'rb'))}
    h = datetime.datetime.now().hour
    punch_type = 'on';
    if h > 15:
        punch_type = 'off';
    data = {
        'company_id': '',
        'user_id': '',
        'token': '124a2ae',
        'img_type': 'pic',
        'width': 640,
        'height': 480,
        'size': os.path.getsize(imgPath),
        'type': punch_type,
        'device_name': '机器学习-图像识别',
        'device_no': '001',
        'longitude': '104.xxxxx',
        'latitude': '30.xxxxx',
        'location': '成都',
        'remark': '成都（机器学习-图像识别-自动检测）'
    }
    r = requests.post(url, files=files, data=data)
    print(r.text)
    j = json.loads(r.text)
    if j['data'] == 'TRUE':
        isPunch = True;


def run_model(img, img_backup):
    # Launch the graph
    with tf.Session() as sess:
        saver.restore(sess, "./model/model.ckpt")
        # Keep training until reach max iterations
        # list = os.listdir(dir)
        # print(list)
        # print(len(list))
        # list = ['road_4210-0.jpg', 'road_4220-0.jpg', 'road_4230-0.jpg', 'road_4240-0.jpg', 'road_4250-0.jpg', 'road_4260-0.jpg', 'road_4270-0.jpg', 'road_4285-1.jpg', 'road_4290-1.jpg', 'road_4390-0.jpg'];
        batch_xs = []
        batch_ys = []
        if type(img) == type('a'):
            img = Image.open(dir + img)
        img_ndarray = np.asarray(img, dtype='float32')
        img_ndarray = np.reshape(img_ndarray, [128, 96, 3])
        # print(img_ndarray.shape)
        batch_x = img_ndarray
        batch_xs.append(batch_x)

        # print(batch_ys)
        batch_xs = np.asarray(batch_xs)
        print(batch_xs.shape)

        # Run optimization op (backprop)
        pred_result_test = sess.run(pred_result, feed_dict={x: batch_xs, keep_prob: 1.})
        print(pred_result_test)
        ctime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S");
        if(0 in pred_result_test):
            print(ctime + ' 检测到特征图片：');
            # img_backup.show();
            imgPath = './backup/' + ctime + '.jpg';
            # img_backup.save(imgPath);
            #打卡
            # punchIn(imgPath);
        else:
            print(ctime + ' .........');
            # img.show();


def resize_img(img):
    # print(img.size)
    basewidth = 128;
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img_128 = img.resize((basewidth, hsize));
    # print(img_128.size)
    return img_128;

def recognition(url):
    i = 0;
    while True:
        stream = ur.urlopen(url, None, 5);
        imgNp = np.array(bytearray(stream.read()), dtype=np.uint8);
        img = cv2.imdecode(imgNp, -1);
        cv2.namedWindow("origin", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("origin", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('origin', img)

        if isPunch == True:
            print(isPunch);
        else:
            if(i % 2 == 0):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
                img_origin = Image.fromarray(img);
                img_backup = img_origin;
                img_128 = resize_img(img_origin);
                # img_128.show();
                run_model(img_128, img_backup);
        
        i += 1;
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# recognition('http://192.168.43.1:8080/shot.jpg');
# recognition('http://172.20.10.2:8080/shot.jpg');
run_model('2017-08-14 14:52:53.jpg');
# punchIn('./backup/2017-08-14 14:52:56.jpg');