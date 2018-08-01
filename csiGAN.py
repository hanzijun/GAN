# --coding: utf-8 --
import tensorflow as tf
import numpy as np
import pickle
import normalization
import matplotlib.pyplot as plt
import display

is_training=True
subcarrier = 56
matrix = display.date_wrapper()
matrix = matrix[:,0:120]

# matrix = normalization.scalerNormalization(matrix)
matrix = normalization.MINMAXNormalization(matrix)
plt.imshow(matrix.reshape((subcarrier, 120)))
plt.show()
# pylab.figure()
# pylab.plot(matrix, 'k+', label='noisy measurements')  # 测量值
# pylab.show()

def get_generator(noise_img, out_dim, is_train=True, alpha=0.01):
    """
    :param noise_img: 噪声信号，tensor类型
    :param is_train: 是否为训练状态，该参数主要用于作为batch_normalization方法中的参数使用
    :param alpha: Leaky ReLU系数
    """
    with tf.variable_scope("generator") as scope0:
        if not is_train:
            scope0.reuse_variables()
        # none*100  to none*4 x 4 x 512
        # 全连接层
        layer1 = tf.layers.dense(noise_img, 7 * 15 * 512)
        layer1 = tf.reshape(layer1, [-1, 7, 15, 512])
        # batch normalization
        layer1 = tf.layers.batch_normalization(layer1, training=is_train)
        # Leaky ReLU
        layer1 = tf.maximum(alpha * layer1, layer1)
        layer1 = tf.nn.dropout(layer1, keep_prob=0.8)
        # print layer1.shape
        # 4 x 4 x 512 to 7 x 7 x 256
        # 卷积的一个逆向过程
        layer2 = tf.layers.conv2d_transpose(layer1, 256, 3, strides=2, padding='same')
        layer2 = tf.layers.batch_normalization(layer2, training=is_train)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob=0.8)
        # print layer2.shape
        # 7 x 7 256 to 14 x 14 x 128
        layer3 = tf.layers.conv2d_transpose(layer2, 128, 3, strides=2, padding='same')
        layer3 = tf.layers.batch_normalization(layer3, training=is_train)
        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3 = tf.nn.dropout(layer3, keep_prob=0.8)
        # print layer3.shape
        # 14 x 14 x 128 to 28 x 28 x 1
        logits = tf.layers.conv2d_transpose(layer3, out_dim, 3, strides=2, padding='same')
        # MNIST原始数据集的像素范围在0-1，这里的生成图片范围为(-1,1)
        # 因此在训练时，记住要把MNIST像素范围进行resize
        outputs = tf.tanh(logits)
        return outputs

def get_discriminator(inputs_img, reuse=False, alpha=0.01):
    """
    n_units: 隐层结点数量
    alpha: Leaky ReLU系数
    """
    with tf.variable_scope("discriminator") as scope1:
        # 28 x 28 x 1 to 14 x 14 x 128
        # 第一层不加入BN
        if reuse:
            scope1.reuse_variables()
        layer1 = tf.layers.conv2d(inputs_img, 128, 3, strides=2, padding='same')
        layer1 = tf.maximum(alpha * layer1, layer1)
        layer1 = tf.nn.dropout(layer1, keep_prob=0.8)
        # 14 x 14 x 128 to 7 x 7 x 256
        layer2 = tf.layers.conv2d(layer1, 256, 3, strides=2, padding='same')
        layer2 = tf.layers.batch_normalization(layer2, training=True)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob=0.8)
        # 7 x 7 x 256 to 4 x 4 x 512
        layer3 = tf.layers.conv2d(layer2, 512, 3, strides=2, padding='same')
        layer3 = tf.layers.batch_normalization(layer3, training=True)
        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3 = tf.nn.dropout(layer3, keep_prob=0.8)
        # 4 x 4 x 512 to 4*4*512 x 1
        flatten = tf.reshape(layer3, (-1, 7 * 15 * 512))  # (?,8192)
        logits = tf.layers.dense(flatten, 1)  # (?, 1)
        outputs = tf.sigmoid(logits)  # (?, 1)
        return logits, outputs


def get_loss(inputs_real, inputs_noise, image_depth, smooth=0.1):
                """
                @param inputs_real: 输入图片，tensor类型
                @param inputs_noise: 噪声图片，tensor类型
                @param image_depth: 图片的depth（或者叫channel）
                @param smooth: label smoothing的参数
                """
                g_outputs = get_generator(inputs_noise, image_depth, is_train=True)
                d_logits_real, d_outputs_real = get_discriminator(inputs_real)
                d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, reuse=True)
                # 计算Loss
                g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                                labels=tf.ones_like(d_outputs_fake)*(1-smooth)))
                d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                                     labels=tf.ones_like(d_outputs_real)*(1-smooth)))
                d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                                     labels=tf.zeros_like(d_outputs_fake)))
                d_loss = tf.add(d_loss_real, d_loss_fake)

                return g_loss, d_loss, d_loss_real, d_loss_fake

def get_optimizer(g_loss, d_loss, beta1=0.4, learning_rate=0.001):
                """
                @param g_loss: Generator的Loss
                @param d_loss: Discriminator的Loss
                @learning_rate: 学习率
                """
                train_vars = tf.trainable_variables()  #抓取所有的变量
                g_vars = [var for var in train_vars if var.name.startswith("generator")]
                d_vars = [var for var in train_vars if var.name.startswith("discriminator")]
                # Optimizer  就会返回所有定义的summary op
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
                    d_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)

                return g_vars, g_opt, d_opt

# 真实图像的size
img_height=56
img_width=120
img_depth = 1
beta = 0.4
noise_size = 200
alpha = 0.01
# learning_rate
learning_rate = 0.001
# label smoothing
smooth = 0.1
batch_size = 1
# 训练迭代轮数
epochs = 100
# 抽取样本数
n_sample = 1
# 存储测试样例
samples = []
# 存储loss
losses = []

tf.reset_default_graph()
real_img = tf.placeholder(tf.float32, [None, img_height, img_width, img_depth], name='inputs_real')
noise_img = tf.placeholder(tf.float32, [None, noise_size], name='inputs_noise')
g_loss, d_loss, d_loss_real, d_loss_fake = get_loss(real_img, noise_img, img_depth)
g_vars, g_train_opt, d_train_opt = get_optimizer(g_loss, d_loss, beta, learning_rate)

if is_training is True:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for e in range(epochs):
            for batch_i in range(0,100):
                batch_images = matrix.reshape((batch_size,img_height, img_width, img_depth))
                # 对图像像素进行scale，这是因为tanh输出的结果介于(-1,1),real和fake图片共享discriminator的参数
                # batch_images = batch_images * 2 - 1
                # generator的输入噪声
                batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
                # Run optimizers
                _ = sess.run(d_train_opt, feed_dict={real_img: batch_images, noise_img: batch_noise})
                _ = sess.run(g_train_opt, feed_dict={real_img: batch_images, noise_img: batch_noise})
            # 每一轮结束计算loss
            train_loss_d = sess.run(d_loss, feed_dict={real_img: batch_images, noise_img: batch_noise})
            # real img loss
            train_loss_d_real = sess.run(d_loss_real, feed_dict={real_img: batch_images, noise_img: batch_noise})
            # fake img loss
            train_loss_d_fake = sess.run(d_loss_fake, feed_dict={real_img: batch_images, noise_img: batch_noise})
            # generator loss
            train_loss_g = sess.run(g_loss, feed_dict={noise_img: batch_noise})
            print("Epoch {}/{}...".format(e + 1, epochs),
                "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(train_loss_d, train_loss_d_real, train_loss_d_fake),
                "Generator Loss: {:.4f}".format(train_loss_g))
            # 记录各类loss值
            losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))
            # 抽取样本后期进行观察
            sample_noise = np.random.uniform(-1, 1, size=(n_sample, noise_size))
            gen_samples = sess.run(get_generator(noise_img, img_depth, is_train=False),
                                   feed_dict={noise_img: sample_noise})
            samples.append(gen_samples)
            # 存储checkpoints
        saver.save(sess, './checkpoints/generator.ckpt')
else:
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        sample_noise = np.random.uniform(-1, 1, size=(1, noise_size))
        gen_samples = sess.run(get_generator(noise_img, 1, is_train=False), feed_dict={noise_img: sample_noise})
        m = plt.imshow(gen_samples.reshape((56, 120)),)
        plt.show()

# 将sample的生成数据记录下来
with open('train_samples.pkl', 'wb') as f:
    pickle.dump(samples, f)

def view_samples(epoch, samples):
    """
    epoch代表第几次迭代的图像
    samples为我们的采样结果
    """
    fig, axes = plt.subplots(figsize=(7, 7), nrows=5, ncols=5, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch][1]):  # 这里samples[epoch][1]代表生成的图像结果，而[0]代表对应的logits
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
    return fig, axes


