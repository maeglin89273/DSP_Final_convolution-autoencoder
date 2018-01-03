import scipy.io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def reshape_to_images(data, img_size):
    return data.reshape(-1, img_size, img_size, 1)

def normalize(data, mean=None):
    data = data.astype(np.float32) / 255
    if mean is None:
        mean = np.mean(data, axis=0)

    return data - mean, mean

def denormalize(norm_data, mean):
    return np.clip(255 * (norm_data + mean), 0, 255).astype(np.uint8)

def build_sparse_conv_autoencoder(kernel_size, kernel_num, input_name='input'):
    input_batch = tf.placeholder(tf.float32, [None, 28, 28, 1], name=input_name)
    encode_filters = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, 1, kernel_num], 0.0, 0.2))
    encode_biases = tf.Variable(tf.truncated_normal([kernel_num], 0, 0.2))
    encode_out = tf.nn.bias_add(tf.nn.conv2d(input_batch, encode_filters, [1, 1, 1, 1], 'SAME'), encode_biases)
    encode_out = tf.nn.relu(encode_out)

    #for sparsity
    sparsity = tf.nn.zero_fraction(encode_out)
    non_zero_count = tf.count_nonzero(encode_out)
    sparsity_loss = tf.reduce_mean(tf.abs(encode_out))


    decode_filters = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, kernel_num, 1], 0.0, 0.2))
    decode_biases = tf.Variable(tf.truncated_normal([1], 0, 0.2))
    reconstruct = tf.nn.bias_add(tf.nn.conv2d(encode_out, decode_filters, [1, 1, 1, 1], 'SAME'), decode_biases)

    loss = tf.losses.mean_squared_error(input_batch, reconstruct) + 0.5 * sparsity_loss

    optimizer = tf.train.AdamOptimizer()
    return encode_filters, (sparsity, non_zero_count), reconstruct, loss, optimizer.minimize(loss)


def build_conv_autoencoder(kernel_size, kernel_num, upsampling_method='resize-convolution', input_name='input'):
    input_batch = tf.placeholder(tf.float32, name=input_name)

    encode_filters = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, 1, kernel_num], 0.0, 0.2))
    encode_biases = tf.Variable(tf.truncated_normal([kernel_num], 0, 0.2))

    encode_out = tf.nn.bias_add(tf.nn.conv2d(input_batch, encode_filters, [1, 1, 1, 1], 'SAME'), encode_biases)
    encode_out = tf.nn.relu(encode_out)

    encode_out = tf.nn.max_pool(encode_out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')


    reconstruct = None
    if upsampling_method == 'resize-convolution':

        interpolated = tf.image.resize_nearest_neighbor(encode_out, tf.shape(input_batch)[1: 3])
        decode_filters = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, kernel_num, 1], 0.0, 0.2))
        decode_biases = tf.Variable(tf.truncated_normal([1], 0, 0.2))
        reconstruct = tf.nn.bias_add(tf.nn.conv2d(interpolated, decode_filters, [1, 1, 1, 1], 'SAME'), decode_biases)

    elif upsampling_method == 'deconvolution':

        decode_filters = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, 1, kernel_num], 0.0, 0.2))
        reconstruct = tf.nn.conv2d_transpose(encode_out, decode_filters, tf.shape(input_batch), [1, 2, 2, 1], padding='SAME')

    loss = tf.losses.mean_squared_error(input_batch, reconstruct)

    optimizer = tf.train.AdamOptimizer()
    return reconstruct, loss, optimizer.minimize(loss)

def train_sparse(sess, loss_op, opt_op, sparsity_ops, epoch, batch_size, data_num):
    for e in range(epoch):
        loss_avg = 0
        sparsity_avg = 0
        nz_count_avg = 0
        for b in range(0, data_num, batch_size):
            batch = norm_train[b: b + batch_size]
            loss_val, sparsity, non_zero_count, _ = sess.run([loss_op, *sparsity_ops, opt_op], {'input:0': batch})
            batch_fraction = batch.shape[0] / data_num
            loss_avg += loss_val * batch_fraction
            sparsity_avg += sparsity * batch_fraction
            nz_count_avg += non_zero_count * batch_fraction

        print('epoch %s: loss: %s sparsity: %s nz count: %s' % (e, loss_avg, sparsity_avg, nz_count_avg))

def train_normal(sess, loss_op, opt_op, epoch, batch_size, data_num):
    for e in range(epoch):
        loss_avg = 0

        for b in range(0, data_num, batch_size):
            batch = norm_train[b: b + batch_size]
            loss_val, _ = sess.run([loss_op, opt_op], {'input:0': batch})
            batch_fraction = batch.shape[0] / data_num
            loss_avg += loss_val * batch_fraction

        print('epoch %s: loss: %s' % (e, loss_avg))

def train_and_evaluate(sess, loss_op, opt_op, reconstruct_op, sparsity_ops=None):
    sess.run(tf.global_variables_initializer())

    epoch = 50
    batch_size = 256
    data_num = X_train.shape[0]

    if sparsity_ops:
        train_sparse(sess, loss_op, opt_op, sparsity_ops, epoch, batch_size, data_num)
    else:
        train_normal(sess, loss_op, opt_op, epoch, batch_size, data_num)

    # Full training MSE
    evaluate(reconstruct_op, X_train, norm_train, batch_size, data_num, 'training')
    evaluate(reconstruct_op, X_test, norm_test, batch_size, data_num, 'test')



def evaluate(reconstruct_op, x, norm_x, batch_size, data_num, data_name):
    # Full training MSE
    r_x_data = np.zeros_like(x, dtype=np.uint8)
    for b in range(0, data_num, batch_size):
        batch = norm_x[b: b + batch_size]
        r_norm_data = sess.run(reconstruct_op, {'input:0': batch})
        r_x_data[b: b + batch_size] = denormalize(r_norm_data, mean)

    loss_avg = np.mean(np.square(r_x_data - x))
    print('%s reconstruction loss: %s' % (data_name, loss_avg))


def show_filters(sess, kernel_var):
    kernels = np.squeeze(sess.run(kernel_var))
    rows = 4
    cols = kernels.shape[2] // rows
    for i in range(kernels.shape[2]):
        plt.subplot(rows, cols, i + 1)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(kernels[:, :, i])

    plt.show()

def vis_test_econstruct():
    sample_indices = [0, 1, 2, 4, 6, 8, 9, 13, 18, 19]

    r_norm_data = sess.run(reconstruct, {'input:0': norm_test[:max(sample_indices) + 1]})
    r_test_data = denormalize(r_norm_data, mean)

    for i, ind in enumerate(sample_indices):
        original_image = np.squeeze(X_test[ind])
        recon_image = np.squeeze(r_test_data[ind])
        plt.subplot(2, 10, i + 1)
        plt.xticks([], [])
        plt.yticks([], [])
        # plt.imshow(original_image, cmap='Greys_r')
        # plt.subplot(2, 10, i+11)
        # plt.xticks([], [])
        # plt.yticks([], [])
        plt.imshow(recon_image, cmap='Greys_r')
    plt.tight_layout()
    plt.show()



KERNEL_NUM = 16
KERNEL_SIZE = 5
f_mnist_data = scipy.io.loadmat('fashion_mnist_dataset.mat')
X_train = f_mnist_data['X_train']
X_test = f_mnist_data['X_test']

IMG_SIZE = int(np.sqrt(X_train.shape[1]))

X_train = reshape_to_images(X_train, IMG_SIZE)
X_test = reshape_to_images(X_test, IMG_SIZE)

norm_train, mean = normalize(X_train)
norm_test, _ = normalize(X_test, mean)
kernel_var, sparsity, reconstruct, loss, opt = build_sparse_conv_autoencoder(KERNEL_SIZE, KERNEL_NUM)
# reconstruct, loss, opt = build_conv_autoencoder(KERNEL_SIZE, KERNEL_NUM, upsampling_method='resize-convolution')
sess = tf.InteractiveSession()

train_and_evaluate(sess, loss, opt, reconstruct, sparsity)
# train_and_evaluate(sess, loss, opt, reconstruct)

show_filters(sess, kernel_var)


#Visualization on test set
