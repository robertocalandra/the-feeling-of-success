import tensorflow as tf, numpy as np

n = 1000
d = 32

def train():
    # w*x
    np.random.seed(0)
    w_gt = np.float32(np.random.randn(d))
    x = np.float32(np.random.randn(n, d))
    y_gt = np.dot(x, w_gt)

    gpu = '/cpu:0'
    print 'w_gt =', w_gt
    with tf.Graph().as_default(), tf.device(gpu), tf.Session() as sess:
        w = tf.get_variable('w', d, dtype = tf.float32,
                            initializer = tf.truncated_normal_initializer())
        x_tf = tf.constant(x)

        y_hat = x_tf * tf.expand_dims(w, 0)
        y_hat = tf.reduce_sum(y_hat, 1)
        r = y_hat - tf.constant(y_gt)
        loss = tf.reduce_mean(r**2)

        opt = tf.train.MomentumOptimizer(0.01, 0.9)
        grad_step = opt.minimize(loss)

        sess.run(tf.global_variables_initializer())

        for i in xrange(10000):
            _, loss_val, w_val = sess.run([grad_step, loss, w])
            print 'Iteration:', i, 'Loss:', loss_val

if __name__ == '__main__':
    train()