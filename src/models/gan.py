import tensorflow as tf
import numpy as np
from tensorflow.python.training import session_run_hook
import glob
import logging

IMAGE_SIZE = (64, 64)
Z_DIM = 128
C_DIM = 1
D_DIM = 10
TINY = 1e-8


def null_dataset():
    def _input_fn():
        return None

    return _input_fn


def eval_fn():
    inputs = ['0']

    def _input_fn():
        ds = tf.data.Dataset.from_tensor_slices(inputs)

        def _fake_image(_):
            z = tf.zeros([1, 1, 1, 3], dtype=tf.float32)
            return {'image': z}, 0

        ds = ds.map(_fake_image, num_parallel_calls=1)
        return ds

    return _input_fn


def input_fn(params, is_training):
    limit = params['limit_train']
    if limit is None:
        limit = -1

    inputs = []
    i = 0
    for f in glob.iglob(params['data_set'], recursive=True):
        inputs.append(f)
        i += 1
        if (limit > 0) and (i >= limit):
            break
    logging.info('Training files count: {}'.format(len(inputs)))

    def _input_fn():
        ds = tf.data.Dataset.from_tensor_slices(inputs)
        if is_training:
            ds = ds.repeat(count=params['epoch']).shuffle(params['batch_size'] * 10)

        def _read_image(filename):
            image = tf.image.decode_jpeg(tf.read_file(filename), 3)
            image = tf.image.resize_images(image, IMAGE_SIZE)
            image = image / 127.5 - 1
            return {'image': image}, 0

        ds = ds.map(_read_image, num_parallel_calls=1)
        if is_training:
            ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(params['batch_size']))
        else:
            ds = ds.padded_batch(params['batch_size'], padded_shapes=(
                {'image': [IMAGE_SIZE[0], IMAGE_SIZE[1], 3]}, tf.TensorShape([])))
        return ds

    return inputs, _input_fn


# TF
# valid
# input_length * stride + kernel - stride
# full
# input_length * stride - stride - kernel + 2
def generator(z, latent_c, latent_d, mode):
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    with tf.variable_scope("gen"):
        net = z
        if latent_c is not None:
            net = tf.concat([net, latent_c], axis=1)
        if latent_d is not None:
            net = tf.concat([net, latent_d], axis=1)

        net = tf.layers.dense(net, 1024)
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.relu(net)
        logging.info('GShape Dense 1 {}'.format(net.shape))
        net = tf.layers.dense(net, 4 * 4 * 128)
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.relu(net)
        net = tf.reshape(net, [-1, 4, 4, 128])
        logging.info('GShape Dense 2 {}'.format(net.shape))
        conv = [64, 32, 16]
        for i, f in enumerate(conv):
            net = tf.layers.conv2d_transpose(net, f, 3, strides=(2, 2), padding='same')
            net = tf.layers.batch_normalization(net, training=training)
            net = tf.nn.relu(net)
            logging.info('GShape DeConv{} {}'.format(i, net.shape))
        net = tf.layers.conv2d_transpose(net, 3, 3, padding='same', strides=(2, 2), activation=tf.nn.tanh)
        logging.info('GShape DeConv{} {}'.format(len(conv), net.shape))
        img = tf.clip_by_value((net + 1) / 2, 0, 1, name='gout')
        if training:
            tf.summary.image("gen", net, max_outputs=8)
            return net
        else:
            return img


def discriminator(input, reuse, mode, global_step, dropout=None, d_dim=0, c_dim=0,noise_level=100):
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    with tf.variable_scope("discr", reuse=reuse):
        def _noise(net):
            if training and noise_level>0:
                sigma = tf.log(4.0) * 0.2 / tf.log(4.0 + tf.cast(global_step, tf.float32) / float(noise_level))
                return net + tf.random_normal(shape=net.shape, mean=0.0, stddev=sigma, dtype=tf.float32)
            else:
                return net

        def _dropout(net):
            if training and dropout is not None and dropout>0:
                return tf.layers.dropout(net, dropout, name='dropout')
            return net

        logging.info('DShape Input {}'.format(input.shape))
        # [64x64]->[32,32]
        convs = [64, 128, 256]
        net = _noise(input)
        net = tf.layers.conv2d(net, 64, 3, strides=(2, 2), padding='same', reuse=reuse, name='conv{}'.format(0))
        net = tf.nn.leaky_relu(net, alpha=0.1)
        logging.info('DShape Conv{} {}'.format(0, input.shape))
        for i, f in enumerate(convs):
            net = tf.layers.conv2d(net, f, 3, strides=(2, 2), padding='same', reuse=reuse, name='conv{}'.format(i + 1))
            net = tf.layers.batch_normalization(net, training=training, name='conv{}_norm'.format(i + 1))
            net = tf.nn.leaky_relu(net, alpha=0.1)
            net = _dropout(net)
            logging.info('DShape Conv{} {}'.format(i + 1, net.shape))

        net = tf.layers.flatten(net)
        logging.info('DShape Flatten {}'.format(net.shape))
        net = tf.layers.dense(net, 1024, reuse=reuse, name='out')
        net = tf.layers.batch_normalization(net, training=training, name='out_norm')
        net = tf.nn.leaky_relu(net, alpha=0.1, name='pout')
        if mode != tf.estimator.ModeKeys.TRAIN:
            return net
        d_net = tf.layers.dense(net, 1, reuse=reuse, name='out_z', activation=tf.nn.sigmoid)
        if c_dim is None and d_dim is None:
            return d_net

    with tf.variable_scope("latent_c"):
        net = tf.layers.dense(net, 128, name='out_q')
        net = tf.layers.batch_normalization(net, training=training, name='out_q_norm')
        net = tf.nn.leaky_relu(net, alpha=0.1)
        if d_dim > 0:
            qd_net = tf.layers.dense(net, d_dim, activation=tf.nn.softmax, name="out_qd")
        else:
            qd_net = None
        if c_dim > 0:
            mean = tf.layers.dense(net, c_dim, name="out_qc_mean")
            # logstd = tf.layers.dense(net, c_dim, name="out_qc_logstd")
            # logstd = tf.maximum(logstd, -16)
            qc_net = mean
        else:
            qc_net = None
        return d_net, qc_net, qd_net


def gaussian_loss(y_true, y_pred):
    mean = y_pred[:, :y_true.shape[1]]
    logstd = y_pred[:, y_true.shape[1]:]
    epsilon = (y_true - mean) / (tf.exp(logstd) + TINY)
    loss = (logstd + 0.5 * tf.square(epsilon))
    loss = tf.reduce_mean(loss)
    return loss


def model_fn(features, labels, mode, params=None, config=None, model_dir=None):
    features = features['image']
    logging.info('Features shape: {}'.format(features.shape))
    global_step = tf.train.get_or_create_global_step()
    metrics = {}
    training_hooks = []
    evaluation_hooks = []
    export_outputs = None
    d_trainer = None
    if mode == tf.estimator.ModeKeys.PREDICT:
        d_real = discriminator(features, False, mode, 0, dropout=None)
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                d_real)}
        loss = None
    elif mode == tf.estimator.ModeKeys.EVAL:
        #z = tf.random_uniform([D_DIM * 10, Z_DIM], maxval=1., minval=0., dtype=tf.float32)
        z = tf.zeros([D_DIM * 10, Z_DIM], dtype=tf.float32)
        tmp1 = np.zeros((D_DIM * 10, C_DIM), dtype=np.float32)
        for k in range(D_DIM):
            tmp1[k * 10:(k + 1) * 10, 0] = np.linspace(-2, 2, 10)
        tmp2 = np.zeros((D_DIM * 10, D_DIM), dtype=np.float32)
        for k in range(D_DIM):
            tmp2[k * 10:(k + 1) * 10, k] = 1
        tmp1 = tf.constant(tmp1, dtype=tf.float32)
        tmp2 = tf.constant(tmp2, dtype=tf.float32)
        g_fake = generator(z, tmp1, tmp2, mode)
        images = []
        for i in range(D_DIM):
            h = []
            for j in range(10):
                im = g_fake[i * 10 + j]
                h.append(im)
            h = tf.concat(h, axis=1)
            images.append(h)
        images = tf.concat(images, axis=0)
        images = tf.stack([images])
        loss = tf.reduce_mean(images)
        # metrics['loss'] = tf.metrics.mean(images)
        d_real = None
        tf.summary.image("demo", images, max_outputs=1)
        eval_summary_hook = tf.train.SummarySaverHook(
            save_steps=1,
            output_dir=model_dir + "/demo",
            summary_op=tf.summary.merge_all())
        evaluation_hooks.append(eval_summary_hook)
    else:
        d_real = discriminator(features, False, mode, global_step, dropout=params['dropout'], c_dim=None, d_dim=None,noise_level=params['noise_level'])
        z = tf.random_uniform([params['batch_size'], Z_DIM], maxval=1., minval=0., dtype=tf.float32)
        latent_c = tf.random_normal([params['batch_size'], C_DIM], dtype=tf.float32) * 0.5 if C_DIM > 0 else None
        if D_DIM > 0:
            prior_d = tf.ones([params['batch_size'], D_DIM], dtype=tf.float32) / D_DIM
            p = tf.random_uniform([params['batch_size']], maxval=D_DIM, dtype=tf.int32) + tf.constant(
                [i * D_DIM for i in range(params['batch_size'])], dtype=tf.int32)
            p = tf.reshape(p, [params['batch_size'], 1])
            latent_d = tf.scatter_nd(p, tf.ones([params['batch_size']], dtype=tf.float32),
                                     tf.constant([params['batch_size'] * D_DIM], dtype=tf.int32))
            latent_d = tf.reshape(latent_d, [params['batch_size'], D_DIM])
        else:
            latent_d = None
        g_fake = generator(z, latent_c, latent_d, mode)
        d_fake, q_c, q_d = discriminator(g_fake, True, mode, global_step, dropout=params['dropout'], c_dim=C_DIM,
                                         d_dim=D_DIM,noise_level=params['noise_level'])

        t_vars = tf.trainable_variables()
        logging.info('tvars: {}'.format(t_vars))

        d_loss_a = -tf.reduce_mean(tf.log(d_real + TINY) + tf.log(1 - d_fake + TINY))
        tf.summary.scalar('d_loss_a', d_loss_a)
        # Mutual Information Loss
        if C_DIM > 0:
            def _gausina_logli(x, mean, stddev):
                epsilon = (x - mean) / stddev
                return tf.reduce_sum(- 0.5 * np.log(2 * np.pi) - tf.log(stddev) - 0.5 * tf.square(epsilon), 1)

            cont_cross_ent = tf.reduce_mean(_gausina_logli(latent_c, q_c, 0.5))
            cont_ent = tf.reduce_mean(_gausina_logli(latent_c, 0, 0.5))
            q_loss_c = cont_ent - cont_cross_ent
        else:
            q_loss_c = 0
        tf.summary.scalar('q_loss_c', q_loss_c)
        if D_DIM > 0:
            disc_log_q_c_given_x = tf.reduce_sum(tf.log(q_d + TINY) * latent_d, 1)
            disc_log_q_c = tf.reduce_sum(tf.log(prior_d + TINY) * latent_d, 1)
            disc_cross_ent = tf.reduce_mean(disc_log_q_c_given_x)
            disc_ent = tf.reduce_mean(disc_log_q_c)
            q_loss_d = disc_ent - disc_cross_ent
        else:
            q_loss_d = 0

        tf.summary.scalar('q_loss_d', q_loss_d)

        d_loss = d_loss_a + q_loss_c + 1.0 * q_loss_d
        tf.summary.scalar('d_loss', d_loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            d_trainer = tf.train.AdamOptimizer(params['discriminator_learning_rate'], beta1=.5).minimize(
                d_loss, global_step=global_step,
                var_list=[v for v in t_vars if 'discr/' in v.name or 'latent_c/' in v.name])

        g_loss_a = -tf.reduce_mean(tf.log(d_fake + TINY))
        tf.summary.scalar('g_loss_a', g_loss_a)

        g_loss = g_loss_a + 1 * q_loss_c + 1.0 * q_loss_d
        tf.summary.scalar('g_loss', g_loss)
        with tf.control_dependencies(update_ops):
            g_trainer = tf.train.AdamOptimizer(params['generator_learning_rate'], beta1=.5).minimize(
                g_loss, var_list=[v for v in t_vars if 'gen/' in v.name])

        training_hooks = [UpdateGeneratorHook(g_trainer)]
        loss = d_loss + g_loss

    return tf.estimator.EstimatorSpec(
        mode=mode,
        eval_metric_ops=metrics,
        predictions=d_real,
        loss=loss,
        training_hooks=training_hooks,
        export_outputs=export_outputs,
        evaluation_hooks=evaluation_hooks,
        train_op=d_trainer)


class UpdateGeneratorHook(session_run_hook.SessionRunHook):
    def __init__(self, g_train):
        self._g_train = g_train

    def begin(self):
        None

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return None

    def after_run(self, run_context, run_values):
        run_context.session.run([self._g_train])
        return None


class INFOGAN(tf.estimator.Estimator):
    def __init__(
            self,
            params=None,
            model_dir=None,
            config=None,
            warm_start_from=None
    ):
        def _model_fn(features, labels, mode, params, config):
            return model_fn(
                features=features,
                labels=labels,
                mode=mode,
                params=params,
                config=config,
                model_dir=model_dir
            )

        super(INFOGAN, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )
