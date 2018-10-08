import tensorflow as tf
import argparse
import os
import logging
import configparser
import models.gan as gan
import json
import numpy as np
import pickle
import sys
from mlboardclient.api import client

mlboard = client.Client()


def parse_args():
    conf_parser = argparse.ArgumentParser(
        add_help=False
    )
    conf_parser.add_argument(
        '--checkpoint_dir',
        default=os.environ.get('TRAINING_DIR', 'training') + '/' + os.environ.get('BUILD_ID', '1'),
        help='Directory to save checkpoints and logs',
    )
    args, remaining_argv = conf_parser.parse_known_args()
    parser = argparse.ArgumentParser(
        parents=[conf_parser],
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    checkpoint_dir = args.checkpoint_dir
    logging.getLogger().setLevel('INFO')
    tf.logging.set_verbosity(tf.logging.INFO)
    parser.add_argument(
        '--limit_train',
        type=int,
        default=-1,
        help='Limit number files for train. For testing.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size.',
    )
    parser.add_argument(
        '--discriminator_learning_rate',
        type=float,
        default=2e-4,
        help='Recommended learning_rate is 2e-4',
    )
    parser.add_argument(
        '--generator_learning_rate',
        type=float,
        default=1e-3,
        help='Recommended learning_rate is 1e-3',
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=None,
        help='Drop out to use',
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=1,
        help='Epoch to trian',
    )
    parser.add_argument(
        '--save_summary_steps',
        type=int,
        default=10,
        help="Log summary every 'save_summary_steps' steps",
    )
    parser.add_argument(
        '--save_checkpoints_secs',
        type=int,
        default=600,
        help="Save checkpoints every 'save_checkpoints_secs' secs.",
    )
    parser.add_argument(
        '--save_checkpoints_steps',
        type=int,
        default=100,
        help="Save checkpoints every 'save_checkpoints_steps' steps",
    )
    parser.add_argument(
        '--keep_checkpoint_max',
        type=int,
        default=5,
        help='The maximum number of recent checkpoint files to keep.',
    )
    parser.add_argument(
        '--log_step_count_steps',
        type=int,
        default=100,
        help='The frequency, in number of global steps, that the global step/sec and the loss will be logged during training.',
    )
    parser.add_argument(
        '--data_set',
        default=None,
        help='Location of training files or evaluation files',
    )
    parser.add_argument(
        '--openvino',
        dest='openvino',
        action='store_true',
        help='Do convert to OpenVino format',
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.set_defaults(worker=False)
    group.set_defaults(evaluator=False)
    group.set_defaults(test=False)
    group.add_argument('--worker', dest='worker', action='store_true',
                       help='Start in Worker(training) mode.')
    group.add_argument('--evaluator', dest='evaluator', action='store_true',
                       help='Start in evaluation mode')
    group.add_argument('--test', dest='test', action='store_true',
                       help='Test mode')
    p_file = os.path.join(checkpoint_dir, 'parameters.ini')
    if tf.gfile.Exists(p_file):
        parameters = configparser.ConfigParser(allow_no_value=True)
        parameters.read(p_file)
        parser.set_defaults(**dict(parameters.items("PARAMETERS", raw=True)))
    args = parser.parse_args(remaining_argv)
    print('\n*************************\n')
    print(args)
    print('\n*************************\n')
    return checkpoint_dir, args


def make_openvino(exported_path):
    app = mlboard.apps.get()
    task = app.tasks.get('openvino')
    task.resource('converter')['workdir'] = exported_path
    task.resource('converter')['command'] = 'mo_tf.py --input_model frozen.pb --input_shape [1,64,64,3]'
    task.resource('converter')['args'] = {
        'input_model': 'frozen.pb',
        'input_shape': '[1,64,64,3]',
    }
    task.start(comment='Convert to OpenVino')
    completed = task.wait()
    if completed.status != 'Succeeded':
        logging.warning(
            "Task %s-%s completed with status %s."
            % (completed.name, completed.build, completed.status)
        )
        logging.warning(
            'Please take a look at the corresponding task logs'
            ' for more information about failure.'
        )
        logging.warning("Workflow completed with status ERROR")
        sys.exit(1)

    client.update_task_info({'openvino_model': os.path.join(exported_path, 'frozen.xml')})
    logging.info(
        "Task %s-%s completed with status %s."
        % (completed.name, completed.build, completed.status)
    )


def freeze(exported_path, do_openvino=False):
    from tensorflow.python.saved_model import loader
    from tensorflow.python.saved_model import signature_constants as tf_const
    sess = tf.Session(graph=tf.Graph())
    graph = loader.load(sess, [tf.saved_model.tag_constants.SERVING], exported_path)
    outputs = graph.signature_def.get(tf_const.DEFAULT_SERVING_SIGNATURE_DEF_KEY).outputs
    toutputs = []
    for out in list(outputs.values()):
        name = sess.graph.get_tensor_by_name(out.name).name
        p = name.split(':')
        name = p[0] if len(p) > 0 else name
        toutputs += [name]
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), toutputs)
    out_path = os.path.join(exported_path, 'frozen.pb')
    client.update_task_info({'frozen_graph': out_path})
    with tf.gfile.GFile(out_path, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    if do_openvino:
        make_openvino(exported_path)
    return out_path


def catalog_ref(name, ctype, version):
    return '#/{}/catalog/{}/{}/versions/{}'. \
        format(os.environ.get('WORKSPACE_NAME'), ctype, name, version)


def train(mode, checkpoint_dir, params):
    logging.info("start build  model")

    save_summary_steps = params['save_summary_steps']
    save_checkpoints_secs = params['save_checkpoints_secs'] if params['save_checkpoints_steps'] is None else None
    save_checkpoints_steps = params['save_checkpoints_steps']

    gpu = os.environ.get('GPUXX_Y', None)
    if gpu is None:
        session_config = None
    else:
        session_config = tf.ConfigProto()
        session_config.gpu_options.visible_device_list = gpu
    conf = tf.estimator.RunConfig(
        model_dir=checkpoint_dir,
        save_summary_steps=save_summary_steps,
        save_checkpoints_secs=save_checkpoints_secs,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=params['keep_checkpoint_max'],
        log_step_count_steps=params['log_step_count_steps'],
        session_config=session_config
    )

    net = gan.INFOGAN(
        params=params,
        model_dir=checkpoint_dir,
        config=conf,
    )
    logging.info("Start %s mode", mode)
    if mode == 'train':
        _, input_fn = gan.input_fn(params, True)
        net.train(input_fn=input_fn)
        logging.info('Export model')
        feature_placeholders = {
            'image': tf.placeholder(tf.float32, [1, gan.IMAGE_SIZE[0], gan.IMAGE_SIZE[1], 3],
                                    name='image_placeholder')
        }
        receiver = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_placeholders)
        export_path = net.export_savedmodel(checkpoint_dir, receiver)
        export_path = export_path.decode("utf-8")

        client.update_task_info({'model_path': export_path})

        logging.info('Build index to {}'.format(export_path))
        frozen = freeze(export_path, params['openvino'])
        logging.info('Save frozen graph: {}'.format(frozen))
        params['epoch'] = 1
        input_set, input_fn = gan.input_fn(params, False)
        predictions = net.predict(input_fn=input_fn)
        i = 0
        clip = 1e-3
        features = []
        for p in predictions:
            np.clip(p, -clip, clip, p)
            i += 1
            features.append(p)
            if i == len(input_set):
                break
        features = np.stack(features)
        with open(export_path + '/features.pkl', 'wb') as output:
            pickle.dump(features, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(input_set, output, pickle.HIGHEST_PROTOCOL)

        version = '1.{}.0'.format(os.environ['BUILD_ID'])
        mlboard.model_upload('infogan-similarity', version, export_path)
        client.update_task_info({'model': catalog_ref('infogan-similarity', 'mlmodel', version)})
        logging.info("New model uploaded as 'infogan-similarity', version '%s'." % (version))

    elif mode == 'eval':
        train_fn = gan.null_dataset()
        train_spec = tf.estimator.TrainSpec(input_fn=train_fn)
        eval_fn = gan.eval_fn()
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_fn, steps=1, start_delay_secs=10, throttle_secs=10)
        tf.estimator.train_and_evaluate(net, train_spec, eval_spec)

    else:
        logging.info("Not implemented")


def main():
    checkpoint_dir, args = parse_args()
    logging.info('------------------')
    logging.info('TF VERSION: {}'.format(tf.__version__))
    logging.info('ARGS: {}'.format(args))
    logging.info('------------------')
    if args.worker:
        mode = 'train'
    elif args.test:
        mode = 'test'
    else:
        mode = 'eval'
        cluster = {'chief': ['fake_worker1:2222'],
                   'ps': ['fake_ps:2222'],
                   'worker': ['fake_worker2:2222']}
        os.environ['TF_CONFIG'] = json.dumps(
            {
                'cluster': cluster,
                'task': {'type': 'evaluator', 'index': 0}
            })

    params = {

        'batch_size': args.batch_size,
        'discriminator_learning_rate': args.discriminator_learning_rate,
        'generator_learning_rate': args.generator_learning_rate,
        'save_summary_steps': args.save_summary_steps,
        'save_checkpoints_steps': args.save_checkpoints_steps,
        'save_checkpoints_secs': args.save_checkpoints_secs,
        'keep_checkpoint_max': args.keep_checkpoint_max,
        'log_step_count_steps': args.log_step_count_steps,
        'data_set': args.data_set,
        'epoch': args.epoch,
        'dropout': args.dropout,
        'openvino': args.openvino,
        'limit_train': args.limit_train
    }

    if not tf.gfile.Exists(checkpoint_dir):
        tf.gfile.MakeDirs(checkpoint_dir)

    train(mode, checkpoint_dir, params)


if __name__ == '__main__':
    main()
