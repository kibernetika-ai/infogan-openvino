import logging
import PIL.Image
import io
import numpy as np
import pickle
import os
import base64
import json
import math

LOG = logging.getLogger(__name__)


def log(func):
    def decorator(*args, **kwargs):
        LOG.info('Running %s...' % func.__name__)
        return func(*args, **kwargs)

    return decorator


all_features = []
all_paths = []


@log
def init_hook(**params):
    LOG.info("Init hooks {}".format(params))
    model_path = params['model_path']
    base_path = os.path.dirname(model_path)
    with open(os.path.join(base_path, 'features.pkl'), 'rb') as input:
        global all_features
        all_features = pickle.load(input)
        global all_paths
        all_paths = pickle.load(input)
    LOG.info("Init hooks")


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


@log
def preprocess(inputs, **kwargs):
    LOG.info('Preprocess: {}, args: {}'.format(inputs, kwargs))
    images = inputs['image']
    batch = []
    for image in images:
        image = PIL.Image.open(io.BytesIO(image)).convert("RGB")
        image = image.resize((64, 64))
        image = np.asarray(image)
        image = image / 127.5 - 1
        image = np.transpose(image, (2, 0, 1))
        batch.append(image)
    batch = np.stack(batch)
    LOG.info('Batch shape: {}'.format(batch.shape))
    return {'image_placeholder': batch}


clip = 1e-3


@log
def postprocess(outputs, **kwargs):
    logging.info('Postprocess: {}, args: {}'.format(outputs, kwargs))
    for k, v in outputs.items():
        outputs = v
        logging.info('Use {} as output'.format(k))
        break
    np.clip(outputs, -clip, clip, outputs)
    new_im = PIL.Image.new('RGB', (64 * 10, 64 * len(outputs)))
    y = 0
    for o in outputs:
        p = np.sum((all_features - o) ** 2, axis=1)
        indexes = p.argsort()[0:min(len(all_features), 10)]
        x = 0
        table = []
        for i in indexes:
            im_file = all_paths[i]
            image = PIL.Image.open(im_file).convert("RGB")
            image = image.resize((64, 64))
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='JPEG', quality=80)
            b_val = image_bytes.getvalue()
            encoded = base64.encodebytes(b_val).decode()
            new_im.paste(image, (x, y))
            table.append(
                {
                    'type': 'shoes',
                    'name': im_file,
                    'prob': sigmoid(float(p[i])),
                    'image': encoded
                }
            )
            x += 64
        y += 64
    with io.BytesIO() as output:
        new_im.save(output, format='PNG')
        contents = output.getvalue()
    table = json.dumps(table)
    return {'output': contents, 'table_output': table}
