# InfoGAN
## InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets.
See [https://arxiv.org/pdf/1606.03657.pdf](https://arxiv.org/pdf/1606.03657.pdf) for details.

## Tasks description.

### Standalone Training.
Used to train InfoGAN model [https://arxiv.org/pdf/1606.03657.pdf](https://arxiv.org/pdf/1606.03657.pdf) on the Zappos dataset.

Execution command:
```
python main.py --gan --logdir=$TRAINING_DIR/$BUILD_ID --batch_size=5 --file_pattern=$DATA_DIR/*/*/*/*.jpg --epochs=1
```
Argumets:

* batch_size - Batch size for trining.
* epochs - number epochs. One is enough for model testing, for real training at least five epochs should be used.
* logdir - directory to save trained model and log metrics for viewing with tensorboard. Use provided enviroment variables to set this parameter.

<mark>Training this model on CPU is to slow, please request your administrator to obtain sufficient GPU quotas.<mark>

### Parallel Training.
Used to train InfoGAN model [https://arxiv.org/pdf/1606.03657.pdf](https://arxiv.org/pdf/1606.03657.pdf) on the Zappos dataset.
Same as the standalone training, but use distrubeted tensorflow configuration to train model on multiple nodes or GPU cards.
Execution command for workers:
```
python parallel.py --role=worker --num_gpus=$GPU_COUNT --task $REPLICA_INDEX --ps_hosts $PS_NODES --worker_hosts $WORKER_NODES --logdir=$TRAINING_DIR/$BUILD_ID --batch_size=5 --file_pattern=$DATA_DIR/*/*/*/*.jpg --epochs=1
```
Execution command for parameter server:
```
python parallel.py --role=ps --task $REPLICA_INDEX --ps_hosts $PS_NODES --worker_hosts $WORKER_NODES
```
Argumets:

* batch_size - Batch size for trining.
* epochs - number epochs. One is enough for model testing, for real training at least five epochs should be used.
* task - trianing process index. Use provided enviroment variable to set this parameter.
* num_gpus - number GPU availble. Use provided enviroment variable to set this parameter.
* ps_hosts - adresses of parameter servers. Use provided enviroment variable to set this parameter.
* worker_hosts - adresses of workers. Use provided enviroment variable to set this parameter.
* role - role of task - 'worker' or 'ps'.
* logdir - directory to save trained model and log metrics for viewing with tensorboard. Use provided enviroment variables to set this parameter.


<mark>Training this model on CPU is to slow, please request your administrator to obtain sufficient GPU quotas.<mark>

### Similarity
Extract feature vector from provided datasets. This is a demo step that required to start serving. This step can be included into sering process directly.

Execution command:
```
python main.py --similarity --logdir=$TRAINING_DIR/2 --batch_size=5 --file_pattern=$DATA_DIR/*/*/*/*.jpg
```
Arguments:

* logdir - directory to save trained model and log metrics for viewing with tensorboard. Use provided enviroment variables to set this parameter and any build number from previous step.
