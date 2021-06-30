import os
import sys
import shutil
from omegaconf import OmegaConf
from argparse import ArgumentParser
from logger import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow_addons.losses import TripletSemiHardLoss

from model import InceptionResNetV1, get_l2_norm_model
from dataloaders import SimpleTripletGenerator


def train(args):
    chkpt_dir = f'checkpoints/train_simple/{args.runname}/'
    log_dir = f'log/train_simple/{args.runname}/'
    os.makedirs(chkpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    with open(log_dir + 'args.txt', 'w') as f:
        f.write(repr(vars(args)))
    shutil.copyfile(args.config, log_dir + 'config.yaml')

    if args.gpus != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    cf = OmegaConf.load(args.config)

    # Data generators
    train_gen = SimpleTripletGenerator(
        root_dirs=cf.path.train_dir,
        anno_path=cf.path.ID_annotation,
        batch_size=cf.train.batch_size
    )
    val_gen = SimpleTripletGenerator(
        root_dirs=cf.path.val_dir,
        anno_path=cf.path.ID_annotation,
        batch_size=cf.train.batch_size
    )

    # Callbacks
    last_cp = ModelCheckpoint(filepath=chkpt_dir + 'last.h5')
    best_cp = ModelCheckpoint(
        filepath=chkpt_dir + 'best_val.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
    )

    # Model
    model = InceptionResNetV1(weights_path=cf.path.checkpoint_weights)
    model = get_l2_norm_model(model)
    model.compile(
        optimizer=Adam(
            learning_rate=cf.train.learning_rate,
            beta_1=cf.train.beta_1,
            beta_2=cf.train.beta_2
        ),
        loss=TripletSemiHardLoss(margin=cf.train.margin)
    )

    # Train
    sys.stdout = Logger(log_dir + 'train.txt')
    model.fit(
        train_gen,
        epochs=100,
        callbacks = [last_cp, best_cp],
        validation_data=val_gen,
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--runname', type=str, required=True,
                        help='Run name')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('-g', '--gpus', type=str, default='',
                        help='List of GPUs to be used, in default, use CPU only. e.g. 0,1')
    args = parser.parse_args()

    train(args)
    