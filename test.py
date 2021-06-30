import os
from icecream import ic 
from argparse import ArgumentParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
from losses import CustomTripletLoss
tf.keras.losses.TripletSemiHardLoss = tfa.losses.TripletSemiHardLoss

from model import InceptionResNetV1, get_l2_norm_model
from evaluators import RandomPairEval
from dataloaders import RandomPairGenerator



def log_line(f, line):
    print(line)
    f.write(line + '\n')

    
def run(args):
    if args.gpus != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    real_model = InceptionResNetV1(
        weights_path='pretrained_facenet/facenet_keras_weights.h5')
    real_model = get_l2_norm_model(real_model)
    
    if args.model != '':
        fake_model = load_model(args.model, custom_objects={'loss_function': CustomTripletLoss()})
    else:
        fake_model = InceptionResNetV1(
            weights_path=args.weights)
        fake_model = get_l2_norm_model(fake_model)
    

    if args.random:
        evaluator = RandomPairEval(
            real_model=real_model,
            fake_model=fake_model,
            data_generator=RandomPairGenerator(args.datadir, args.anno)
        )
        res_path = f'results/{args.runname}/'
        os.makedirs(res_path, exist_ok=True)
        with open(res_path + 'log.txt', 'w') as f:
            log_line(f, 'Evaluation of random pairs')
            if args.model != '':    
                log_line(f, f'model path: {args.model}')
            else:
                log_line(f, f'weights path: {args.weights}')
            log_line(f, f'data directory: {args.datadir}')
            log_line(f, f'n_sample = {args.samples}')
            
            dom1 = 'fake'
            for dom2 in ('fake', 'real'):
                for same in (True, False):
                    distances = evaluator.get_distances(args.samples, dom1, dom2, same)
                    VAL = evaluator.VAL(distances=distances)
                    FAR = evaluator.FAR(distances=distances)
                    evaluator.plot_distance_distribution(
                        dom1=dom1,
                        dom2=dom2,
                        same=same,
                        distances=distances,
                        savedir=res_path
                    )
                    log_line(f, '-' * 50)
                    log_line(f, f'{dom1} vs. {dom2}, same = {same}')
                    if same:
                        log_line(f, f'VAL = {VAL}')
                    else:
                        log_line(f, f'FAR = {FAR}')
        


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--runname', type=str, required=True,
                        help='Run name')
    parser.add_argument('-d', '--datadir', type=str, required=True,
                        help='Root directory of dataset')
    parser.add_argument('-a', '--anno', type=str, required=True,
                        help='Path to annotation file')
    parser.add_argument('-m', '--model', type=str, default='',
                        help='Path to model')
    parser.add_argument('-w', '--weights', type=str, default='',
                        help='Path to model weights, overwrited by -m')
    parser.add_argument('-s', '--samples', type=int, default=10,
                        help='Number of samples')
    parser.add_argument('-g', '--gpus', type=str, default='',
                        help='List of GPU to use. e.g. 0,1')
    parser.add_argument('-r', '--random', action='store_true',
                        help='Random pair selection flag')
    args = parser.parse_args()
    run(args)
    