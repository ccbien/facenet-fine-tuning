import os
import matplotlib.pyplot as plt
import numpy as np

from dataloaders import RandomPairGenerator
from utils import normalize_input


class RandomPairEval:
    def __init__(self, real_model, fake_model, data_generator: RandomPairGenerator):
        self.real_model = real_model
        self.fake_model = fake_model
        self.datagen = data_generator

    
    def get_distances(self, n_samples, dom1, dom2, same=True, batch_size=10):
        if n_samples % batch_size != 0:
            raise ValueError('n_samples must be divisible by batch_size')
        image1s, image2s = [], []
        for i in range(n_samples):
            if same:
                image1, image2 = self.datagen.get_same(dom1, dom2)
            else:
                image1, image2 = self.datagen.get_diff(dom1, dom2)
            image1s.append(image1)
            image2s.append(image2)

        y1, y2 = [], []
        for images, ys, dom in ((image1s, y1, dom1), (image2s, y2, dom2)):
            for i in range(n_samples // batch_size):
                x = np.array(images[i*batch_size: (i+1)*batch_size])
                x = normalize_input(x)
                if dom == 'real':
                    y = self.real_model.predict(x)
                else:
                    y = self.fake_model.predict(x)
                for j in range(batch_size):
                    ys.append(y[j,:])
        
        

        distances = []
        for i in range(n_samples):
            dist = np.sqrt(np.sum((y1[i] - y2[i])**2))
            distances.append(dist)
        return distances


    def VAL(self, n_samples=None, dom1=None, dom2=None, threshold=1.1, distances=None):
        """Validation rate"""
        if distances is None:
            distances = self.get_distances(n_samples, dom1, dom2, same=True)
        else:
            n_samples = len(distances)
        return sum([d <= threshold for d in distances]) / n_samples


    def FAR(self, n_samples=None, dom1=None, dom2=None, threshold=1.1, distances=None):
        """False-Accept rate"""
        if distances is None:
            distances = self.get_distances(n_samples, dom1, dom2, same=False)
        else:
            n_samples = len(distances)
        return sum([d <= threshold for d in distances]) / n_samples

    def plot_distance_distribution(self, n_samples=None, dom1=None, dom2=None, same=None, show=False, savedir=None, distances=None):
        plt.figure()
        if distances is None:
            distances = self.get_distances(n_samples, dom1, dom2, same=same)
        else:
            n_samples = len(distances)
        hist = plt.hist(distances, bins=100, range=(0, 2))
        if show:
            plt.show()
        if savedir:
            tmp = 'same' if same else 'diff'
            file = f'{n_samples}_{dom1}_{dom2}_{tmp}.png'
            plt.savefig(os.path.join(savedir, file))


    