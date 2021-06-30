import os
import random
import cv2 as cv
import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
from utils import normalize_input
from model import InceptionResNetV1, get_l2_norm_model


def is_image_file(filename):
    for ext in ('.jpg', '.jpeg', '.png'):
        if filename.lower().endswith(ext):
            return True
    return False


def load_image(image_path):
    image = cv.imread(image_path)
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


class RandomPairGenerator():
    def __init__(self, root_dirs, anno_path):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        all_files = []
        for root_dir in root_dirs:
            all_files.extend([
                os.path.join(root, file).replace('\\', '/')
                    for root, dirs, files in os.walk(root_dir)
                        for file in files
                            if is_image_file(file)
            ])
        
        self.ID = dict() # dict: filepath -> ID
        with open(anno_path, 'r') as f:
            for line in f.readlines():
                file, ID = line.split()
                ID = int(ID)
                self.ID[file] = ID

        self.domain = dict() # dict: filepath -> 'real' or 'fake'
        for file in all_files:
            domain = 'real' if '/photo/' in file else 'fake'
            self.domain[file] = domain
            
        self.files = dict() # (ID, domain) -> list of filepaths
        for file in all_files:
            ID = self.ID[file]
            dom = self.domain[file]
            if (ID, dom) not in self.files:
                self.files[(ID, dom)] = []
            self.files[(ID, dom)].append(file)


    def get_same(self, dom1, dom2):
        """return a tuple of two UINT8 numpy array"""
        ID, _ = random.choice(list(self.files.keys()))
        if dom1 == dom2:
            file1, file2 = random.sample(self.files[(ID, dom1)], 2)
        else:
            file1 = random.choice(self.files[(ID, dom1)])
            file2 = random.choice(self.files[(ID, dom2)])
        return load_image(file1), load_image(file2)

    
    def get_diff(self, dom1, dom2):
        """return a tuple of two UINT8 numpy array"""
        keys = random.sample(list(self.files.keys()), 3)
        ID1 = keys[0][0]
        ID2 = keys[1][0] if ID1 != keys[1][0] else keys[2][0]
        file1 = random.choice(self.files[(ID1, dom1)])
        file2 = random.choice(self.files[(ID2, dom2)])
        return load_image(file1), load_image(file2)


class SimpleTripletGenerator(Sequence):
    """
    NOT DEBUGGED YET :"(
    Using with tfa.losses.TripletSemiHardLoss
    i.e. there's no distinction between photo and sketch images.
    """
    def __init__(self, root_dirs, anno_path, batch_size=100):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        
        all_files = []
        for root_dir in root_dirs:
            all_files.extend([
                os.path.join(root, file).replace('\\', '/')
                    for root, dirs, files in os.walk(root_dir)
                        for file in files
                            if is_image_file(file)
            ])
            
        all_file2ID = dict()
        with open(anno_path, 'r') as f:
            for line in f.readlines():
                file, ID = line.split()
                ID = int(ID)
                all_file2ID[file] = ID
                    
        self.ID = dict() # file -> ID
        self.files_grouped = dict() # ID -> list of files
        for file in all_files:
            ID = all_file2ID[file]
            self.ID[file] = ID
            if ID not in self.files_grouped:
                self.files_grouped[ID] = [file]
            else:
                self.files_grouped[ID].append(file)
        self.batch_size = batch_size
        self._shuffle()


    def _shuffle(self):
        IDs = list(self.files_grouped.keys())
        random.shuffle(IDs)
        self.files = []
        for ID in IDs:
            files = self.files_grouped[ID]
            random.shuffle(files)
            self.files.extend(files)
        


    def __len__(self):
        return len(self.files) // self.batch_size


    def __getitem__(self, index):
        l = index * self.batch_size
        r = l + self.batch_size
        x, y = [], []
        for file in self.files[l:r]:
            x.append(load_image(file))
            y.append(self.ID[file])
        x = normalize_input(np.array(x))
        y = np.array(y)
        return x, y


    def on_epoch_end(self):
        self._shuffle()


class RandomNegativeGenerator(Sequence):
    """
    Loop through all anchor(sketch)-positive(photo) pairs.
    With each pair, randomly choose a negative(photo) image.
    """
    def __init__(self, batch_size, root_dirs, anno_path):
        self.batch_size = batch_size
        self.facenet = InceptionResNetV1(weights_path='pretrained_facenet/facenet_keras_weights.h5')
        self.facenet = get_l2_norm_model(self.facenet)
        
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        all_files = []
        for root_dir in root_dirs:
            all_files.extend([
                os.path.join(root, file).replace('\\', '/')
                    for root, dirs, files in os.walk(root_dir)
                        for file in files
                            if is_image_file(file)
            ])
            
        all_file2ID = dict()
        with open(anno_path, 'r') as f:
            for line in f.readlines():
                file, ID = line.split()
                ID = int(ID)
                all_file2ID[file] = ID
            
        self.files_grouped = dict() # (ID, domain) --> list of files
        for file in all_files:
            ID = all_file2ID[file]
            domain = 'photo' if '/photo/' in file else 'sketch'
            if (ID, domain) not in self.files_grouped:
                self.files_grouped[(ID, domain)] = [file]
            else:
                self.files_grouped[(ID, domain)].append(file)
                
        self.IDs = [key[0] for key in self.files_grouped]
        self.triplet_files = []
        self._generate_triplet_files()
        

    def _generate_triplet_files(self):
        self.triplet_files = []
        for ID in self.IDs:
            for fileA in self.files_grouped[(ID, 'sketch')]:
                for fileP in self.files_grouped[(ID, 'photo')]:
                    ID_N = random.choice(self.IDs)
                    while ID_N == ID:
                        ID_N = random.choice(self.IDs)
                    fileN = random.choice(self.files_grouped[(ID_N, 'photo')])
                    self.triplet_files.append((fileA, fileP, fileN))
    

    def __len__(self):
        return len(self.triplet_files) // self.batch_size
    
    
    def __getitem__(self, index):
        # x.shape = (batch_size, 160, 160, 3)
        # y.shape = (batch_size, 256)
        x, xP, xN, y = [], [], [], []
        L = index * self.batch_size
        R = L + self.batch_size
        for fileA, fileP, fileN in self.triplet_files[L:R]:
            x.append(load_image(fileA))
            xP.append(load_image(fileP))
            xN.append(load_image(fileN))
        x = normalize_input(np.array(x))
        xP = normalize_input(np.array(xP))
        yP = self.facenet.predict(xP)
        xN = normalize_input(np.array(xN))
        yN = self.facenet.predict(xN)
        
        y = K.concatenate([yP, yN], axis=1)
            
        return x, y
    
    def on_epoch_end(self):
        random.shuffle(self.IDs)
        self._generate_triplet_files()