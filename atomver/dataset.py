import os
import pickle
import numpy as np
import re
import torch
import random
from os.path import join
from glob import glob
from utils import save_y0_from_xyzs, one_hot_encoding, min_max_normalize, standardization
from config import dset_config, train_config
from torch.utils.data import Dataset


class DescriptorDataset(Dataset):
    def __init__(self, x: np.array, y: np.array, seqs: list):
        self.x = x
        self.y = y
        self.seqs = seqs

    def __str__(self):
        return f'x_shape : {self.x.shape}, y_shape : {self.y.shape}, seq_len : {len(self.seqs)}'

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.Tensor(self.x[idx]), torch.Tensor([self.y[idx]]), self.seqs[idx]

class DescriptorDatasetLoader:
    def __init__(self):
        self.__saved_dset_path = dset_config.saved_dset_path
        self.whether_to_save_dset = dset_config.whether_to_save_dset
        self.output_preprocessing = dset_config.output_preprocessing

        if self.__saved_dset_path is None:
            # Get Sequences
            dipeps = sorted(os.listdir(dset_config.xyzs_path))
            compare_dipeps = dipeps if dset_config.use_all_dipeps else dset_config.use_dipeps
            dset_path = {seq: dict() for seq in compare_dipeps}
            
            if self.output_preprocessing == 'y0_subtract':
                # Get y0 values
                if os.path.exists(dset_config.y0_list_path):
                    with open(dset_config.y0_list_path, 'rb') as f:
                        self.y0_list = pickle.load(f)
                    print('Load y0_list... Done.')
                else:
                    self.y0_list = save_y0_from_xyzs()
            else:
                self.y0_list = dict()

            # Get xyzs path
            for dipep in dipeps:
                xyzs_path = glob(join(dset_config.xyzs_path, dipep, '*.xyz'))
                for p in xyzs_path:
                    fname = os.path.basename(p)
                    if 'vdwdf2' in fname:
                        for seq in compare_dipeps:
                            if seq in fname:
                                dset_path[seq]['xyz'] = p
                                break

            # Get descriptors path
            descriptors_path = glob(join(dset_config.descriptors_path, '*.npy'))
            for p in descriptors_path:
                fname = os.path.basename(p)
                for seq in compare_dipeps:
                    if seq in compare_dipeps:
                        if seq in fname:
                            dset_path[seq]['descriptor'] = p
                            break

            # Check Error
            if len(dset_path) == 0:
                raise Exception('Error! Not a single sequence path exists.')
            elif len(dset_path) != len(compare_dipeps):
                raise Exception('Error! The number of data does not match.')
            else:
                for seq, value in dset_path.items():
                    if 'xyz' not in value.keys():
                        raise Exception(f'Error! {seq} sequence xyz path does not exist.')
                    elif 'descriptor' not in value.keys():
                        raise Exception(f'Error! {seq} sequence descriptor path does not exist.')
            
            # Print found sequences
            print(f'Found {len(dset_path)} sequences : {compare_dipeps}')
            self.dset_path = dset_path
        else:
            print(f'Found dataset path : {self.__saved_dset_path}')

    @property
    def saved_dset_path(self):
        return self.__saved_dset_path

    def load_seq_dataset(self, seq: str):
        print(f'Loading Dataset... Sequence : {seq}')
        xyz_path, descriptor_path = self.dset_path[seq]['xyz'], self.dset_path[seq]['descriptor']
        
        # Read xyz texts
        with open(xyz_path, 'r') as f:
            txt_lines = f.readlines()
            txt = ''.join(txt_lines)
        atom_num = int(txt_lines[0])
        
        # Load descriptor dataset
        x = np.load(descriptor_path)
        x = np.real(x)
        num_dipeps = int(len(x) / atom_num)

        # Random index
        rand_idx = [idx for idx in range(num_dipeps)]
        rand_idx = random.sample(rand_idx, len(rand_idx))
        x = x.reshape(num_dipeps, atom_num, x.shape[-1])[rand_idx]

        # Set train, valid, test nums
        use_num_dipeptides = round(num_dipeps * dset_config.max_dset_ratio)
        train_num = round(use_num_dipeptides * dset_config.train_val_split)
        valid_num = round((use_num_dipeptides - train_num) * dset_config.val_test_split)
        test_num = int(use_num_dipeptides - train_num - valid_num)

        if train_num == 0 or valid_num == 0 or test_num == 0:
            print(f"""\
            total_atom_num : {len(x)}, atom_num : {atom_num}
            train_num : {train_num}, valid_num : {valid_num}, test_num : {test_num}
            {seq} sequence has been excluded!
            """)
            return (None, None), (None, None), (None, None), None

        # Preprocess inputs
        input_preprocess_values = dict()
        input_preprocess_values['atom_num'] = atom_num
        if dset_config.input_preprocessing == 'min_max_normalize':
            print(f'Input Preprocessing : {dset_config.input_preprocessing}')
            input_preprocess_values['type'] = dset_config.input_preprocessing
            x_train_min = np.min(x[:train_num])
            input_preprocess_values['min'] = x_train_min
            x_train_max = np.max(x[:train_num])
            input_preprocess_values['max'] = x_train_max
            x = min_max_normalize(x, x_train_min, x_train_max)
        elif dset_config.input_preprocessing == 'standardization':
            print(f'Input Preprocessing : {dset_config.input_preprocessing}')
            input_preprocess_values['type'] = dset_config.input_preprocessing
            x_train_mean = np.mean(x[:train_num])
            input_preprocess_values['mean'] = x_train_mean
            x_train_std = np.std(x[:train_num])
            input_preprocess_values['std'] = x_train_std
            x = standardization(x, x_train_mean, x_train_std)
        elif dset_config.input_preprocessing == 'custom_std':
            input_preprocess_values['type'] = dset_config.input_preprocessing
            x_train_mean = 0.00015165320577508674
            input_preprocess_values['mean'] = x_train_mean
            x_train_std = 9.147343315840497e-06
            input_preprocess_values['std'] = x_train_std
            x = standardization(x, x_train_mean, x_train_std)
        else:
            print(f'Input Preprocessing : None')
            input_preprocess_values['type'] = 'None'

        # Get atom types
        i = 0
        atom_type = []
        while True:
            atom_nums = int(txt_lines[i])
            for lnum in range(i+2, i+2+atom_nums):
                atom_type.append(txt_lines[lnum][0])
            i += 2 + atom_nums
            
            if i >= len(txt_lines):
                break
        # One-hot-encode
        atom_type_one_hot = one_hot_encoding(atom_type)
        atom_type_one_hot = atom_type_one_hot.reshape(num_dipeps, atom_num, atom_type_one_hot.shape[-1])[rand_idx]
        # Error check
        if len(atom_type_one_hot) != len(x):
            raise Exception(f'atom_type length error! : {len(atom_type_one_hot)} != {len(x)}')

        # Stack atom_type(n, 5) & x(n, 75) -> (n, 80) shape
        x = np.concatenate([atom_type_one_hot, x], axis=-1)

        # Concatenate extra atom num vectors
        extra_atom_num = dset_config.max_atom_num - atom_num
        if extra_atom_num >= 1:
            x = np.concatenate([x, np.zeros([num_dipeps, extra_atom_num, x.shape[-1]])], axis=1)
        input_preprocess_values['data_shape'] = x.shape

        # Get atom energy(y_label)
        y = []
        z_iter = re.finditer(r'\s+energy=(?P<energy>[+-]\d*\.?\d*)\s+', txt)
        for z in z_iter:
            y.append(float(z.group('energy')))
        # Error check
        if len(x) != len(y):
            raise Exception(f'energy length error! : {len(x)} != {len(y)}')
        y = np.array(y, dtype=np.float64)[rand_idx]

        # output preprocessing
        if self.output_preprocessing == 'y0_subtract':
            y0 = self.y0_list[seq]
        elif self.output_preprocessing == 'mean_subtract':
            y0 = int(np.mean(y))
            self.y0_list[seq] = y0
        else:
            self.y0_list[seq] = 0
        y = y - y0

        x_train, y_train = x[:train_num], y[:train_num]
        x_valid, y_valid = x[train_num:train_num+valid_num], y[train_num:train_num+valid_num]
        x_test, y_test = x[train_num+valid_num:train_num+valid_num+test_num], y[train_num+valid_num:train_num+valid_num+test_num]

        print(f"""\
            total_atom_num : {len(x)*atom_num}, total_dipeptides_num : {len(y)}, atom_num : {atom_num}
            train_num : {train_num}, valid_num : {valid_num}, test_num : {test_num},
            x_train_shape : {x_train.shape}, x_valid_shape : {x_valid.shape}, x_test_shape : {x_test.shape}
            y_train_len : {len(y_train)}, y_valid_len = {len(y_valid)}, y_test_len = {len(y_test)}
            """)

        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), input_preprocess_values

    def load_all_seq_dataset(self):
        if self.saved_dset_path is not None:
            return self.load_dataset_from_path()

        input_preprocess_values_list = dict()
        seqs = {'train': list(), 'valid': list(), 'test': list()}
        is_first = True

        for i, seq in enumerate(self.dset_path.keys(), 1):
            print(f'--------- {i:03}/{len(self.dset_path.keys())}')
            (x_train_seq, y_train_seq), (x_valid_seq, y_valid_seq), (x_test_seq, y_test_seq), input_preprocess_values = self.load_seq_dataset(seq)
            if x_train_seq is None:
                continue
            input_preprocess_values_list[seq] = input_preprocess_values

            if is_first:
                x_train, y_train = x_train_seq, y_train_seq
                x_valid, y_valid = x_valid_seq, y_valid_seq
                x_test, y_test = x_test_seq, y_test_seq
                is_first = False
            else:
                x_train, y_train = np.vstack([x_train, x_train_seq]), np.hstack([y_train, y_train_seq])
                x_valid, y_valid = np.vstack([x_valid, x_valid_seq]), np.hstack([y_valid, y_valid_seq])
                x_test, y_test = np.vstack([x_test, x_test_seq]), np.hstack([y_test, y_test_seq])
            
            seqs['train'].extend([seq for _ in range(len(y_train_seq))])
            seqs['valid'].extend([seq for _ in range(len(y_valid_seq))])
            seqs['test'].extend([seq for _ in range(len(y_test_seq))])

        print(f"""All Datasets Loaded... Total Dataset : 
            x_train_shape : {x_train.shape}, x_valid_shape : {x_valid.shape}, x_test_shape : {x_test.shape}
            y_train_len : {len(y_train)}, y_valid_len = {len(y_valid)}, y_test_len = {len(y_test)}
            input_preprocess_values_list : {input_preprocess_values_list}
            """)

        trainset = DescriptorDataset(x_train, y_train, seqs['train'])
        validset = DescriptorDataset(x_valid, y_valid, seqs['valid'])
        testset = DescriptorDataset(x_test, y_test, seqs['test'])

        if self.whether_to_save_dset is True:
            self.save_all_seq_dataset(trainset, validset, testset, input_preprocess_values_list, self.y0_list)
        
        return trainset, validset, testset, input_preprocess_values_list, self.y0_list

    def save_all_seq_dataset(self, trainset, validset, testset, input_preprocess_values_list, y0_list):
        dset = dict()
        dset['train'] = trainset
        dset['valid'] = validset
        dset['test'] = testset
        dset['input_preprocess_values_list'] = input_preprocess_values_list
        dset['y0_list'] = y0_list
        
        fpath = join(train_config.model_save_name, 'saved_dataset.pk')
        with open(fpath, 'wb') as f:
            pickle.dump(dset, f)
        print(f'Dataset has been saved at {fpath}')

    def load_dataset_from_path(self):
        with open(self.saved_dset_path, 'rb') as f:
            dset = pickle.load(f)
        print(f'Dataset has been loaded from {self.saved_dset_path}')

        return dset['train'], dset['valid'], dset['test'], dset['input_preprocess_values_list'], dset['y0_list']


if __name__ == '__main__':
    dset = DescriptorDatasetLoader()
    trainset, validset, testset, input_preprocess_values_list, y0_list = dset.load_all_seq_dataset()