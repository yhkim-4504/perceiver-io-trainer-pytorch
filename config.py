from easydict import EasyDict
"""
The dataset should be sorted as below.

{xyz_path Folder}
    ---- {AA Folder}
        ---- {AA-vdwdf2.xyz}
    ---- {AC Folder}
        ---- {AC-vdwdf2.xyz}
    ...

{descriptor_path Folder}
    ---- {AA.npy}
    ---- {AC.npy}
    ...

"""

# loss list
loss_config = EasyDict(
    {
        'MSELoss': {'name': 'MSELoss', 'params': {}},
        'L1Loss' : {'name': 'L1Loss', 'params': {}}
    }
)

# optimizer list
optimizer_config = EasyDict(
    {
        'optims': 
            {
                'Adam': {'name': 'Adam', 'params': {}},
                'AdamW': {'name': 'AdamW', 'params': {}},
                'SGD': {'name': 'SGD', 'params': {}},
                'RMSprop': {'name': 'RMSprop', 'params': {}}
            },
        'scheduler':
            {
                'None': {'name': None, 'params': {}},
                'StepLR': {'name': 'StepLR', 'params': {'step_size': 5000, 'gamma': 0.5}},
                'MultiStepLR': {'name': 'MultiStepLR', 'params': {'milestones': [5000, 10000, 15000, 20000], 'gamma': 0.5}},
                'CosineAnnealingLR': {'name': 'CosineAnnealingLR', 'params': {'T_max': 1000, 'eta_min': 0}},
            }
    }
)

# dataset option config
input_preprocessing_type = {'min_max_normalize': 'min_max_normalize', 'standardization': 'standardization'}  # dataset preprocessing type
output_preprocessing_type = {'y0_subtract': 'y0_subtract', 'mean_subtract': 'mean_subtract'}  # total energy preproceesing type
dset_config = EasyDict(
    {
        'saved_dset_path': None,  # if saved_dset_path is not None, load dataset from saved_dset_path.
        'xyzs_path': r'C:\Users\ky450\Documents\PythonScripts\github\gap\data\dipeptides',  # xyzs path
        'descriptors_path': r'dataset',  # descriptors path
        'y0_list_path': r'y0_list.pk',  # y0_list path, if doesn't exist, get y0 values and save at the path.
        'whether_to_save_dset': False,  # dataset will be saved at {train_config.model_save_name} folder.
        'use_all_dipeps': False,  # if use, all dipeptide sequences in xyzs_path will be used.
        'descriptor_dim': 80,  # 75(descriptor)+5(atom_type)
        'max_atom_num': 51,  # max atom number of all sequences from utils.check_max_atom_nums method.
        'use_dipeps': ['EN'],  # when use_all_dipeps is False -> only this sequcenes will be used.
        'max_dset_ratio': 0.4,  # used dataset ratio just to use small dataset.
        'train_val_split': 0.5,  # train - valid split ratio
        'val_test_split': 0.7,  # valid - test split ratio
        'input_preprocessing': input_preprocessing_type['standardization'],  # x(descriptor) preprocess
        'output_preprocessing': output_preprocessing_type['mean_subtract']  # y(total_energy) preprocess : subtract y0 or mean(y) of each sequence
    }
)

# train option config
train_config = EasyDict(
    {
        'epochs': 30000,
        'batch_size': 512,
        'epoch_print_range': 200,
        'print_validation': True,
        'initial_lr': 0.0006,
        'loss': loss_config.MSELoss,
        'optimizer': optimizer_config.optims.AdamW,
        'scheduler': optimizer_config.scheduler.MultiStepLR,
        'random_seed': 777,
        'torch_dtype': 'float32',
        'min_chkpoint_epoch': 5000,
        'model_save_name': 'base1',  # create a folder with this name and save infomation and model checkpoint.
        'whether_to_save': True  # whether to save model checkpoint.
    }
)

# model option config
model_config = EasyDict(
    {
            'dim': dset_config.descriptor_dim,         # dimension of sequence to be encoded
            'queries_dim': dset_config.descriptor_dim, # dimension of decoder queries
            'logits_dim': 1,            # dimension of final logits
            'depth': 3,                   # depth of net
            'num_latents': 8,           # number of latents, or induced set points, or centroids. different papers giving it different names
            'latent_dim': 8,            # latent dimension
            'cross_heads': 1,             # number of heads for cross attention. paper said 1
            'latent_heads': 4,            # number of heads for latent self attention, 8
            'cross_dim_head': 32,         # number of dimensions per cross attention head
            'latent_dim_head': 32,        # number of dimensions per latent self attention head
            'weight_tie_layers': False,   # whether to weight tie layers (optional, as indicated in the diagram)
            'self_per_cross_attn': 2     # number of self attention blocks per cross attention
    }
)

CHEMICAL_SYMBOL_TO_ENERGY = {
            'H': -16.705003113816887,
            'C': -258.9727666548056,
            'N': -393.6962697601746,
            'O': -578.760929138896,
            'S': -878.760929138896,
            'Ni': -5836.7755,
            'Fe': -5000
            }

if __name__ == '__main__':
    print(dset_config)
    print(train_config)
    print(model_config)