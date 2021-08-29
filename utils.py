import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
plt.style.use('classic')

def get_rmse_error(true_energies, predicted_energies):
    df = pd.DataFrame(list(zip(true_energies, predicted_energies)), columns=['True', 'Model'])
    true_energies = df['True']
    predicted_energies = df['Model']
    mae = mean_absolute_error(true_energies, predicted_energies)
    rmse = sqrt(mean_squared_error(true_energies, predicted_energies))

    return mae, rmse*23.061

def plot_val_true(true_energies, predicted_energies, title='Validset', save_path=None, print_plot=True):
    df = pd.DataFrame(list(zip(true_energies, predicted_energies)), columns=['True', 'Model'])
    true_energies = df['True']
    predicted_energies = df['Model']
    mae = mean_absolute_error(true_energies, predicted_energies)
    print('Mean Absolute Error (MAE) = {0.real:.6f} eV'.format(mae))
    rmse = sqrt(mean_squared_error(true_energies, predicted_energies))
    print('Root Mean Squared Error (RMSE) = {0.real:.6f} eV'.format(rmse))
    print('Root Mean Squared Error (RMSE) = {0.real:.6f} kcal/mol'.format(rmse*23.061))

    plt.style.use('bmh')
    x = (true_energies - true_energies.min())*23.061
    y =  (predicted_energies - true_energies.min())*23.061
    xlabel='True energies (kcal/mol)'
    ylabel='Predicted energies (kcal/mol)'
    title = f'{title} - RMSE Error : {rmse*23.061:.6f} kcal/mol'
    x1 = np.array(x)
    y1 = np.array(y)

    # diagonal line
    xmin = min([np.min(x1), np.min(y1)])
    xmax = max([np.max(x1), np.max(y1)])

    xwidth = xmax - xmin
    xmargin = 0.03*xwidth

    xstart = xmin - xmargin
    xend = xmax + xmargin

    x0 = np.linspace(xstart, xend, 2)
    y0 = x0
    
    if print_plot is True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x0, y0, color='C1')
        ax.scatter(x1, y1, color='C2', alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12)
        # ax.axis([xstart, xend, xstart, xend])
        ax.set_aspect(1.0)
        if save_path is not None:
            plt.savefig(save_path, dpi=200)
        plt.show()

    return rmse*23.061

def get_hh_mm_left_time(left_time: int) -> str:
    minutes, _ = divmod(left_time, 60)
    hours, minutes = divmod(minutes, 60)
    str_left_time = f'{hours:02}h {minutes:02}m'

    return str_left_time

# One-hot-encode atom types(H,C,N,O,S)
def one_hot_encoding(atom_types: list):
    atom_type_one_hot = []
    for atom in atom_types:
        if atom == 'H':
            lst = [1, 0, 0, 0, 0]
        elif atom == 'C':
            lst = [0, 1, 0, 0, 0]
        elif atom == 'N':
            lst = [0, 0, 1, 0, 0]
        elif atom == 'O':
            lst = [0, 0, 0, 1, 0]
        elif atom == 'S':
            lst = [0, 0, 0, 0, 1]
        else:
            raise Exception('Unknown Atom!')
        atom_type_one_hot.append(lst)
            
    return np.array(atom_type_one_hot)

def min_max_normalize(array: np.array, min_value, max_value):
    normalized = (array-min_value) / (max_value - min_value)
    
    return normalized

def min_max_unnormalize(array: np.array, min_value, max_value):
    unnormalized = array * (max_value - min_value) + min_value
    
    return unnormalized

def standardization(array: np.array, mean_value, std_value):
    std_array = (array-mean_value) / std_value
    
    return std_array

def unstandardization(array: np.array, mean_value, std_value):
    unstd_array = array * std_value + mean_value
    
    return unstd_array


def check_max_atom_nums():  # max : 51, min : 17, mean : 35.39
    import os
    from os.path import join
    from glob import glob
    from config import dset_config
    from numpy import mean
    dipeps = os.listdir(dset_config.xyzs_path)
    compare_dipeps = dipeps
    dset_path = {seq: dict() for seq in compare_dipeps if len(seq)==2}

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

    atom_nums = []
    for i, (seq, value) in enumerate(dset_path.items(), 1):
        print(f'---------{i:04}---------')
        with open(value['xyz'], 'r') as f:
            txt_lines = f.readlines()
            txt = ''.join(txt_lines)
        atom_num = int(txt_lines[0])
        atom_nums.append(atom_num)
    print(max(atom_nums), min(atom_nums), round(mean(atom_nums)))

def save_y0_from_xyzs():  # max : 51, min : 17, mean : 35.39
    import os
    import pickle
    from os.path import join
    from glob import glob
    from config import dset_config
    from config import CHEMICAL_SYMBOL_TO_ENERGY

    dipeps = os.listdir(dset_config.xyzs_path)
    compare_dipeps = dipeps
    dset_path = {seq: dict() for seq in compare_dipeps if len(seq)==2}

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

    y0_list = dict()
    for i, (seq, value) in enumerate(dset_path.items(), 1):
        print(f'\rGetting y0 values... {i:03}/{len(dset_path)} ', end='')
        with open(value['xyz'], 'r') as f:
            txt_lines = f.readlines()
            txt = ''.join(txt_lines)
        atom_num = int(txt_lines[0])

        chemical_symbols_list = []
        for lnum in range(2, 2+atom_num):
            chemical_symbols_list.append(txt_lines[lnum][0])
        
        y0 = sum([CHEMICAL_SYMBOL_TO_ENERGY[symbol] for symbol in chemical_symbols_list])
        y0_list[seq] = y0
    with open(dset_config.y0_list_path, 'wb') as f:
        pickle.dump(y0_list, f)
    print('Done.')

    return y0_list

# if __name__ == '__main__':
#     import pickle
#     with open('y0_list.pk', 'rb') as f:
#         y0_list = pickle.load(f)
#     print(y0_list)