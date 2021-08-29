import torch
import torch.optim as optim
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import os
from os.path import join
from config import model_config, train_config, dset_config
from utils import plot_val_true
from dataset import DescriptorDatasetLoader
from perceiver_io import PerceiverIO
from torch.utils.data import DataLoader

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Training with the following device :", device)

RANDOM_SEED = train_config.random_seed
if RANDOM_SEED is not None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

if train_config.torch_dtype == 'float32':
    torch.set_default_dtype(torch.float32)
elif train_config.torch_dtype == 'float64':
    torch.set_default_dtype(torch.float64)
else:
    raise Exception(f'Error! {train_config.torch_dtype} data type does not exist.')

class Trainer:
    def __init__(self):
        # Create workspace folder
        self.__model_save_name = train_config.model_save_name
        print(f'Creating folder... Path : {self.__model_save_name}')
        os.mkdir(self.__model_save_name)

        # Dataset Load
        dset_loader = DescriptorDatasetLoader()
        self.trainset, self.validset, self.testset, self.input_preprocess_values_list, self.y0_list = dset_loader.load_all_seq_dataset()

        # Set configs
        self.queries_dim = model_config.queries_dim
        self.model, self.loss, self.optimizer, self.scheduler, self.losses = [None] * 5
        self.epochs = train_config.epochs
        self.epoch_print_range = train_config.epoch_print_range
        self.batch_size = train_config.batch_size
        self.whether_to_save = train_config.whether_to_save
        self.min_chkpoint_epoch = train_config.min_chkpoint_epoch
        self.print_validation = train_config.print_validation

    @property
    def model_save_name(self):
        return self.__model_save_name

    def __str__(self):
        scheduler_state_dict = self.scheduler.state_dict() if self.scheduler is not None else None
        str_y0_list = ''
        for i, (seq, value) in enumerate(self.y0_list.items(), 1):
            str_y0_list += f'{seq}: {value:.6f}' + ' '*3
            if i%5 == 0:
                str_y0_list += '\n'
        str_input_preprocess_values_list = ''
        for seq, value in self.input_preprocess_values_list.items():
            str_input_preprocess_values_list += f'{seq}: {value}' + '\n'

        txt = f'''
        [Model]
dim = {model_config.dim}
queries_dim = {model_config.queries_dim}
logits_dim = {model_config.logits_dim}
depth = {model_config.depth}
num_latents = {model_config.num_latents}
latent_dim = {model_config.latent_dim}
cross_heads = {model_config.cross_heads}
latent_heads = {model_config.latent_heads}
cross_dim_head = {model_config.cross_dim_head}
latent_dim_head = {model_config.latent_dim_head}
weight_tie_layers = {model_config.weight_tie_layers}
self_per_cross_attn = {model_config.self_per_cross_attn}
Count of model parameters : {self.param_count:,}

        [Train]
loss_type : {self.loss}
optimizer : {self.optimizer}
scheduler : {self.scheduler}
scheduler values :
{scheduler_state_dict}
epochs : {self.epochs}
epoch_print_range : {self.epoch_print_range}
batch_size : {self.batch_size}
torch_dtype : {train_config.torch_dtype}
model_save_name : {self.model_save_name},
whether_to_save : {self.whether_to_save},
min_chkpoint_epoch : {self.min_chkpoint_epoch}

        [Dataset]
trainset : {self.trainset}
validset : {self.validset}
testset : {self.testset}
max_dset_ratio : {dset_config.max_dset_ratio}
train_val_split : {dset_config.train_val_split}
val_test_split : {dset_config.val_test_split}
input_preprocess_values_list :
{str_input_preprocess_values_list}
y0_list :
{str_y0_list}
        '''

        return txt

    def train(self, print_plot=True):
        epochs = self.epochs
        last_lr = self.optimizer.param_groups[0]['lr']
        min_val_loss = 100
        min_rmse_error = 100
        mean_train_loss = 100
        start_time = time.time()
        train_losses = list()
        self.losses = {'epoch': list(), 'train': list(), 'valid': list(), 'valid_rmse': list()}

        # Save Trainer Log
        with open(join(self.__model_save_name, f'trainer_info_{self.__model_save_name}.txt'), 'w') as f:
            f.write(str(self))

        for epoch in range(epochs+1):
            self.model.train()

            # Train Batch
            for batch, (x, y, seqs) in enumerate(self.train_loader):
                print(f'\rEpoch: {epoch:04}/{epochs}, batch: {batch+1:03}/{len(self.train_loader)}, Train_loss: {mean_train_loss:.6f}', end='')
                x, y = x.to(device), y.to(device)
                mean_train_loss = self.forward_and_update(x, y, train_losses)

            # Scheduler Step
            if self.scheduler is not None:
                self.scheduler.step()
                last_lr = self.scheduler.get_last_lr()[0]
                
            # Print Epoch info                
            if epoch % self.epoch_print_range == 0:
                mean_train_loss = sum(train_losses) / len(train_losses)
                train_losses.clear()
                self.losses['epoch'].append(epoch)
                self.losses['train'].append(mean_train_loss)

                # Calculate validation Error
                if self.print_validation is True:
                    mean_val_loss, true_energies, model_energies = self.process_validation(self.valid_loader)
                    self.losses['valid'].append(mean_val_loss)
                    if mean_val_loss < min_val_loss:
                        min_val_loss = mean_val_loss
                use_time = round(time.time() - start_time)
                info = 'Epoch: {:4d}/{} Cost: {:.6f} Validate Cost: {:.6f}, lr: {:f}, use_time: {:5d}, min_rmse: {:.6f}'.format(
                    epoch, epochs, mean_train_loss, mean_val_loss, last_lr, use_time, min_rmse_error)
                print(f'\r{info}')

                # Calculate RMSE Error & Plot
                if self.print_validation is True:
                    mean_rmse_error = plot_val_true(true_energies, model_energies, print_plot=print_plot)
                    self.losses['valid_rmse'].append(mean_rmse_error)
                    self.plot_losses()

                    if mean_rmse_error < min_rmse_error:
                        min_rmse_error = mean_rmse_error
                        # Save model
                        if (self.whether_to_save is True) and (epoch >= self.min_chkpoint_epoch):
                            print(f'---------- New checkpoint updated! min_rmse : {min_rmse_error:.6f} kcal/mol')
                            self.save_checkpoint('chk_point', info, epoch, true_energies, model_energies, print_plot=print_plot)
                        
        # Training Complete & Model Save
        print('---------- Training Done!')
        if self.whether_to_save is True:
            self.save_checkpoint('train_done', info, epoch, true_energies, model_energies, print_plot=print_plot)
    
    def save_checkpoint(self, type: str, info: str, epoch: int, true_energies, model_energies, print_plot: bool):
        valid_rmse, test_rmse = -1, -1
        if print_plot is True:
            valid_plot_fname = join(self.model_save_name, f'{self.model_save_name}_{type}_{epoch}_valid_plot.png')
            test_plot_fname = join(self.model_save_name, f'{self.model_save_name}_{type}_{epoch}_test_plot.png')
            loss_plot_fname = join(self.model_save_name, f'{self.model_save_name}_{type}_{epoch}_loss_plot.png')
            print(f'[{type} Validset Plot]')
            valid_rmse = plot_val_true(true_energies, model_energies, print_plot=True, save_path=valid_plot_fname)
            print(f'[{type} Testset Plot]')
            test_rmse = self.plot_dataset(test_plot_fname, self.test_loader)
            print(f'[{type} Train & Valid Loss Plot]')
            self.plot_losses(save_path=loss_plot_fname)

        self.save_model(type)
        self.save_log(type, info, epoch, valid_rmse, test_rmse)

    def load_dataloader(self):
        self.train_loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.valid_loader = DataLoader(self.validset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def load_model(self, state_dict_path=None):
        model = PerceiverIO(
            dim = model_config.dim,             
            queries_dim = model_config.queries_dim,          
            logits_dim = model_config.logits_dim,       
            depth = model_config.depth,               
            num_latents = model_config.num_latents,           
            latent_dim = model_config.latent_dim,          
            cross_heads = model_config.cross_heads,         
            latent_heads = model_config.latent_heads,        
            cross_dim_head = model_config.cross_dim_head,      
            latent_dim_head = model_config.latent_dim_head,     
            weight_tie_layers = model_config.weight_tie_layers,
            self_per_cross_attn = model_config.self_per_cross_attn
        )
        # Prameter Count
        self.param_count = 0
        for param in model.parameters():
            for tensor in param:
                self.param_count += tensor.shape.numel()
        print(f'Count of model parameters : {self.param_count:,}')

        if state_dict_path is not None:
            model.load_state_dict(torch.load(state_dict_path))
            print(f'Pretrained model loaded from {state_dict_path}')
            
        self.model = model.to(device)
        print('Model Loading Done.')

    def load_loss_layer(self):
        loss_name = train_config.loss.name
        if loss_name == 'MSELoss':
            self.loss = torch.nn.MSELoss().to(device)
        elif loss_name == 'L1Loss':
            self.loss = torch.nn.L1Loss().to(device)
        else:
            raise Exception(f'Error! {loss_name} layer does not exist.')

    def load_optimizer_scheduler(self):
        # Load Optimizer
        optim_name = train_config.optimizer.name
        initial_lr = train_config.initial_lr

        if optim_name == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=initial_lr)
        elif optim_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=initial_lr)
        elif optim_name == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=initial_lr)
        elif optim_name == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=initial_lr)
        else:
            raise Exception(f'Error! {optim_name} optimizer does not exist.')

        # Load Scheduler
        scheduler_name = train_config.scheduler.name
        params = train_config.scheduler.params
        if not scheduler_name:
            self.scheduler = None
        elif scheduler_name == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, params.step_size, params.gamma)
        elif scheduler_name == 'MultiStepLR':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, params.milestones, params.gamma)
        elif scheduler_name == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, params.T_max, params.eta_min)
        else:
            raise Exception(f'Error! {scheduler_name} scheduler does not exist.')

    def forward_and_update(self, x, y, train_losses):
        predicted = self.forward_model(x).unsqueeze(-1)
        cost = self.loss(predicted, y)
        train_losses.append(cost.item())
        mean_train_loss = sum(train_losses) / len(train_losses)
        
        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        return mean_train_loss

    def forward_model(self, batch_x):  # forward
        queries = torch.ones(len(batch_x), 1, self.queries_dim).to(device)
        predicted = self.model(batch_x, queries=queries).squeeze()

        return predicted

    def predict_total_energy(self, batch_x, seqs):  # predict total_energy with y0
        with torch.no_grad():
            if len(batch_x.shape) == 2:
                batch_x = batch_x.unsqueeze(0)
                seqs = [seqs]
            queries = torch.ones(len(batch_x), 1, self.queries_dim).to(device)
            predicted = self.model(batch_x, queries=queries).squeeze()
            predicted += torch.Tensor([self.y0_list[seq] for seq in seqs]).squeeze()

        return predicted
        
    def process_validation(self, dataloader):
        # Get val loss
        with torch.no_grad():
            self.model.eval()
            val_losses = list()
            true_energies = list()
            model_energies = list()
            
            for batch, (x, y, seqs) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)

                out = self.forward_model(x).unsqueeze(-1)
                val_loss = self.loss(out, y)
                val_losses.append(val_loss.item())
                y, out = y.squeeze().tolist(), out.squeeze().tolist()
                y = y if type(y) is list else [y]
                out = out if type(out) is list else [out]
                true_energies.extend(y)
                model_energies.extend(out)

            mean_val_loss = sum(val_losses) / len(val_losses)

        return mean_val_loss, true_energies, model_energies

    def save_model(self, type: str):
        save_model_name = join(self.model_save_name, f'{self.model_save_name}_{type}.torch')
        torch.save(self.model.state_dict(), save_model_name)
        print(f'Model has been saved at {save_model_name}')

    def save_log(self, type: str, info: str, epoch: int, valid_rmse: float, test_rmse: float):
        log_fname = join(self.model_save_name, f'{self.model_save_name}_{type}_{epoch}_info.txt')

        losses = self.losses
        str_losses = ''
        if len(losses['valid']) == len(losses['epoch']):
            for i in range(len(self.losses['epoch'])):
                str_losses += "epoch: {:6}, train_loss: {:.9f}, valid_loss: {:.9f}, valid_rmse: {:.9f}\n"\
                    .format(losses['epoch'][i], losses['train'][i], losses['valid'][i], losses['valid_rmse'][i])
        else:
            for i in range(len(self.losses['epoch'])):
                str_losses += "epoch: {:6}, train_loss: {:.9f}\n"\
                    .format(losses['epoch'][i], losses['train'][i])
                    
        with open(log_fname, 'w') as f:
            f.write(f'validset_rmse_error : {valid_rmse:.6f}, testset_rmse_error : {test_rmse:.6f}\n')
            f.write(f'{info}\n')
            f.write('[Loss Info]\n')
            f.write(str_losses)
            f.write('[Trainer Info]\n')
            f.write(str(self))
        print(f'Log has been saved at {log_fname}')

    def plot_dataset(self, save_path=None, dataloader=None):  # deafult : testset
        if not dataloader:
            print('Testset Plot')
            dataloader = self.test_loader
        _, true_energies, model_energies = self.process_validation(dataloader)
        rmse_error = plot_val_true(true_energies, model_energies, save_path=save_path)

        return rmse_error

    def plot_losses(self, save_path=None):
        losses = self.losses
        if len(losses['epoch']) > 1:
            plt.figure(figsize=(15, 5))
            plt.title('Train & Valid Loss Plot for Epochs')
            plt.plot(losses['epoch'], np.log10(losses['train']), label='train_loss')
            if len(losses['valid']) == len(losses['epoch']):
                plt.plot(losses['epoch'], np.log10(losses['valid']), label='valid_loss')
                plt.plot(losses['epoch'], np.log10(losses['valid_rmse']), label='valid_rmse_loss')
            plt.legend(loc='upper right')
            plt.xticks([x for x in range(0, losses['epoch'][-1]+1, int(losses['epoch'][-1]/20))])
            plt.xlabel('epoch')
            plt.ylabel('loss (log10)')
            if save_path is not None:
                plt.savefig(save_path, dpi=200)
            plt.show()

if __name__ == '__main__':
    # Load
    trainer = Trainer()
    trainer.load_dataloader()
    trainer.load_model(state_dict_path=None)
    trainer.load_loss_layer()
    trainer.load_optimizer_scheduler()
    print('\n[Trainer Info]')
    print(trainer)