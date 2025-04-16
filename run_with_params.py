import os
import subprocess
import datetime
import time
import json
import shutil


def run(experiment_path_pre, settings, commands):
    with open(f'{experiment_path_pre}/_status.tmp', 'w') as f:
        now = datetime.datetime.now()
        f.write(now.strftime(f"Experiment: %d/%m/%Y %H:%M:%S\n"))
        f.write('exp,' + ','.join(f'{key}' for key, value in settings[0].items()) + f',timing\n')

    for i, (setting, command) in enumerate(zip(settings, commands)):
        print(f'--- Beginning experiment {i} ---')
        start_time = time.process_time()
        subprocess.call(command, shell=True) # The experiment will save results on its own
        end_time = time.process_time()
        minutes = (end_time - start_time) / 60

        with open(f'{experiment_path_pre}/_status.tmp', 'a+') as f:
            f.write(f'{i},' + ','.join(f'{value}' for key, value in setting.items()) + f',{minutes}\n')

def train_with_params(mode, settings, experiment_number, selection_method=None, submode=None, overwrite=False):

    commands = []
    experiment_path = ''
    experiment_path_pre = f'experiments/{experiment_number}'
    shutil.copy('models.py', experiment_path_pre)

    for setting in settings:
        experiment_path_base = f'{setting["latent_dim_1"]}-{setting["L_1"]}'
        if mode == 'D':
            base_Lambda_dir = 'Lambda_d={:.5f}'.format(setting['Lambda_d'])
        elif mode == 'P':
            base_Lambda_dir = 'Lambda_p={:.5f}'.format(setting['Lambda_p'])
        else:
            raise ValueError('Unknown mode')
        
        classi_Lambda_dir = 'Lambda_c={:.5f}'.format(setting['Lambda_c'])

        experiment_path = os.path.join(experiment_path_pre,
                                           experiment_path_base,
                                           base_Lambda_dir,
                                           classi_Lambda_dir)
        if setting['Lambda_d'] > 0:
                setting['load_mse_model_path'] = os.path.join(experiment_path_pre,
                                                             experiment_path_base,
                                                             'Lambda=0.00000')
        
        commands.append(
            'python train.py ' + ' '.join(f'--{key} {value}' for key, value in setting.items()) \
                + f' -experiment_path {experiment_path}' + f' -mode {mode}'
        )

    for command in commands:
        print(command)
    print('Number of commands to execute:', len(commands))

    with open(f'{experiment_path_pre}/_status.tmp', 'w') as f:
        now = datetime.datetime.now()
        f.write(now.strftime(f"Experiment: %d/%m/%Y %H:%M:%S\n"))
        f.write('exp,' + ','.join(f'{key}' for key, value in settings[0].items()) + f',timing\n')

        total_time = 0
    for i, (setting, command) in enumerate(zip(settings, commands)):
        print(f'--- Beginning experiment {i} ---')
        start_time = time.process_time()
        subprocess.call(command, shell=True) # The experiment will save results on its own
        end_time = time.process_time()
        minutes = (end_time - start_time) / 60
        total_time += minutes

        with open(f'{experiment_path_pre}/_status.tmp', 'a+') as f:
            f.write(f'{i},' + ','.join(f'{value}' for key, value in setting.items()) + f',{minutes}\n')

    with open(f'{experiment_path_pre}/_status.tmp', 'a+') as f:
        f.write(f'Total time taken (minutes): {total_time}')

    print(f'Finished running experiment {experiment_number}')




if __name__ == '__main__':

    # Base mode example FOR USUAL BASE MODEL
    settings = []
    experiment_number = 'M_d=0'
    mode = 'P'
    latent_dim_1, L_1 = 20, 8
    
    Lambda_distortion = [1] # for distortion weight
    Lambda_perception = [0, 0.00025, 0.00075, 0.00125, 0.002]
    Lambda_classification = [0, 0.00025, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.002]
    
    for Lambda_d in Lambda_distortion:
        for Lambda_c in Lambda_classification:
            for Lambda_p in Lambda_perception:    
                ##### MNIST
                settings.append({'dataset': 'mnist', 'latent_dim_1': latent_dim_1, 'L_1': L_1,
                                'Lambda_d': Lambda_d, 'Lambda_c': Lambda_c, 'Lambda_p': Lambda_p, 'n_critic': 1, 'n_epochs': 30, 'progress_intervals': 6,
                                'enc_layer_scale': 1, 'initialize_mse_model': 0, 'test_batch_size': 1000})
                ##### SVHN
                # settings.append({'dataset': 'svhn', 'latent_dim_1': latent_dim_1, 'L_1': L_1,
                #      'Lambda_d': Lambda_d, 'Lambda_c': Lambda_c, 'Lambda_p': Lambda_p, 'n_critic': 1, 'n_epochs': 80, 'progress_intervals': 10,
                #      'enc_layer_scale': 1, 'initialize_mse_model': 0, 'test_batch_size': 1000,
                #      'lr_encoder': 1e-4, 'lr_decoder': 1e-4, 'lr_critic': 1e-4,
                #      'beta1_encoder': 0.5, 'beta1_decoder': 0.5, 'beta1_critic': 0.5,
                #      'beta2_encoder': 0.999, 'beta2_decoder': 0.999, 'beta2_critic': 0.999})
    train_with_params(mode, settings, experiment_number, overwrite=False)