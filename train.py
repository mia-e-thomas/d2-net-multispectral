import argparse
import numpy as np
import os
import shutil
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import warnings
#from lib.dataset import MegaDepthDataset # Not Needed
from lib.exceptions import NoGradientError
from lib.loss import loss_function
from lib.model import D2Net

# Added
from multipoint.datasets import ImagePairDataset
import yaml
import copy
from lib.utils import preprocess_batch

def main(): 

    # ---- Args ---- #
    # Argument parsing
    parser = argparse.ArgumentParser(description='Training script')
    # Dataset
    parser.add_argument('-y', '--yaml-config', required=True, help='YAML config file')
    # Not needed
    #parser.add_argument('--dataset_path', type=str, required=True, help='path to the dataset')
    #parser.add_argument('--scene_info_path', type=str, required=True, help='path to the processed scenes')
    parser.add_argument('--preprocessing', type=str, default='torch', help='image preprocessing (caffe or torch)')
    parser.add_argument('--model_file', type=str, default=None, help='path to the full model') # Changed default

    # TODO: Switch the rest of the args to a yaml file
    # Training Hyperparams
    parser.add_argument('--num_epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    # Dataset
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--use_validation', dest='use_validation', action='store_true', help='use the validation split')
    parser.set_defaults(use_validation=True) # Changed to true

    # Logging, plotting, checkpoints
    parser.add_argument('--log_interval', type=int, default=250, help='loss logging interval')
    parser.add_argument('--log_file', type=str, default='log.txt', help='loss logging file')
    parser.add_argument('--plot', dest='plot', action='store_true', help='plot training pairs')
    parser.set_defaults(plot=True) # Changed to true
    parser.add_argument('--checkpoint_directory', type=str, default='checkpoints',help='directory for training checkpoints')
    parser.add_argument('--checkpoint_prefix', type=str, default='d2', help='prefix for training checkpoints')

    args = parser.parse_args()
    print(args)

    # ---- Device ---- #
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Using device: " + str(device))

    # ---- Seed ---- #
    torch.manual_seed(1)
    if use_cuda:
        torch.cuda.manual_seed(1)
    np.random.seed(1)


    # ---- YAML ---- #
    with open(args.yaml_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # ---- Plotting ---- #
    # Create the folders for plotting if need be
    if args.plot:
        plot_path = 'train_vis'
        if os.path.isdir(plot_path):
            print('[Warning] Plotting directory already exists.')
        else:
            os.mkdir(plot_path)

    # ---- Model ---- #
    # Creating CNN model
    model = D2Net(
        model_file=args.model_file,
        use_cuda=use_cuda
    )

    # ---- Optimizer ---- #
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    # TODO: The original code set "train=False" when constructing validation set. See if there's some type of equivalent way you can do that.
    # ---- Dataset ---- #
    training_dataset = ImagePairDataset(config['dataset'])

    if args.use_validation:
        # Get training & val lengths 
        len_dataset = len(training_dataset.memberslist)
        # TODO: make this 20% an argument or yaml parameter
        len_validation = round(0.2*len_dataset)
        len_training = len_dataset - len_validation
        # Split training and validation
        training_dataset, validation_dataset = random_split(training_dataset, [len_training, len_validation])
        # Get dataloader
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Get dataloader
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


    # ---- Checkpoint Dir ---- #
    # Create the checkpoint directory
    if os.path.isdir(args.checkpoint_directory):
        print('[Warning] Checkpoint directory already exists.')
    else:
        os.mkdir(args.checkpoint_directory)
        
    # ---- Log File ---- #
    # Open the log file for writing
    if os.path.exists(args.log_file):
        print('[Warning] Log file already exists.')
    log_file = open(args.log_file, 'a+')

    # ---- Initialize ---- #
    # Initialize the history
    train_loss_history = []
    validation_loss_history = []
    if args.use_validation:
        #validation_dataset.build_dataset() # Not needed
        min_validation_loss = process_epoch(
            0,
            model, loss_function, optimizer, validation_dataloader, device,
            log_file, args,
            train=False
        )

    # Start the training
    # TODO: make this use tqdm
    for epoch_idx in range(1, args.num_epochs + 1):
        # Process epoch
        '''training_dataset.build_dataset()''' # From old code
        train_loss_history.append(
            process_epoch(
                epoch_idx,
                model, loss_function, optimizer, training_dataloader, device,
                log_file, args
            )
        )

        if args.use_validation:
            validation_loss_history.append(
                process_epoch(
                    epoch_idx,
                    model, loss_function, optimizer, validation_dataloader, device,
                    log_file, args,
                    train=False
                )
            )

        # TODO: change it not to save every epoch
        # Save the current checkpoint
        checkpoint_path = os.path.join(
            args.checkpoint_directory,
            '%s.%02d.pth' % (args.checkpoint_prefix, epoch_idx)
        )
        checkpoint = {
            'args': args,
            'epoch_idx': epoch_idx,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss_history': train_loss_history,
            'validation_loss_history': validation_loss_history
        }
        torch.save(checkpoint, checkpoint_path)
        if (args.use_validation and (validation_loss_history[-1] < min_validation_loss)):
            min_validation_loss = validation_loss_history[-1]
            best_checkpoint_path = os.path.join(
                args.checkpoint_directory,
                '%s.best.pth' % args.checkpoint_prefix
            )
            shutil.copy(checkpoint_path, best_checkpoint_path)

    # Close the log file
    log_file.close()

# ---- Epoch ---- #
# Define epoch function
def process_epoch(
        epoch_idx,
        model, loss_function, optimizer, dataloader, device,
        log_file, args, train=True
):
    epoch_losses = []

    torch.set_grad_enabled(train)

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_idx, batch in progress_bar:

        # TODO: Should this go before or after zero grad? Does it even matter?
        # ---- PREPROCESSING ---- #
        # Input:  Shape: [B,1,H,W], Range: 0-1, Type: Tensor Float
        # Output: Shape: [B,3,H,W], Range: 0-1* (mean/std normalization), Type: Tensor Double
        batch['image1'] = preprocess_batch(batch['optical']['image'], preprocessing=args.preprocessing)
        batch['image2'] = preprocess_batch(batch['thermal']['image'], preprocessing=args.preprocessing)

        if train:
            optimizer.zero_grad()

        batch['train'] = train
        batch['epoch_idx'] = epoch_idx
        batch['batch_idx'] = batch_idx
        batch['batch_size'] = args.batch_size
        batch['log_interval'] = args.log_interval
        batch['preprocessing'] = args.preprocessing # pass preprocess type so image can be corrected for viewing

        try:
            loss = loss_function(model, batch, device, plot=args.plot)
        except NoGradientError:
            continue

        current_loss = loss.data.cpu().numpy()[0]
        epoch_losses.append(current_loss)

        progress_bar.set_postfix(loss=('%.4f' % np.mean(epoch_losses)), epoch=epoch_idx)

        if batch_idx % args.log_interval == 0:
            log_file.write('[%s] epoch %d - batch %d / %d - avg_loss: %f\n' % (
                'train' if train else 'valid',
                epoch_idx, batch_idx, len(dataloader), np.mean(epoch_losses)
            ))

        if train:
            loss.backward()
            optimizer.step()

    log_file.write('[%s] epoch %d - avg_loss: %f\n' % (
        'train' if train else 'valid',
        epoch_idx,
        np.mean(epoch_losses)
    ))
    log_file.flush()

    return np.mean(epoch_losses)


if __name__ == "__main__":
    main()