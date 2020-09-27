import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import albumentations as A
from src.dataset import SegDataset
from src.engine import fit, train_one_epoch, evaluate
from src.callbacks import TrainMetricRecorder
from src.losses import  DiceLoss
from src.models import UNet
from src import logger

import argparse

import warnings
warnings.filterwarnings('ignore')


def train(args):
    
    train_dir = args.train_dir
    batch_size = args.batch_size
    model_save_path = args.model_save_path
    checkpoint_path = args.checkpoint_path
    img_size = args.img_size
    val_img_size = args.val_img_size
    epochs = args.epochs
    mixed_prec = args.mixed_prec
    batch_size = args.batch_size

    
    train_img_dir = os.path.join(train_dir, 'input')
    train_mask_dir = os.path.join(train_dir, 'output')

    args.train_img_dir = train_img_dir
    args.train_mask_dir = train_mask_dir
    
    #initialize logger and start loggging
    logger.initialize(prefix='train')
    #log all parameters
    logger.log(args)
    

    if model_save_path:
        if not os.path.exists(model_save_path):
            print(f'Creating directory: {model_save_path}')
            os.makedirs(model_save_path)

    ids = sorted([filename.split('.')[0] for filename in os.listdir(train_mask_dir) if filename.endswith('.png')])
    val_size = 100
    train_ids = ids[:-val_size]
    val_ids = ids[-val_size:]
    logger.log(f'Total samples: {len(ids)} Train samples: {len(train_ids)} Val samples: {len(val_ids)}')

    #define augmentation and preprocessing for train and test set
    train_transforms = A.Compose([ A.RandomCrop(height=img_size, width=img_size, p=1.0), 
                                A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.5), 
                                A.RandomRotate90(p=0.5),
                                A.Transpose(p=0.5),
                                A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25)])

    val_transforms = A.Resize(height=val_img_size, width=val_img_size, p=1)

    train_dataset = SegDataset(train_img_dir, train_mask_dir, train_ids, train_transforms)
    val_dataset = SegDataset(train_img_dir, train_mask_dir, val_ids, val_transforms)

    assert len(train_dataset) == len(train_ids)
    assert len(val_dataset) == len(val_ids)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
    
    #create device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.log(f'Using device: {device}')

    #create model
    model = UNet(n_channels=3, n_classes=1)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
        logger.log(f'Model weights loaded from path: {checkpoint_path} ')
     
    #move model to right device
    model.to(device)
    #define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001)
    #criterion
    criterion = DiceLoss()

    #metric
    recorder = TrainMetricRecorder(['iou', 'accuracy', 'precision', 'recall'])
    

    if mixed_prec:
        logger.log('Mixed precision training.')
    
    #train model
    fit(model, train_loader, val_loader, optimizer, criterion, epochs, device, recorder, mixed_prec=mixed_prec, model_save_path=model_save_path, logger=logger)



def test(args):
    test_dir = args.test_dir
    checkpoint_path = args.checkpoint_path
    test_img_size = args.test_img_size
    batch_size = args.batch_size
    test_img_dir = os.path.join(test_dir, 'input')
    test_mask_dir = os.path.join(test_dir, 'output')
    save_preds = args.save_preds
    args.test_img_dir = test_img_dir
    args.test_mask_dir = test_mask_dir

    #initialize logger and start loggging
    logger.initialize(prefix='test')
    #log all parameters
    logger.log(args)

    test_ids = sorted([filename.split('.')[0] for filename in os.listdir(test_mask_dir) if filename.endswith('.png')])
    logger.log(f'Test size: {len(test_ids)}')

    test_transforms = A.Resize(height=test_img_size, width=test_img_size, p=1)
    test_dataset = SegDataset(test_img_dir, test_mask_dir, test_ids, test_transforms)
    assert len(test_dataset) == len(test_ids)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)


    #create device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.log(f'Using device: {device}')

    #create model
    model = UNet(n_channels=3, n_classes=1)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
        logger.log(f'Model weights loaded from path: {checkpoint_path} ')
     
    #move model to right device
    model.to(device)
    
    #criterion
    criterion = DiceLoss()

    #metric
    recorder = TrainMetricRecorder(['iou', 'accuracy', 'precision', 'recall'])
    

    evaluate(model, test_loader, criterion, device, recorder)
    #calculate the test metric
    recorder.on_epoch_end()

    history = recorder.history
    msg_str = ''
    for key in recorder.history.keys():
        if len(history[key]) > 0:
            if 'val_' in key:
                display_key = key.replace('val_', 'test_')
            msg_str += f'{display_key}: {history[key][0]:.4f} '
    #log metric
    logger.log(msg_str)

    #save predicted images if save_preds is true
    #get the predictions from recorder and save in a directory
    if save_preds:
        prediction_dir = os.path.join(test_dir, 'predictions')
        #create directory if not present
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)

        for i, pred in enumerate(recorder.val_predictions):
            filepath = os.path.join(prediction_dir, f'img-{i+1}.png')
            pred = pred.cpu().numpy().squeeze()
            pred = ((pred>0.5)*255).astype('uint8')
            Image.fromarray(pred).save(filepath, 'PNG')

        logger.log(f'Predicted masks saved to directory: {prediction_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Road Segmentation')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--train_dir', type=str, required=True)
    parser_train.add_argument('--model_save_path', type=str, default=None)
    parser_train.add_argument('--checkpoint_path', type=str, default=None)
    parser_train.add_argument('--img_size', type=int, default=512)
    parser_train.add_argument('--val_img_size', type=int, default=1500)
    parser_train.add_argument('--batch_size', type=int, default=8)
    parser_train.add_argument('--learning_rate', type=float, default=1e-3)
    parser_train.add_argument('--epochs', type=int, default=10)
    parser_train.add_argument('--mixed_prec', action='store_true', default=False)

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--test_dir', type=str, required=True)
    parser_test.add_argument('--checkpoint_path', type=str, required=True)
    parser_test.add_argument('--test_img_size', type=int, default=1500)
    parser_test.add_argument('--batch_size', type=int, default=1)
    parser_test.add_argument('--save_preds', action='store_true', default=False)


    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)    
    else:    
        raise Exception('Invalid argument!')
