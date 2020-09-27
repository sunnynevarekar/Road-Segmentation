import os
from datetime import datetime
import torch

if torch.__version__.startswith('1.6'):
    from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm


def train_one_epoch_amp(model, dataloader, optimizer, criterion, device, scaler, recorder=None):
    #set model in train mode
    model.train()
    
    #accumulate running loss
    running_loss = 0

    for inputs, labels in tqdm(dataloader):
        if device:
            #move data to right device
            inputs = inputs.to(device)
            labels = labels.to(device)
        
        with autocast():
            #forward pass
            logits = model(inputs)
            #calculate loss
            loss = criterion(logits, labels)

        #set gredients to zero
        optimizer.zero_grad()
        
        #scale loss
        scaler.scale(loss).backward()
        
        #unscale gradients and update weights
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        running_loss = running_loss + loss.item()
        #records batch labels and targets for computing metrics
        if recorder:
            with torch.no_grad():
                preds = torch.sigmoid(logits)
            
            recorder.on_train_batch_end(labels, preds, loss.item())
    
    return running_loss/len(dataloader)

def train_one_epoch(model, dataloader, optimizer, criterion, device=None, recorder=None):
    #set model in train mode
    model.train()
    
    #accumulate running loss
    running_loss = 0

    for inputs, labels in tqdm(dataloader):
        if device:
            #move data to right device
            inputs = inputs.to(device)
            labels = labels.to(device)

        #forward pass
        logits = model(inputs)

        #calculate loss
        loss = criterion(logits, labels)

        #set gredients to zero
        optimizer.zero_grad()

        #backward pass
        loss.backward()

        #update weights
        optimizer.step()

        running_loss = running_loss + loss.item()
        #records batch labels and targets for computing metrics
        if recorder:
            with torch.no_grad():
                preds = torch.sigmoid(logits)
            recorder.on_train_batch_end(labels, preds, loss.item())
    
    return running_loss/len(dataloader)


    

def evaluate(model, dataloader, criterion, device=None, recorder=None):
    #set model in eval mode
    model.eval()
    #accumulate loss
    running_loss = 0

    for inputs, labels in tqdm(dataloader):
        if device:
            #move data to right device
            inputs = inputs.to(device)
            labels = labels.to(device)    
        #forward pass to get logits
        #we dont need gradients for evaluation
        with torch.no_grad():
            logits = model(inputs)

        #calculate loss
        loss = criterion(logits, labels)
        running_loss += loss.item()

        if recorder:
            preds = torch.sigmoid(logits)
            recorder.on_val_batch_end(labels, preds, loss.item())
    
    return running_loss/len(dataloader)      
        

def fit(model, train_loader, va_loader, optimizer, criterion, epochs, device=None, recorder=None, mixed_prec=False, model_save_path=None, lr_scheduler=None, logger=None):
    
    #add current time to checkpoint path
    strft = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')

    if mixed_prec:
        scaler = GradScaler()

    for epoch in range(epochs):
            
        if recorder:
            recorder.on_epoch_start()
            
        if mixed_prec:
            train_loss = train_one_epoch_amp(model, train_loader, optimizer, criterion, device, scaler, recorder)
        else:
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, recorder)

        val_loss = evaluate(model, va_loader, criterion, device, recorder)
        
        
        if lr_scheduler:
            lr_scheduler.step()
        
        if recorder:
            recorder.on_epoch_end()
        
        print()
        if logger:
            if recorder:
                msg_str = f'Epoch {epoch+1}: '
                history = recorder.history
                for key in recorder.history.keys():
                    msg_str += f'{key}: {history[key][epoch]:.4f} '
                logger.log(msg_str)
            else:
                logger.log(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} val_loss: {val_loss: .4f}')        
        else:
            if recorder:
                msg_str = f'Epoch {epoch+1}: '
                history = recorder.history
                for key in recorder.history.keys():
                    msg_str += f'{key}: {history[key][epoch]:.4f} '
                print(msg_str)
            else:
                print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} val_loss: {val_loss: .4f}')     

        print()

        if model_save_path:
            path = os.path.join(model_save_path, f'epoch_{epoch+1}_{strft}.pth')
            # Save checkpoint
            torch.save(model.state_dict(), path)
            logger.log(f'Model saved to path: {path}')