import pickle
import random
import logging

import cv2
import torch
import numpy as np
from tqdm import tqdm, trange
import requests
import os
def zip_file_in_parts(file_path, part_size_mb=99):
    # Define the size in bytes for the part size (49 MB)
    part_size = part_size_mb * 1024 * 1024  # Convert MB to Bytes
    base_name = os.path.splitext(file_path)[0]
    file_number = 1

    # Create a zip file for writing
    with zipfile.ZipFile(f'{base_name}_part{file_number}.zip', 'w', zipfile.ZIP_DEFLATED) as zip_file:
        current_size = 0

        # Open the original file for reading in binary mode
        with open(file_path, 'rb') as original_file:
            while True:
                # Read the data chunk
                chunk = original_file.read(part_size)
                if not chunk:  # If there is no data left to read, break
                    break
                
                # Check if adding this chunk would exceed the part size
                if current_size + len(chunk) > part_size:
                    # If yes, close the current zip file and start a new one
                    zip_file.close()
                    file_number += 1
                    zip_file = zipfile.ZipFile(f'{base_name}_part{file_number}.zip', 'w', zipfile.ZIP_DEFLATED)
                    current_size = 0  # Reset current size
                
                # Write the chunk to the zip file
                zip_file.writestr(os.path.basename(file_path), chunk)
                current_size += len(chunk)  # Update the current size
file_path = '/kaggle/working/LaneATT/laneatt_r18_tusimple/models/model_001.pt'

# Check if file exists


# Replace these with your bot token and chat ID
BOT_TOKEN = '7651391280:AAEqT4XRPZZTQNjyQvx_2FzRUNKDdc387BU'
CHAT_ID = '-134642039'
  # Replace with the path to the file you want to send

def send_file_to_telegram(bot_token, chat_id, file_path):
    url = f'https://api.telegram.org/bot{bot_token}/sendDocument'
    with open(file_path, 'rb') as file:
        response = requests.post(
            url,
            data={'chat_id': chat_id},
            files={'document': file}
        )
    
    # Check the response
    if response.status_code == 200:
        print("File sent successfully!")
    else:
        print("Failed to send file:", response.text)

# Run the function



class Runner:
    def __init__(self, cfg, exp, device, resume=False, view=None, deterministic=False):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.view = view
        self.logger = logging.getLogger(__name__)

        # Fix seeds
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        random.seed(cfg['seed'])

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self):
        self.exp.train_start_callback(self.cfg)
        starting_epoch = 1
        model = self.cfg.get_model()
        model = model.to(self.device)
        optimizer = self.cfg.get_optimizer(model.parameters())
        scheduler = self.cfg.get_lr_scheduler(optimizer)
        if self.resume:
            last_epoch, model, optimizer, scheduler = self.exp.load_last_train_state(model, optimizer, scheduler)
            starting_epoch = last_epoch + 1
        max_epochs = self.cfg['epochs']
        train_loader = self.get_train_dataloader()
        loss_parameters = self.cfg.get_loss_parameters()
        for epoch in trange(starting_epoch, max_epochs + 1, initial=starting_epoch - 1, total=max_epochs):
            self.exp.epoch_start_callback(epoch, max_epochs)
            model.train()
            pbar = tqdm(train_loader)
            for i, (images, labels, _) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = model(images, **self.cfg.get_train_parameters())
                loss, loss_dict_i = model.loss(outputs, labels, **loss_parameters)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Scheduler step (iteration based)
                scheduler.step()

                # Log
                postfix_dict = {key: float(value) for key, value in loss_dict_i.items()}
                postfix_dict['lr'] = optimizer.param_groups[0]["lr"]
                self.exp.iter_end_callback(epoch, max_epochs, i, len(train_loader), loss.item(), postfix_dict)
                postfix_dict['loss'] = loss.item()
                pbar.set_postfix(ordered_dict=postfix_dict)
            self.exp.epoch_end_callback(epoch, max_epochs, model, optimizer, scheduler)
            
            if os.path.exists(file_path):
                zip_file_in_parts(file_path)
                send_file_to_telegram(BOT_TOKEN, CHAT_ID, '/kaggle/working/LaneATT/laneatt_r18_tusimple/models/model_0001_part1.zip')
                send_file_to_telegram(BOT_TOKEN, CHAT_ID, '/kaggle/working/LaneATT/laneatt_r18_tusimple/models/model_0001_part2.zip')
                send_file_to_telegram(BOT_TOKEN, CHAT_ID, '/kaggle/working/LaneATT/laneatt_r18_tusimple/models/model_0001_part3.zip')
            else:
                print("File does not exist.")# Validate
            if (epoch + 1) % self.cfg['val_every'] == 0:
                self.eval(epoch, on_val=True)
        self.exp.train_end_callback()

    def eval(self, epoch, on_val=False, save_predictions=False):
        model = self.cfg.get_model()
        model_path = self.exp.get_checkpoint_path(epoch)
        self.logger.info('Loading model %s', model_path)
        model.load_state_dict(self.exp.get_epoch_model(epoch))
        model = model.to(self.device)
        model.eval()
        if on_val:
            dataloader = self.get_val_dataloader()
        else:
            dataloader = self.get_test_dataloader()
        test_parameters = self.cfg.get_test_parameters()
        predictions = []
        self.exp.eval_start_callback(self.cfg)
        with torch.no_grad():
            for idx, (images, _, _) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)
                output = model(images, **test_parameters)
                prediction = model.decode(output, as_lanes=True)
                predictions.extend(prediction)
                if self.view:
                    img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img, fp, fn = dataloader.dataset.draw_annotation(idx, img=img, pred=prediction[0])
                    if self.view == 'mistakes' and fp == 0 and fn == 0:
                        continue
                    cv2.imshow('pred', img)
                    cv2.waitKey(0)

        if save_predictions:
            with open('predictions.pkl', 'wb') as handle:
                pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)

    def get_train_dataloader(self):
        train_dataset = self.cfg.get_dataset('train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.cfg['batch_size'],
                                                   shuffle=True,
                                                   num_workers=8,
                                                   worker_init_fn=self._worker_init_fn_)
        return train_loader

    def get_test_dataloader(self):
        test_dataset = self.cfg.get_dataset('test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.cfg['batch_size'] if not self.view else 1,
                                                  shuffle=False,
                                                  num_workers=8,
                                                  worker_init_fn=self._worker_init_fn_)
        return test_loader

    def get_val_dataloader(self):
        val_dataset = self.cfg.get_dataset('val')
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=self.cfg['batch_size'],
                                                 shuffle=False,
                                                 num_workers=8,
                                                 worker_init_fn=self._worker_init_fn_)
        return val_loader

    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
