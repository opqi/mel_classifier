#!/usr/bin/env python
import argparse
import torch
import os
import sys

from torch import nn
from torch.nn import functional as F

from data import get_df, data_loader, DataLoader
from utils import check_device, open_mel
from models import demo, resnet50, vit
from torch.autograd import Variable
from matplotlib import pyplot as plt


def train(model, train_dl: DataLoader,
          num_epoch: int, lr: float, device: str):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                    steps_per_epoch=int(
                                                        len(train_dl)),
                                                    epochs=num_epoch,
                                                    anneal_strategy='linear')
    print('Strat training')
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epoch):
        running_loss = 0
        correct_prediction = 0
        total_prediction = 0

        for data in train_dl:
            inputs, labels = data[0].to(device), data[1].to(device)

            inp_mean, inp_std = inputs.mean(), inputs.std()
            inputs = (inputs - inp_mean) / inp_std

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(
                Variable(outputs.detach(), requires_grad=True), labels.detach())
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            num_batches = len(train_dl)
            avg_loss = running_loss / num_batches
            acc = correct_prediction / total_prediction

            print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

    print('Training complited')


def evaluate(model, val_dl: DataLoader, device: str):
    print('Strat validation')
    correct_prediction = 0
    total_prediction = 0
    with torch.no_grad():
        for data in val_dl:
            inputs, labels = data[0].to(device), data[1].to(device)

            inp_mean, inp_std = inputs.mean(), inputs.std()
            inputs = (inputs - inp_mean) / inp_std

            outputs = model(inputs)

            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction / total_prediction
    print(
        f'Accuracy: {acc:.2f}, Correct predictions: {correct_prediction} out of Total predictions: {total_prediction}')


def test(model, mel: torch.tensor, results_path):
    print('Strat testing')

    logit_pred = model(mel)
    preds = F.softmax(logit_pred, dim=1).cpu().detach().numpy()

    plt.imshow(torch.squeeze(mel.cpu())[0])
    plt.title(
        f'Clean: {round(preds[0][0], 2) * 100}%; Noisy: {round(preds[0][1], 2) * 100}%')
    plt.savefig(results_path)
    print(f'The results are recorded in: {results_path}')


def main(data_path: str, launch_type: str,
         batch_size: int, epochs: int, lr: float,
         duration: int, channels: int, device: str,
         model_path, model_arch: str, num_classes: int,
         results_path: str):

    print('Start loading data')
    if launch_type == 'test':
        mel = open_mel(data_path, channels, duration).to(device)
    else:
        df = get_df(data_path)
        dl = data_loader(df, duration, batch_size, channels)

    device = check_device(device)

    if model_arch == 'resnet50':
        model = resnet50(num_classes, device)
    elif model_arch == 'vit':
        model = vit(num_classes, channels, duration, device)
    elif model_arch == 'demo':
        model = demo(channels, device)

    if device == 'cuda':
        model.cuda()

    if launch_type == 'train':
        train(model, dl, epochs, lr, device)
        torch.save(model.state_dict(), model_path)

    elif launch_type == 'val':
        model.load_state_dict(torch.load(model_path))
        model.eval()
        evaluate(model, dl, device)

    elif launch_type == 'test':
        model.load_state_dict(torch.load(model_path))
        model.eval()
        test(model, mel, results_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Mel Spectogram Classifier')
    parser.add_argument('data_path', type=str,
                        help='Path for the audio (.wav, .flac) or mel file(s) (.npy)')
    parser.add_argument('--launch_type', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Launch type')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--channels', type=int, default=3,
                        help='Input data chanels')
    parser.add_argument('--duration', type=int, default=800,
                        help='Mel spectogram duration')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Computation device; default: cuda')

    model_path = os.path.abspath(
        (os.path.join(os.path.join(__file__, '../'), 'model.pth')))
    parser.add_argument('--model', type=str, default=model_path,
                        help='Path for where the model will be stored or loaded from(.pth); default: model.pth')
    parser.add_argument('--model_arch', type=str, default='resnet50',
                        choices=['resnet50', 'vit', 'demo'],
                        help='Model Archeteture; default: resnet50')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes')
    parser.add_argument('--results_path', type=str, default='test_results.png',
                        help='Path where results will be stored (.png file); default: test_results.png')

    args = vars(parser.parse_args())

    if args['device'] == 'cuda':
        if not torch.cuda.is_available():
            print('CUDA is not available, starting on CPU')
            args['device'] = 'cpu'

    if not args['launch_type'] == 'train':
        if not os.path.exists(args['model']):
            print(f'No such file: {args["model"]}')
            sys.exit(1)

    main(args['data_path'], args['launch_type'],
         args['batch_size'], args['epochs'], args['lr'],
         args['duration'], args['channels'], args['device'],
         os.path.abspath(args['model']), args['model_arch'],
         args['num_classes'], args['results_path'])
