## implementation adapted from https://github.com/tbmoon/basic_vqa
## The reported accuracy is exp2: (Multiple choice ingnoring the model's prediction to '<unk>') 

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from dataset import get_loader
from models import VqaModel
from tqdm import tqdm
import logging

from utils import train_one_epoch, validate_one_epoch
logging.basicConfig(level="INFO")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.cuda.amp.GradScaler()
LOG_INTERVAL = 20

def print_setup(args):
    logging.info(f"Maximum question length: {args.max_qst_length}")
    logging.info(f"Maximum number of answers: {args.max_num_ans}")
    logging.info(f"Embedding size of feature vector (img & qst): {args.embed_size}")
    logging.info(f"Word embedding size (inp. to Recurrent): {args.word_embed_size}")
    logging.info(f"Number of RNN layers: {args.num_layers}")
    logging.info(f"RNN hidden size: {args.hidden_size}")
    logging.info(f"LR: {args.learning_rate}")
    logging.info(f"Num. epochs: {args.num_epochs}")
    logging.info(f"batch size: {args.batch_size}")
    logging.info(f"Step size (StepLR Scheduler): {args.step_size}")
    logging.info(f"Gamma (StepLR Scheduler): {args.gamma}")
    logging.info(f"Save model every {args.save_step} step(s)")

def main(args):
    epoch_loss = np.inf
    epoch_acc_exp1 = torch.tensor(0.)
    epoch_acc_exp2 =torch.tensor(0.)
    logging.info("Loading datasets")
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    data_loader = get_loader(
        input_dir=args.input_dir,
        input_vqa_train='train.npy',
        input_vqa_valid='valid.npy',
        max_qst_length=args.max_qst_length,
        max_num_ans=args.max_num_ans,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    qst_vocab_size = data_loader['train'].dataset.qst_vocab.vocab_size
    ans_vocab_size = data_loader['train'].dataset.ans_vocab.vocab_size
    ans_unk_idx = data_loader['train'].dataset.ans_vocab.unk2idx

    logging.info("Building model")
    model = VqaModel(
        embed_size=args.embed_size,
        qst_vocab_size=qst_vocab_size,
        ans_vocab_size=ans_vocab_size,
        word_embed_size=args.word_embed_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size
    ).to(device)
    
    logging.info("Training config...")
    criterion = nn.CrossEntropyLoss()

    params = list(model.img_encoder.fc_.parameters()) \
        + list(model.qst_encoder.parameters()) \
        + list(model.fc1.parameters()) \
        + list(model.fc2.parameters())

    optimizer = optim.Adam(params, lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    epoch_bar = tqdm(range(args.num_epochs))
    epoch_bar.set_postfix({
        "epoch_loss":epoch_loss,
        "epoch_acc1":epoch_acc_exp1.item(),
        "epoch_acc2":epoch_acc_exp2.item()
    })
    for epoch in epoch_bar:
        model.train()
        train_acc = train_one_epoch(
            data_loader=data_loader, 
            model=model, 
            optimizer = optimizer,
            args=args, 
            epoch_bar=epoch_bar,
            device=device, 
            criterion=criterion, 
            epoch=epoch, 
            scaler=scaler, 
            LOG_INTERVAL=LOG_INTERVAL, 
            ans_unk_idx=ans_unk_idx
        )

        # validation
        model.eval()
        scheduler.step()

        val_acc = validate_one_epoch(
            data_loader=data_loader, 
            model=model, 
            optimizer = optimizer,
            args=args, 
            epoch_bar=epoch_bar,
            device=device, 
            criterion=criterion, 
            epoch=epoch, 
            LOG_INTERVAL=LOG_INTERVAL, 
            ans_unk_idx=ans_unk_idx
        )

        # Save the model check points.
        if (epoch+1) % args.save_step == 0:
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict()},
                        os.path.join(args.model_dir, 'model-epoch-{:02d}.ckpt'.format(epoch+1)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='../../datasets',
                        help='input directory for visual question answering.')

    parser.add_argument('--log_dir', type=str, default='../../logs',
                        help='directory for logs.')

    parser.add_argument('--model_dir', type=str, default='../../models',
                        help='directory for saved models.')

    parser.add_argument('--max_qst_length', type=int, default=30,
                        help='maximum length of question. \
                              the length in the VQA dataset = 26.')

    parser.add_argument('--max_num_ans', type=int, default=10,
                        help='maximum number of answers.')

    parser.add_argument('--embed_size', type=int, default=1024,
                        help='embedding size of feature vector \
                              for both image and question.')

    parser.add_argument('--word_embed_size', type=int, default=300,
                        help='embedding size of word \
                              used for the input in the LSTM.')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers of the RNN(LSTM).')

    parser.add_argument('--hidden_size', type=int, default=512,
                        help='hidden_size in the LSTM.')

    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001,
                        help='learning rate for training.')

    parser.add_argument('--step_size', type=int, default=10,
                        help='period of learning rate decay.')

    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multiplicative factor of learning rate decay.')

    parser.add_argument('--num_epochs', '-ep', type=int, default=10,
                        help='number of epochs.')

    parser.add_argument('--batch_size', '-bs', type=int, default=256,
                        help='batch_size.')

    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of processes working on cpu.')

    parser.add_argument('--save_step', type=int, default=1,
                        help='save step of model.')

    args = parser.parse_args()

    print_setup(args)
    main(args)
