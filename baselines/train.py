import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchinfo import summary
from data_loader import get_loader
from models import VqaModel
from tqdm import tqdm
from transformers import get_scheduler

import logging
logging.basicConfig(level="INFO")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.cuda.amp.GradScaler()
LOG_INTERVAL = 20

VISION_ONLY = False
QUESTION_ONLY = True

def print_setup(args):
    logging.info(f"Maximum question length: {args.max_qst_length}")
    logging.info(f"Maximum number of answers: {args.max_num_ans}")
    logging.info(f"Embedding size of feature vector (img & qst): {args.embed_size}")
    # logging.info(f"Word embedding size (inp. to Recurrent): {args.word_embed_size}")
    # logging.info(f"Number of RNN layers: {args.num_layers}")
    # logging.info(f"RNN hidden size: {args.hidden_size}")
    logging.info(f"Vision LR: {args.learning_rate}")
    logging.info(f"Num epochs: {args.num_epochs}")
    logging.info(f"batch size: {args.batch_size}")
    logging.info(f"Step size (StepLR Scheduler): {args.step_size}")
    logging.info(f"Gamma (StepLR Scheduler): {args.gamma}")
    logging.info(f"Save model every {args.save_step} step(s)")
    logging.info(f"Vison only: {VISION_ONLY}")
    logging.info(f"Question only: {QUESTION_ONLY}")
    

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
        num_workers=args.num_workers, 
        for_bert=True
    )

    qst_vocab_size = data_loader['train'].dataset.qst_vocab.vocab_size
    ans_vocab_size = data_loader['train'].dataset.ans_vocab.vocab_size
    ans_unk_idx = data_loader['train'].dataset.ans_vocab.unk2idx
    
    logging.info(f"Question vocab. size: {qst_vocab_size}")
    logging.info(f"Answer vocab. size: {ans_vocab_size}")

    logging.info("Building model")
    model = VqaModel(
        embed_size=args.embed_size,
        qst_vocab_size=qst_vocab_size,
        ans_vocab_size=ans_vocab_size,
        word_embed_size=args.word_embed_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size, 
        vision_only=VISION_ONLY,
        question_only=QUESTION_ONLY
    ).to(device)
    
    bs = 4
    img = torch.rand(bs, 3, 224, 224)
    qst  = {
        'input_ids':torch.rand((bs, 30)).long(), 
        'attention_mask':torch.ones((bs, 30)).long()
    }
    
    # print(img.size(), qst['input_ids'].size(), qst['attention_mask'].size())
    summary(
        model=model, 
        input_data=[img, qst], 
        dtypes=[torch.float, dict], 
        device=device
    )
    
    logging.info("Training config...")
    criterion = nn.CrossEntropyLoss()
    try:
        params = list(model.img_encoder.fc_.parameters()) \
            + list(model.fc1.parameters()) \
            + list(model.fc2.parameters())
    except AttributeError:
        pass
    
    if VISION_ONLY:
        vision_optimizer = optim.Adam(params, lr=args.learning_rate)
        vision_scheduler = lr_scheduler.StepLR(optimizer=vision_optimizer, step_size=args.step_size, gamma=args.gamma)    
    
    elif QUESTION_ONLY:
        bert_optimizer = optim.AdamW(model.qst_encoder.parameters(), lr=5e-5)
        num_training_steps = args.num_epochs * len(data_loader['train'])
        bert_scheduler = get_scheduler(
            name="linear", 
            optimizer=bert_optimizer, 
            num_warmup_steps=0, 
            num_training_steps=num_training_steps
        )
    else:
        # vision optim
        vision_optimizer = optim.Adam(params, lr=args.learning_rate)
        vision_scheduler = lr_scheduler.StepLR(optimizer=vision_optimizer, step_size=args.step_size, gamma=args.gamma)

        # bert optim 
        bert_optimizer = optim.AdamW(model.qst_encoder.parameters(), lr=5e-5)
        num_training_steps = args.num_epochs * len(data_loader['train'])
        bert_scheduler = get_scheduler(
            name="linear", 
            optimizer=bert_optimizer, 
            num_warmup_steps=0, 
            num_training_steps=num_training_steps
        )
    

    epoch_bar = tqdm(range(args.num_epochs))
    epoch_bar.set_postfix({
        "epoch_loss":epoch_loss,
        "epoch_acc1":epoch_acc_exp1.item(),
        "epoch_acc2":epoch_acc_exp2.item()
    })
    for epoch in epoch_bar:

        for phase in ['train', 'valid']:

            running_loss = 0.0
            running_corr_exp1 = 0
            running_corr_exp2 = 0
            batch_step_size = len(data_loader[phase].dataset) / args.batch_size

            if phase == 'train':
                model.train()
            else:
                model.eval()
                if VISION_ONLY:
                    vision_scheduler.step()
                elif QUESTION_ONLY:
                    bert_scheduler.step()
                else:
                    vision_scheduler.step()
                    bert_scheduler.step()

            for batch_idx, batch_sample in enumerate(data_loader[phase]):
                iters = len(data_loader[phase])
                pct = 100. * batch_idx / iters
                epoch_bar.set_description(f"Epoch {epoch+1}/{args.num_epochs} - {phase} {pct:.2f}")

                image = batch_sample['image'].to(device)
                question = batch_sample['question'].to(device)
                label = batch_sample['answer_label'].to(device)
                multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.
                
                if VISION_ONLY:
                    vision_optimizer.zero_grad()
                elif QUESTION_ONLY:
                    bert_optimizer.zero_grad()
                else:
                    vision_optimizer.zero_grad()
                    bert_optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # AMP
                    with torch.cuda.amp.autocast():
                        output = model(image, question)      # [batch_size, ans_vocab_size=1000]
                        _, pred_exp1 = torch.max(output, 1)  # [batch_size]
                        _, pred_exp2 = torch.max(output, 1)  # [batch_size]
                        loss = criterion(output, label)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        if VISION_ONLY:
                            scaler.step(vision_optimizer)
                        elif QUESTION_ONLY:
                            scaler.step(bert_optimizer)
                        else:
                            scaler.step(vision_optimizer)
                            scaler.step(bert_optimizer)
                            
                        scaler.update()

                # Evaluation metric of 'multiple choice'
                # Exp1: our model prediction to '<unk>' IS accepted as the answer.
                # Exp2: our model prediction to '<unk>' is NOT accepted as the answer.
                pred_exp2[pred_exp2 == ans_unk_idx] = -9999
                running_loss += loss.item()
                running_corr_exp1 += torch.stack([(ans == pred_exp1.cpu()) for ans in multi_choice]).any(dim=0).sum()
                running_corr_exp2 += torch.stack([(ans == pred_exp2.cpu()) for ans in multi_choice]).any(dim=0).sum()

                # Print the average loss in a mini-batch.
                if batch_idx % LOG_INTERVAL == 0:
                    epoch_bar.set_postfix(
                        Step=f"{batch_idx:04d}/{int(batch_step_size):04d}",
                        batch_loss=loss.item(),
                        epoch_loss=epoch_loss,
                        epoch_acc1=epoch_acc_exp1.item(),
                        epoch_acc2=epoch_acc_exp2.item()                    
                    )


            # Print the average loss and accuracy in an epoch.
            epoch_loss = running_loss / batch_step_size
            epoch_acc_exp1 = running_corr_exp1.double() / len(data_loader[phase].dataset)      # multiple choice
            epoch_acc_exp2 = running_corr_exp2.double() / len(data_loader[phase].dataset)      # multiple choice

            epoch_bar.set_postfix({
                        "epoch_loss":epoch_loss,
                        "epoch_acc1":epoch_acc_exp1.item(),
                        "epoch_acc2":epoch_acc_exp2.item()
                    })

            # Log the loss and accuracy in an epoch.
            with open(os.path.join(args.log_dir, '{}-log-epoch-{:02}.txt')
                      .format(phase, epoch+1), 'w') as f:
                f.write(str(epoch+1) + '\t'
                        + str(epoch_loss) + '\t'
                        + str(epoch_acc_exp1.item()) + '\t'
                        + str(epoch_acc_exp2.item()))

        # Save the model check points.
        if (epoch+1) % args.save_step == 0:
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict()},
                       os.path.join(args.model_dir, 'model-epoch-{:02d}.ckpt'.format(epoch+1)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='./datasets',
                        help='input directory for visual question answering.')

    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='directory for logs.')

    parser.add_argument('--model_dir', type=str, default='./models',
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

    parser.add_argument('--num_epochs', '-ep', type=int, default=30,
                        help='number of epochs.')

    parser.add_argument('--batch_size', '-bs', type=int, default=512,
                        help='batch_size.')

    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of processes working on cpu.')

    parser.add_argument('--save_step', type=int, default=1,
                        help='save step of model.')

    args = parser.parse_args()

    print_setup(args)
    main(args)
