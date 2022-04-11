### Drop all the helper functions here
from argparse import Namespace
import torch 
import torch.nn as nn
import os


def train_one_epoch(
	data_loader:dict, 
	model:nn.Module, 
	optimizer:torch,
	args:Namespace,
	epoch_bar,
	device,
	criterion,
	epoch:int,
	scaler,
	LOG_INTERVAL,
	ans_unk_idx
):
	running_loss = 0.0
	running_corr_exp1 = 0
	running_corr_exp2 = 0
	batch_step_size = len(data_loader["train"].dataset) / args.batch_size

	for batch_idx, batch_sample in enumerate(data_loader["train"]):
		iters = len(data_loader["train"])
		pct = 100. * batch_idx / iters
		epoch_bar.set_description(f"Epoch {epoch+1}/{args.num_epochs} - {'train'} {pct:.2f}")

		image = batch_sample['image'].to(device)
		question = batch_sample['question'].to(device)
		label = batch_sample['answer_label'].to(device)
		multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.

		optimizer.zero_grad()

		with torch.set_grad_enabled(True):
			# AMP
			with torch.cuda.amp.autocast():
				output = model(image, question)      # [batch_size, ans_vocab_size=1000]
				_, pred_exp1 = torch.max(output, 1)  # [batch_size]
				_, pred_exp2 = torch.max(output, 1)  # [batch_size]
				loss = criterion(output, label)

				scaler.scale(loss).backward()
				scaler.step(optimizer)
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
	epoch_acc_exp1 = running_corr_exp1.double() / len(data_loader["train"].dataset)      # multiple choice
	epoch_acc_exp2 = running_corr_exp2.double() / len(data_loader["train"].dataset)      # multiple choice


	epoch_bar.set_postfix({
				"epoch_loss":epoch_loss,
				"epoch_acc1":epoch_acc_exp1.item(),
				"epoch_acc2":epoch_acc_exp2.item()
			})

	# Log the loss and accuracy in an epoch.
	with open(os.path.join(args.log_dir, '{}-log-epoch-{:02}.txt')
				.format("train", epoch+1), 'w') as f:
		f.write(str(epoch+1) + '\t'
				+ str(epoch_loss) + '\t'
				+ str(epoch_acc_exp1.item()) + '\t'
				+ str(epoch_acc_exp2.item()))

	return epoch_acc_exp2.item()
	
def validate_one_epoch(
            data_loader:dict, 
            model:nn.Module, 
            optimizer,
            args:Namespace, 
            epoch_bar,
            device, 
            criterion, 
            epoch, 
            LOG_INTERVAL, 
            ans_unk_idx
):

	running_loss = 0.0
	running_corr_exp1 = 0
	running_corr_exp2 = 0
	batch_step_size = len(data_loader["train"].dataset) / args.batch_size

	for batch_idx, batch_sample in enumerate(data_loader["valid"]):
		iters = len(data_loader["train"])
		pct = 100. * batch_idx / iters
		epoch_bar.set_description(f"Epoch {epoch+1}/{args.num_epochs} - {'valid'} {pct:.2f}")

		image = batch_sample['image'].to(device)
		question = batch_sample['question'].to(device)
		label = batch_sample['answer_label'].to(device)
		multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.

		optimizer.zero_grad()

		with torch.no_grad():
			# AMP
			with torch.cuda.amp.autocast():
				output = model(image, question)      # [batch_size, ans_vocab_size=1000]
				_, pred_exp1 = torch.max(output, 1)  # [batch_size]
				_, pred_exp2 = torch.max(output, 1)  # [batch_size]
				loss = criterion(output, label)

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
	epoch_acc_exp1 = running_corr_exp1.double() / len(data_loader["valid"].dataset)      # multiple choice
	epoch_acc_exp2 = running_corr_exp2.double() / len(data_loader["valid"].dataset)      # multiple choice


	epoch_bar.set_postfix({
				"epoch_loss":epoch_loss,
				"epoch_acc1":epoch_acc_exp1.item(),
				"epoch_acc2":epoch_acc_exp2.item()
			})

	# Log the loss and accuracy in an epoch.
	with open(os.path.join(args.log_dir, '{}-log-epoch-{:02}.txt')
				.format("valid", epoch+1), 'w') as f:
		f.write(str(epoch+1) + '\t'
				+ str(epoch_loss) + '\t'
				+ str(epoch_acc_exp1.item()) + '\t'
				+ str(epoch_acc_exp2.item()))