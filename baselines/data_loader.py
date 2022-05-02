#------------------------------------------------
# Author: T. Moon (Basic VQS)
# 
# improvements: C. Manouan (Roberta Tokenization)
#------------------------------------------------
import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from utils import text_helper
from transformers import RobertaTokenizer


class VqaDataset(data.Dataset):

    def __init__(self, input_dir, input_vqa, max_qst_length=30, max_num_ans=10, transform=None, for_bert=False):
        self.input_dir = input_dir
        self.vqa = np.load(input_dir+'/'+input_vqa, allow_pickle=True)
        self.qst_vocab = text_helper.VocabDict(input_dir+'/vocab_questions.txt')
        self.ans_vocab = text_helper.VocabDict(input_dir+'/vocab_answers.txt')
        self.max_qst_length = max_qst_length
        self.max_num_ans = max_num_ans
        self.load_ans = ('valid_answers' in self.vqa[0]) and (self.vqa[0]['valid_answers'] is not None)
        self.transform = transform
        self.for_bert = for_bert
        if self.for_bert:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def __getitem__(self, idx):

        vqa = self.vqa
        qst_vocab = self.qst_vocab
        ans_vocab = self.ans_vocab
        max_qst_length = self.max_qst_length
        max_num_ans = self.max_num_ans
        transform = self.transform
        load_ans = self.load_ans
        # preprocess image
        image = vqa[idx]['image_path']
        image = Image.open(image).convert('RGB')
        
        # preprocess question
        qst_tokens = vqa[idx]['question_tokens']
        qst_tokens_as_text = " ".join(qst_tokens)
        
        if self.for_bert:
            qst2idc = self.tokenizer(
                text=qst_tokens_as_text, 
                return_tensors='pt', 
                padding="max_length", 
                truncation=True,
                max_length=max_qst_length
            )
        else:
            qst2idc = np.array([qst_vocab.word2idx('<pad>')] * max_qst_length)  # padded with '<pad>' in 'ans_vocab'
            qst2idc[:len(vqa[idx]['question_tokens'])] = [qst_vocab.word2idx(w) for w in qst_tokens]
        
        sample = {'image': image, 'question': qst2idc}
        
        # preprocess answer if training/validation
        if load_ans:
            ans2idc = [ans_vocab.word2idx(w) for w in vqa[idx]['valid_answers']]
            ans2idx = np.random.choice(ans2idc)
            sample['answer_label'] = ans2idx         # for training

            mul2idc = list([-1] * max_num_ans)       # padded with -1 (no meaning) not used in 'ans_vocab'
            mul2idc[:len(ans2idc)] = ans2idc         # our model should not predict -1
            sample['answer_multi_choice'] = mul2idc  # for evaluation metric of 'multiple choice'

        if transform:
            sample['image'] = transform(sample['image'])

        return sample

    def __len__(self):

        return len(self.vqa)


def get_loader(
    input_dir, 
    input_vqa_train, 
    input_vqa_valid, 
    max_qst_length,
    max_num_ans, 
    batch_size, 
    num_workers, 
    for_bert=False
):

    transform = {
        phase: transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225))]) 
        for phase in ['train', 'valid']}

    vqa_dataset = {
        'train': VqaDataset(
            input_dir=input_dir,
            input_vqa=input_vqa_train,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            for_bert=for_bert,
            transform=transform['train']),
        'valid': VqaDataset(
            input_dir=input_dir,
            input_vqa=input_vqa_valid,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            for_bert=for_bert,
            transform=transform['valid'])}

    data_loader = {
        phase: torch.utils.data.DataLoader(
            dataset=vqa_dataset[phase],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)
        for phase in ['train', 'valid']}

    return data_loader


if __name__ == "__main__":
    
#     tfsm = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
#     ])
    
#     ds = VqaDataset(
#             input_dir="./datasets",
#             input_vqa='train.npy',
#             max_qst_length=30,
#             max_num_ans=10,
#             transform=tfsm,
#             for_bert=True
#     )
    
#     x = ds[225]
#     # print(x['question'])

    pass
