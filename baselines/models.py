#------------------------------------------------
# Author: T. Moon (Basic VQS)
# 
# improvements: C. Manouan 
# (
    # Resnet compatibility,
    # Roberta QstEncoder, 
    # encoder masking, 
    # model sanity checks with torchinfo
# )
#------------------------------------------------

import argparse
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import RobertaTokenizer, RobertaModel
## utils
from data_loader import get_loader
from torchinfo import summary


class ImgEncoder(nn.Module):

    def __init__(self, embed_size):
        """(1) Load the pretrained model as you want.
               cf) one needs to check structure of model using 'print(model)'
                   to remove last fc layer from the model.
           (2) Replace final fc layer (score values from the ImageNet)
               with new fc layer (image feature).
           (3) Normalize feature vector.
        """
        super(ImgEncoder, self).__init__()
        model = models.resnet18(pretrained=True)
        try:
            in_features = model.classifier[-1].in_features  # input size of feature vector
            model.classifier = nn.Sequential(
                *list(model.classifier.children())[:-1])    # remove last fc layer

            self.model = model                              # loaded model without last fc layer
            self.fc_ = nn.Linear(in_features, embed_size)    # feature vector of image
        except:
            in_features = model.fc.in_features  # input size of feature vector
            model.fc = nn.Sequential(
                *list(model.fc.children())[:-1])    # remove last fc layer

            self.model = model                              # loaded model without last fc layer
            self.fc_ = nn.Linear(in_features, embed_size)    # feature vector of image

    def forward(self, image):
        """Extract feature vector from image vector.
        """
        with torch.no_grad():
            img_feature = self.model(image)                  # [batch_size, vgg16(19)_fc=4096]
        img_feature = self.fc_(img_feature)                   # [batch_size, embed_size]

        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        img_feature = img_feature.div(l2_norm)               # l2-normalized feature vector

        return img_feature


class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size, bert_encoder=True):

        super(QstEncoder, self).__init__()
        # word2vec
        # self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        # activation function
        self.activation_fn = nn.Tanh()
        # roberta config
        self.encoder = RobertaModel.from_pretrained('roberta-base')
        # LSTM config
        # self.encoder = nn.LSTM(word_embed_size, hidden_size, num_layers)
        
        # projection layer
        bert_features = 768
        lstm_features = 2*num_layers*hidden_size
        self.dropout = nn.Dropout(p=.3)
        self.fc = nn.Linear(in_features=bert_features, out_features=embed_size)

    def forward(self, question):
        # ----- question encoding-----
        #------
        # LSTM
        #------
        # qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        # qst_vec = self.activation_fn(qst_vec)
        # qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
#         _, (hidden, cell) = self.encoder(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        
#         qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
#         qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
#         qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
#         qst_feature = self.activation_fn(qst_feature)
        
        #------
        # BERT
        #------ 
        # print(question)
        input_ids, attention_mask = question['input_ids'].squeeze(1), question['attention_mask'].squeeze(1)
        # print(input_ids.size(), attention_mask.size())
        qst_feature = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        ).pooler_output # the pooler applies Tanh as final activation 
        
        qst_feature = self.dropout(qst_feature)
        
        # output layer
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature


class VqaModel(nn.Module):

    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size, vision_only=False, question_only=False):

        super(VqaModel, self).__init__()
        self.vision_only = vision_only
        self.question_only = question_only
        
        if self.vision_only:
            self.img_encoder = ImgEncoder(embed_size)
        elif self.question_only:
            self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        else:
            self.img_encoder = ImgEncoder(embed_size)
            self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
            
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)
        

    def forward(self, img, qst):
        if self.vision_only:
            img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
            combined_feature = self.tanh(img_feature)
            combined_feature = self.dropout(combined_feature)
            combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size=1000]
            # combined_feature = self.tanh(combined_feature)
            # combined_feature = self.dropout(combined_feature)
            # combined_feature = self.fc2(combined_feature)           # [batch_size, ans_vocab_size=1000]
            
        elif self.question_only:
            qst_feature = self.qst_encoder(qst)                     # [batch_size, embed_size]
            combined_feature = self.tanh(qst_feature)
            combined_feature = self.dropout(combined_feature)
            combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size=1000]
            # combined_feature = self.tanh(combined_feature)
            # combined_feature = self.dropout(combined_feature)
            # combined_feature = self.fc2(combined_feature)           # [batch_size, ans_vocab_size=1000]     
            
        elif self.question_only==False and self.vision_only==False: # both submodels are used
            img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
            qst_feature = self.qst_encoder(qst)                     # [batch_size, embed_size]
            combined_feature = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
            combined_feature = self.tanh(combined_feature)
            combined_feature = self.dropout(combined_feature)
            combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size=1000]
            combined_feature = self.tanh(combined_feature)
            combined_feature = self.dropout(combined_feature)
            combined_feature = self.fc2(combined_feature)           # [batch_size, ans_vocab_size=1000]

        return combined_feature

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='./datasets',
                        help='input directory for visual question answering.')

    parser.add_argument('--log_dir', type=str, default='../../../logs',
                        help='directory for logs.')

    parser.add_argument('--model_dir', type=str, default='../../../models',
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

    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of processes working on cpu.')

    parser.add_argument('--save_step', type=int, default=1,
                        help='save step of model.')

    args = parser.parse_args()    
    
#     print('[INFO] building dataloders...')
    # data_loader = get_loader(
    #     input_dir=args.input_dir,
    #     input_vqa_train='train.npy',
    #     input_vqa_valid='valid.npy',
    #     max_qst_length=args.max_qst_length,
    #     max_num_ans=args.max_num_ans,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     for_bert=True
    # )

    qst_vocab_size = 17856
    ans_vocab_size = 1000
    
    # print('[INFO] Done !')
    model = VqaModel(
        embed_size=args.embed_size,
        qst_vocab_size=qst_vocab_size,
        ans_vocab_size=ans_vocab_size,
        word_embed_size=args.word_embed_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size, 
        vision_only=False, 
        question_only=True
    )
    
    # print(model)
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
        device="cpu"
    )
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
    
#     for batch_idx, batch_sample in enumerate(data_loader['train']):
#         image = batch_sample['image'].to(device)
#         question = batch_sample['question'].to(device)
#         label = batch_sample['answer_label'].to(device)
#         multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.

#         output = model(image, question)      # [batch_size, ans_vocab_size=1000]

#         print(output.size())
#         break