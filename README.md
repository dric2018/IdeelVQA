# IdeelVQA

## Setup
### Project structure 

├── baselines\
│   ├── basic_vqa\
│   │   ├── png\
│   │   ├── __pycache__\
│   │   ├── tutorials\
│   │   └── utils\
│   │       └── __pycache__\
├── datasets\
│   ├── Annotations\
│   ├── Complementary_pairs\
│   ├── Images\
│   │   ├── test2015\
│   │   ├── train2014\
│   │   └── val2014\
│   ├── Questions\
│   ├── Resized_Images\
│   │   ├── test2015\
│   │   ├── train2014\
│   │   └── val2014\
│   └── zip\
├── logs\
├── models\
└── src

`PS: Download all datasets and save them under their approriate folders.`

You can create your virtual environment using anaconda with:
```
$ conda env create -f env.yml
```
Or install the dependencies using pip with
```
$(your-env) pip install -r requirements.txt
```
### A. baseline models [1, 2, 3]
> Evaluation metric of 'multiple choice'
- Exp1: [1]'s model prediction to '\<unk\>' IS accepted as the answer.
- Exp2: [1]'s model prediction to '\<unk\>' is NOT accepted as the answer **(The reported accuracy)**.
### Usage (From [Taebong Moon, 1])

1. Clone the repositories.
```
$ cd baselines && git clone https://github.com/tbmoon/basic_vqa.git
```
2. Download and unzip the dataset from official url of VQA: https://visualqa.org/download.html.
```
$ cd basic_vqa/utils
$ chmod +x download_and_unzip_datasets.csh
$ ./download_and_unzip_datasets.csh
```
3. Preproccess input data for (images, questions and answers).
```
$ python resize_images.py --input_dir='../../datasets/Images' --output_dir='../../datasets/Resized_Images'  
$ python make_vacabs_for_questions_answers.py --input_dir='../../datasets'
$ python build_vqa_inputs.py --input_dir='../../datasets' --output_dir='../../datasets'
```

4. Train model for VQA task.
```
$ cd ..
$ python train.py
```

### 2. Results
<div>
<center>

| ImageEncoder | QuestionEncoder | Accuracy | Dataset |  Metric | # Epochs |
|--------------|-----------------|------| ------- | ------ | ---- |
| None - Lang. alone [Agrawal et. al, 2] |LSTM Q: 1x1024 LSTM | 53.68 | VQA v2 | Multiple choice (All)| - |
| VGGNet-4096|None - Vision alone [Agrawal et. al, 2] | 30.53 | VQA v2 |  Multiple choice (All)| - |
| VGGNet | 2x2x512 LSTM| 63.09 | VQA v2 |  Multiple choice (All)| - |
| VGG19 [Taebong Moon, 1]| 2x2x512 LSTM| 54.72 | VQA v2 | Multiple choice (All) | 30 |
| VGG19| 2x2x512 LSTM (Our exp.)| **55.24** | VQA v2 | Multiple choice (All) | 10 |
| resnet18| 3x2x256 LSTM (Our exp.)| **56.44** | VQA v2 | Multiple choice (All) | 25 |
| ConvNext| None| 29.5 | VQA v2 | Multiple choice (All) | 30 |
| None| 2x2x512 LSTM| 49.7 | VQA v2 | Multiple choice (All) | 30 |
| ConvNext| 2x2x256 LSTM| 51.3 | VQA v2 | Multiple choice (All) | 30 |
| resnet18| None| 28.3 | VQA v2 | Multiple choice (All) | 30 |
| None| Roberta| 48.5 | VQA v2 | Multiple choice (All) | 30 |
| resnet18| Roberta| **59.1** | VQA v2 | Multiple choice (All) | 30 |
| MobilenetV2| None| TBA | VQA v2 | Multiple choice (All) | 30 |
| None| 2x2x256 LSTM| TBA | VQA v2 | Multiple choice (All) | 30 |
| MobilenetV2| 2x2x256 LSTM| TBA | VQA v2 | Multiple choice (All) | 30 |
| ViT| LSTM| 47 | VQA v2 | Multiple choice (All) | 30 |

</center>
</div>

## References
[1]: baseline models implementation from [Taebong Moon's repository](https://github.com/tbmoon/basic_vqa).

[2]: Paper implementation
 - Paper: VQA: Visual Question Answering
 - URL: https://arxiv.org/pdf/1505.00468.pdf.

[3]: Preprocessing `TEXT_HEALPER`
 - Tensorflow implementation of N2NNM
 - Github: https://github.com/ronghanghu/n2nmn
