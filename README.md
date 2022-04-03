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

### 1. baseline models [1, 2, 3]

### Usage [From 1]

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
### 2. Our experiments


## References
[1]: baseline models implementation from [Taebong Moon' repository](https://github.com/tbmoon/basic_vqa).

[2]: Paper implementation
 - Paper: VQA: Visual Question Answering
 - URL: https://arxiv.org/pdf/1505.00468.pdf.

[3]: Preprocessing `TEXT_HEALPER`
 - Tensorflow implementation of N2NNM
 - Github: https://github.com/ronghanghu/n2nmn