# Open-Domain Question Answering

Open-Domain Question Answering 프로젝트입니다.

## Requirements

```sh
# data (51.2 MB)
$ tar -xzf data.tar.gz

$ pip install -r requirements.txt
```

## Executuion

상세한 arguments 종류는 `args/` 폴더의 `arguments` 클래스들을 확인 부탁드립니다.

### train

```bash
# 학습 예시 (train_dataset 사용)
$ python train.py --output_dir ./models/train_dataset --do_train
```

MRC 모델의 평가는(`--do_eval`) 따로 설정해야 합니다. 위 학습 예시에 단순히 `--do_eval` 을 추가로 입력해서 훈련 및 평가를 동시에 진행할 수도 있습니다.

```bash
# mrc 모델 평가 (train_dataset 사용)
$ python train.py --output_dir ./outputs/train_dataset --model_name_or_path ./models/train_dataset/ --do_eval
```

### inference

retrieval 과 mrc 모델의 학습이 완료되면 `inference.py` 를 이용해 odqa 를 진행할 수 있습니다.

- 학습한 모델의 test_dataset에 대한 결과를 제출하기 위해선 추론(`--do_predict`)만 진행하면 됩니다.

```bash
# ODQA 실행 (test_dataset 사용)
$ python inference.py --output_dir ./outputs/test_dataset/ --dataset_name ./data/test_dataset/ --model_name_or_path ./models/train_dataset/ --do_predict
```

- 학습한 모델이 train_dataset 대해서 ODQA 성능이 어떻게 나오는지 알고 싶다면 평가(`--do_eval`)를 진행하면 됩니다.

```bash
# ODQA 실행 (test_dataset 사용)
$ python inference.py --output_dir ./outputs/train_dataset/ --model_name_or_path ./models/train_dataset/ --do_eval
```

### retrieval

retrieval 성능 평가를 진행할 수 있습니다.

```bash
$ python retrieval.py
```

결과는 아래와 같이 출력됩니다.

```
Metric          Value
-------------------------
recall@1        0.3368
recall@5        0.6164
recall@10       0.7171
recall@30       0.8438
recall@50       0.8829
MRR             0.4651
```

## Class Diagram

### Retriever

![retriever](/assets/retriever.png)

### Reader

![reader](/assets/reader.png)

## Datasets

아래는 제공하는 데이터셋의 분포를 보여줍니다.

![데이터 분포](./assets/dataset.png)

데이터셋은 편의성을 위해 Huggingface 에서 제공하는 datasets를 이용하여 pyarrow 형식의 데이터로 저장되어있습니다. 다음은 데이터셋의 구성입니다.

```bash
./data/                        # 전체 데이터
    ./train_dataset/           # 학습에 사용할 데이터셋. train 과 validation 으로 구성
    ./test_dataset/            # 제출에 사용될 데이터셋. validation 으로 구성
    ./wikipedia_documents.json # 위키피디아 문서 집합. retrieval을 위해 쓰이는 corpus.
```

## Collaborators

<h3 align="center">NLP-14조 Word Maestro(s)</h3>

<div align="center">

|          [김현서](https://github.com/kimhyeonseo0830)          |          [단이열](https://github.com/eyeol)          |          [안혜준](https://github.com/jagaldol)          |          [이재룡](https://github.com/So1pi)          |          [장요한](https://github.com/DDUKDAE)          |
| :------------------------------------------------------------: | :--------------------------------------------------: | :-----------------------------------------------------: | :--------------------------------------------------: | :----------------------------------------------------: |
| <img src="https://github.com/kimhyeonseo0830.png" width="100"> | <img src="https://github.com/eyeol.png" width="100"> | <img src="https://github.com/jagaldol.png" width="100"> | <img src="https://github.com/So1pi.png" width="100"> | <img src="https://github.com/DDUKDAE.png" width="100"> |

</div>
