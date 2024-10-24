import logging
import math

from transformers import set_seed
from sentence_transformers import LoggingHandler
from sentence_transformers import CrossEncoder, InputExample
from datasets import load_from_disk

import random
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import numpy as np

### 하드 코딩된 부분 ###
# line 55 : odqa_dataset
# line 75 : test_size
# line 79 : model_name
# line 83 : hyperparameters
# line 102: model_save_path


# 회귀 문제를 위한 Evaluator
class RegressionEvaluator:
    def __init__(self, dev_samples, name=""):
        self.dev_samples = dev_samples
        self.name = name

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        pred_scores = model.predict([sample.texts for sample in self.dev_samples])
        true_labels = np.array([sample.label for sample in self.dev_samples])

        # Mean Squared Error 계산
        mse = np.mean((pred_scores - true_labels) ** 2)
        print(f"Epoch: {epoch}, Steps: {steps}, MSE: {mse}")

        return -mse  # return negative for better optimization (lower is better)


def train_cross_encoder():
    #### Just some code to print debug information to stdout
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )
    logger = logging.getLogger(__name__)
    #### /print debug information to stdout

    train_samples = []
    dev_samples = []

    # 데이터셋 경로 하드코딩 되어있음
    odqa_dataset = load_from_disk("./data/train_dataset")

    for example in odqa_dataset["train"]:
        question = example["question"]
        context = example["context"]
        answers = example["answers"]["text"]
        answer_starts = example["answers"]["answer_start"]

        label = 0.0  # 기본값은 negative example

        # answer가 context 내에 정확하게 위치해 있는지 확인
        for answer, start_idx in zip(answers, answer_starts):
            end_idx = start_idx + len(answer)

            # context의 해당 부분이 실제 answer와 일치하는지 확인
            if context[start_idx:end_idx] == answer:
                label = 1.0
                break  # 하나의 answer만 맞아도 positive로 처리

        # InputExample 형식으로 변환
        train_samples.append(InputExample(texts=[question, context], label=label))

    # train_samples가 다 모였다고 가정하고, 데이터를 섞음
    random.shuffle(train_samples)

    # 전체의 10%를 dev_samples로, 나머지를 train_samples로 분리
    train_samples, dev_samples = train_test_split(
        train_samples, test_size=0.1, random_state=42
    )

    # model_name이 하드코딩 되어있음
    model_name = "beomi/kcbert-base"

    cross_encoder = CrossEncoder(model_name, num_labels=1)

    # hyperparameters
    train_batch_size = 16
    num_epochs = 7
    learning_rate = 3e-5
    optimizer_params = {"lr": learning_rate}

    # batch_size가 하드코딩 되어있음
    train_dataloader = DataLoader(
        train_samples, shuffle=True, batch_size=train_batch_size
    )

    evaluator = RegressionEvaluator(dev_samples)

    warmup_steps = math.ceil(
        len(train_dataloader) * num_epochs * 0.1
    )  # 10% of train data for warm-up
    logger.info(f"Warmup-steps: {warmup_steps}")

    refined_model_name = model_name.replace("/", "_")
    model_save_path = f"./models/2nd_embedder/{refined_model_name}_cross-encoder"

    # Train the model
    cross_encoder.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=10000,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
        optimizer_params=optimizer_params,
    )


if __name__ == "__main__":

    set_seed(42)

    train_cross_encoder()
