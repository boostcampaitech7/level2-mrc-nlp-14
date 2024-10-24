from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)


class QuestionAnsweringModelLoader:
    def __init__(self, model_args):

        config = AutoConfig.from_pretrained(
            (
                model_args.config_name
                if model_args.config_name is not None
                else model_args.model_name_or_path
            ),
            output_attentions=model_args.use_custom_loss,  # custom_loss 사용시 output_attentions 필요함
        )

        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            (
                model_args.tokenizer_name
                if model_args.tokenizer_name is not None
                else model_args.model_name_or_path
            ),
            use_fast=True,  # rust version tokenizer 사용 여부(좀 더 빠름)
        )

    def get_model_tokenizer(self):
        return self.model, self.tokenizer
