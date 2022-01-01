import argparse

from termcolor import colored, cprint
import paddle
from paddlenlp.transformers import UnifiedTransformerLMHeadModel
from paddlenlp.transformers import UnifiedTransformerTokenizer

from utils import print_args, set_seed, select_response


class Config:
    def __init__(self):
        self.seed = None
        self.min_dec_len = 1
        self.max_dec_len = 64

        self.num_return_sequences = 20
        self.decode_strategy = 'sampling'
        self.top_k = 5

        self.temperature = 1.
        self.top_p = 1.
        self.num_beams = 0
        self.length_penalty = 1.

        self.early_stopping = False
        self.device = 'gpu'
        self.model_path = './model'


def interaction(conf, model, tokenizer):
    history = []
    start_info = "Enter [EXIT] to quit the interaction, [NEXT] to start a new conversation."
    cprint(start_info, "yellow", attrs=["bold"])
    while True:
        user_utt = input(colored("[Human]: ", "red", attrs=["bold"])).strip()


def predict(model, tok, conf: Config, his: list, u_input):
    if u_input == "[EXIT]":
        return '退出聊天'
    elif u_input == "[NEXT]":
        his.clear()
        return '开启新的聊天'
    else:
        his.append(u_input)
        inputs = tok.dialogue_encode(
            his,
            add_start_token_as_response=True,
            return_tensors=True,
            is_split_into_words=False)
        inputs['input_ids'] = inputs['input_ids'].astype('int64')
        ids, scores = model.generate(
            input_ids=inputs['input_ids'],
            token_type_ids=inputs['token_type_ids'],
            position_ids=inputs['position_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=conf.max_dec_len,
            min_length=conf.min_dec_len,
            decode_strategy=conf.decode_strategy,
            temperature=conf.temperature,
            top_k=conf.top_k,
            top_p=conf.top_p,
            num_beams=conf.num_beams,
            length_penalty=conf.length_penalty,
            early_stopping=conf.early_stopping,
            num_return_sequences=conf.num_return_sequences,
            use_faster=True)
        bot_response = select_response(
            ids,
            scores,
            tok,
            conf.max_dec_len,
            conf.num_return_sequences,
            keep_space=False)[0]
        his.append(bot_response)
        return bot_response


def initialize(conf: Config):
    paddle.set_device(conf.device)
    if conf.seed is not None:
        set_seed(conf.seed)

    model = UnifiedTransformerLMHeadModel.from_pretrained(conf.model_path)
    tokenizer = UnifiedTransformerTokenizer.from_pretrained(conf.model_path)

    model.eval()
    return model, tokenizer
    # interaction(conf, model, tok)


if __name__ == '__main__':
    config = Config()
    print_args(config)
    initialize(config)
