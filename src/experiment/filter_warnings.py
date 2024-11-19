import warnings


user_warnings_to_turn_off = [
    'TypedStorage is deprecated',
    'torch.utils.checkpoint',
    'pad_token_id` should be positive but got -1',
    'Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 (default). This will lead to slow inference or training speed',
    'Could not find a config file in /bigwork/nhwpajjy/pre-trained-models/baffo32/decapoda-research-llama-7B-hf'
]
future_warnings_to_turn_off = [
    "evaluation_strategy` is deprecate"
]
def turn_off_warnings():
    for warning in user_warnings_to_turn_off:
        warnings.filterwarnings('ignore', category=UserWarning, message=warning)
    for warning in future_warnings_to_turn_off:
        warnings.filterwarnings('ignore', category=FutureWarning, message=warning)
