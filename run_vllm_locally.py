from vllm import LLM, SamplingParams
llm = LLM(model="/bigwork/nhwpajjy/pre-trained-models/DeepSeek-R1-Distill-Qwen1.5B")
sampling_params = SamplingParams(truncate_prompt_tokens=10)
print(llm.generate("hello. i am not truncating you. you are truncating me. if we keep like this, we will truncate each other", sampling_params=sampling_params))