from vllm import LLM, SamplingParams
llm = LLM(model="/bigwork/nhwpajjy/pre-trained-models/DeepSeek-R1-Distill-Qwen-1.5B")
sampling_params = SamplingParams(truncate_prompt_tokens=2)
print(llm.generate("what is the capital of france?", sampling_params=sampling_params))