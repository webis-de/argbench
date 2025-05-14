from vllm import LLM, SamplingParams
llm = LLM(model="/bigwork/nhwpajjy/pre-trained-models/DeepSeek-R1-Distill-Qwen-1.5B")
sampling_params= {
    "temperature": 0.8,
    "top_p": 1,
    "top_k": -1,
    "max_tokens": 200,
    "min_tokens": 50,
    "truncate_prompt_tokens":2
}
sampling_params = SamplingParams(**sampling_params )
print(llm.generate("what is the capital of france?", sampling_params=sampling_params))