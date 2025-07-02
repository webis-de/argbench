from vllm import LLM, SamplingParams
llm = LLM(model="/mnt/home/yajjour/pre-trained-models/Llama-3.3-70b-Instruct", tensor_parallel_size=4)
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