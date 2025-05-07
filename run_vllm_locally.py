from vllm import LLM
llm = LLM(model="/bigwork/nhwpajjy/pre-trained-models/Llama-3.3-70B-Instruct", tensor_parallel_size=2)
print(llm.generate("hello"))