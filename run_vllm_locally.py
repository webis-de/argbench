from vllm import LLM
llm = LLM(model="/bigwork/nhwpajjy/pre-trained-models/DeepSeek-R1-Distill-Qwen-32B", tensor_parallel_size=2)
print(llm.generate("hello"))