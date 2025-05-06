from vllm import LLM, SamplingParams
llm = LLM(model="/mnt/home/yajjour/pre-trained-models/Llama-3.3-70B-Instruct")
print(llm.generate("hello"))