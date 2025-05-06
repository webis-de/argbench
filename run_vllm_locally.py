from vllm import LLM, SamplingParams
llm = LLM(model="/bigwork/nhwpajjy/pre-trained-models/Llama-3.3-70B-Instruct")
print(llm.generate("hello"))