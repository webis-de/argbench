from vllm import LLM, SamplingParams
llm = LLM(model="/bigwork/nhwpajjy/pre-trained-models/DeepSeek-R1-Distill-Qwen-1.5B", device="cpu")
print(llm.generate("hello"))