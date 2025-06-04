from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class CodeModel:
    def __init__(self, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate(self, prompt, temperature=0.6, top_p=0.9, max_new_tokens=2000, display_thinking=False):
        formatted_prompt = f"{prompt.strip()}\n\nPlease reason step by step, and put the final answer within \\boxed{{}}.\n<think>\n"
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens
        )
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if "</think>" in decoded:
            pre, post = decoded.split("</think>", 1)
            return (
                f"### Thinking\n\n{pre.strip()}</think>\n\n### Final Answer\n\n{post.strip()}"
                if display_thinking else post.strip()
            )
        return decoded.strip()
