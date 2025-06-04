from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

class VlmModel:
    def __init__(self, model_id="llava-hf/llava-v1.6-mistral-7b-hf"):
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate(self, prompt, image, temperature=0.6, top_p=0.9, max_new_tokens=500):
        conversation = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}]
        }]
        prompt_str = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(image, prompt_str, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens
        )
        decoded = self.processor.decode(output[0], skip_special_tokens=True)
        return decoded.split("[/INST]", 1)[-1].strip() if "[/INST]" in decoded else decoded.strip()
