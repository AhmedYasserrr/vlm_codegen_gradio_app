# VLM CodeGen Gradio App
An interactive Gradio application for VQA and code generation, combining:

- LLaVA-NeXT for Image Description and Visual Question Answering
- DeepSeek-R1-Distill-Qwen for code generation and structured thinking

---

### Run on Google Colab

```python
!git clone https://github.com/AhmedYasserrr/vlm_codegen_gradio_app.git
%cd vlm_codegen_gradio_app
!pip install -r requirements.txt

from src.ui import GradioUI
ui = GradioUI()
app = ui.build()
app.launch()
```

### Setup Instructions (Local)

1. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:

   ```bash
   python app.py
   ```

3. Project structure:

   ```
   vlm_codegen_gradio_app/
   ├── app.py                   # Main entry point
   ├── notebooks/               # For testing notebooks
   ├── src/
   │   ├── models/
   │   │   ├── code_model.py    # DeepSeek model wrapper
   │   │   └── vlm_model.py     # LLaVA-NeXT model wrapper
   │   └── ui.py                # Gradio UI definition
   └── requirements.txt
   ```
---

### Prompt Engineering Strategies

#### DeepSeek-R1-Distill-Qwen

Prompt format:

```python
f"{prompt.strip()}\n\nPlease reason step by step, and put the final answer within \\boxed{{}}.\n<think>\n"
```
Without the `<think>` token, the model will hallucinate and treat your prompt as part of its own thought process.

This helps the model:
* Treat everything before `<think>` as the instruction
* Generate detailed reasoning inside the `<think>...</think>` block
* End the reasoning with `</think>` automatically
* I split the output at `</think>` token to show the final model output
  (I added a checkbox in the UI to toggle displaying the thinking process)

---

#### LLaVA-NeXT

LLaVA provides a built-in `apply_chat_template()` method:

```python
conversation = [{
    "role": "user",
    "content": [{"type": "image"}, {"type": "text", "text": prompt}]
}]
prompt_str = processor.apply_chat_template(conversation, add_generation_prompt=True)
```

Which transforms the input into:

```
[INST] <image> your_prompt_here [/INST]
```

The `<image>` token is replaced internally with image embeddings.

---

### Parameter Choices and Recommendations

| Parameter        | Recommended                                | Notes                                                                                                      |
| ---------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| temperature      | 0.5-0.7 (0.6)                              | Recommended for DeepSeek. Lower values reduce randomness but can lead to repetitive output.                |
| top\_p           | 0.9 – 0.95                                 | Values below 0.3 caused more concise outputs, but may oversimplify complex tasks.                          |
| max\_new\_tokens | 500–1000 for LLaVA, 2000–7000 for DeepSeek | LLaVA responses are shorter. DeepSeek may need more due to its thinking steps inside `<think>...</think>`. |

Note:
* Even if the “thinking” section is not shown by default, it still consumes tokens.
* Final visible output may appear short, but total tokens generated can be 2–3 times longer.
* The effect of top\_p effect may not always be noticeable, but higher values allow more diverse token sampling, often resulting in richer and occasionally longer outputs (especially during thinking).
---

### Future Optimizations

* LLaVA-NeXT has relatively slow inference; consider quantizing it using `bitsandbytes` to reduce latency.
* Enable beam search (e.g., `num_beams > 1`) for more coherent outputs, particularly for code generation.
* Use `gr.Blocks` instead of `gr.Interface` to enable more customized UI layouts.

