import gradio as gr
from .models import VlmModel, CodeModel

class GradioUI:
    def __init__(self):
        self.code_model = CodeModel()
        self.vlm_model = VlmModel()

    def build(self):
        code_tab = gr.Interface(
            fn=lambda prompt, temperature, top_p, max_new_tokens, show_thinking: self.code_model.generate(
                prompt, temperature, top_p, max_new_tokens, show_thinking
            ),
            inputs=[
                gr.Textbox(label="Prompt"),
                gr.Slider(0, 1, value=0.6, label="Temperature"),
                gr.Slider(0, 1, value=0.9, label="Top-p"),
                gr.Slider(10, 5000, value=2000, step=10, label="Max New Tokens"),
                gr.Checkbox(label="Show thinking steps", value=False)
            ],
            outputs=gr.Markdown(),
            title="Code Generation"
        )

        desc_tab = gr.Interface(
            fn=lambda image, temperature, top_p, max_new_tokens: self.vlm_model.generate(
                "Describe this image.", image, temperature, top_p, max_new_tokens
            ),
            inputs=[
                gr.Image(type="pil"),
                gr.Slider(0, 1, value=0.6, label="Temperature"),
                gr.Slider(0, 1, value=0.9, label="Top-p"),
                gr.Slider(10, 2000, value=500, step=10, label="Max New Tokens")
            ],
            outputs=gr.Textbox(label="Description"),
            title="Image Description"
        )

        vqa_tab = gr.Interface(
            fn=lambda prompt, image, temperature, top_p, max_new_tokens: self.vlm_model.generate(
                prompt, image, temperature, top_p, max_new_tokens
            ),
            inputs=[
                gr.Textbox(label="Prompt"),
                gr.Image(type="pil"),
                gr.Slider(0, 1, value=0.6, label="Temperature"),
                gr.Slider(0, 1, value=0.9, label="Top-p"),
                gr.Slider(10, 2000, value=500, step=10, label="Max New Tokens")
            ],
            outputs=gr.Textbox(label="Response"),
            title="Visual Question Answering"
        )

        return gr.TabbedInterface(
            [code_tab, desc_tab, vqa_tab],
            ["Code Generation", "Image Description", "VQA"]
        )
