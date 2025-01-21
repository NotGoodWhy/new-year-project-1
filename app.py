import gradio as gr
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification

class AIAssistant:
    def __init__(self):
        # 이미지 분류 모델 초기화
        self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.image_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        
        # 텍스트 생성 모델 초기화 (한국어 모델 사용)
        self.text_model = pipeline('text-generation', 
                                 model='skt/kogpt2-base-v2',
                                 tokenizer='skt/kogpt2-base-v2')

    def analyze_image(self, image):
        if image is None:
            return "이미지를 업로드해주세요."
        
        # 이미지 전처리
        inputs = self.image_processor(images=image, return_tensors="pt")
        outputs = self.image_model(**inputs)
        probs = outputs.logits.softmax(1)
        
        # 상위 3개 예측 결과 가져오기
        top3 = torch.topk(probs, 3)
        results = []
        for i in range(3):
            score = top3.values[0][i].item()
            label = self.image_model.config.id2label[top3.indices[0][i].item()]
            results.append(f"{label}: {score:.2%}")
        
        return "\n".join(results)

    def generate_text(self, prompt):
        if not prompt:
            return "프롬프트를 입력해주세요."
        
        # 텍스트 생성
        result = self.text_model(prompt, 
                               max_length=100, 
                               num_return_sequences=1,
                               pad_token_id=self.text_model.tokenizer.pad_token_id)
        
        return result[0]['generated_text']

    def combined_analysis(self, image, prompt):
        # 이미지 분석과 텍스트 생성을 결합
        image_result = self.analyze_image(image)
        text_result = self.generate_text(prompt)
        
        combined = f"이미지 분석 결과:\n{image_result}\n\n생성된 텍스트:\n{text_result}"
        return combined

# Gradio 인터페이스 생성
def create_interface():
    assistant = AIAssistant()
    
    with gr.Blocks(title="AI 도우미") as demo:
        gr.Markdown("# 🤖 AI 도우미")
        gr.Markdown("이미지 분석과 텍스트 생성을 도와드립니다.")
        
        with gr.Tabs():
            # 이미지 분석 탭
            with gr.Tab("이미지 분석"):
                with gr.Row():
                    image_input = gr.Image(type="pil", label="이미지 업로드")
                    image_output = gr.Textbox(label="분석 결과")
                image_button = gr.Button("이미지 분석하기")
                image_button.click(
                    fn=assistant.analyze_image,
                    inputs=image_input,
                    outputs=image_output
                )
            
            # 텍스트 생성 탭
            with gr.Tab("텍스트 생성"):
                with gr.Row():
                    text_input = gr.Textbox(label="프롬프트 입력", lines=2)
                    text_output = gr.Textbox(label="생성된 텍스트", lines=5)
                text_button = gr.Button("텍스트 생성하기")
                text_button.click(
                    fn=assistant.generate_text,
                    inputs=text_input,
                    outputs=text_output
                )
            
            # 통합 분석 탭
            with gr.Tab("통합 분석"):
                with gr.Row():
                    combined_image = gr.Image(type="pil", label="이미지 업로드")
                    combined_prompt = gr.Textbox(label="프롬프트 입력", lines=2)
                combined_output = gr.Textbox(label="통합 분석 결과", lines=8)
                combined_button = gr.Button("통합 분석하기")
                combined_button.click(
                    fn=assistant.combined_analysis,
                    inputs=[combined_image, combined_prompt],
                    outputs=combined_output
                )
        
        gr.Markdown("### 사용 방법")
        gr.Markdown("""
        1. 이미지 분석: 이미지를 업로드하면 이미지의 내용을 분석합니다.
        2. 텍스트 생성: 프롬프트를 입력하면 관련된 텍스트를 생성합니다.
        3. 통합 분석: 이미지와 프롬프트를 함께 입력하면 종합적인 분석 결과를 제공합니다.
        """)
        
    return demo

# 인터페이스 실행
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True) 