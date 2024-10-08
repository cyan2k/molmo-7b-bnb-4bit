import sys
import os
from pathlib import Path

# this script assumes that you have removed the tensorflow dependency requirements by using the modified image_preprocessing_molmo.py provided in this repo

def set_cuda_paths():
    venv_base = Path(sys.executable).parent.parent
    nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
    cuda_path = nvidia_base_path / 'cuda_runtime' / 'bin'
    cublas_path = nvidia_base_path / 'cublas' / 'bin'
    cudnn_path = nvidia_base_path / 'cudnn' / 'bin'
    paths_to_add = [str(cuda_path), str(cublas_path), str(cudnn_path)]
    env_vars = ['CUDA_PATH', 'CUDA_PATH_V12_1', 'PATH']
    
    for env_var in env_vars:
        current_value = os.environ.get(env_var, '')
        new_value = os.pathsep.join(paths_to_add + [current_value] if current_value else paths_to_add)
        os.environ[env_var] = new_value

set_cuda_paths()

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

model_path = r"D:\Scripts\bench_vision\cyan2k--molmo-7B-D-bnb-4bit"

class VisionModel:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def initialize_model_and_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

    def process_single_image(self, image_path):
        image = Image.open(image_path)

        # Ensure the image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        text = "Describe this image in detail as possible but be succinct and don't repeat yourself."
        inputs = self.processor.process(images=[image], text=text)
        inputs = {k: v.to(self.device).unsqueeze(0) for k, v in inputs.items()}

        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=500, stop_strings=["<|endoftext|>"]),
            tokenizer=self.processor.tokenizer
        )

        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print(f"\nGenerated Text:\n{generated_text}\n")

if __name__ == "__main__":
    image_path = r"D:\Scripts\bench_vision\IMG_140531.JPG"

    vision_model = VisionModel()
    vision_model.initialize_model_and_processor()
    vision_model.process_single_image(image_path)
