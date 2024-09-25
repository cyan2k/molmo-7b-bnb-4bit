from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
)
from PIL import Image


repo_name = "cyan2k/molmo-7B-D-bnb-4bit"
arguments = {"device_map": "auto", "torch_dtype": "auto", "trust_remote_code": True}


# load the processor
processor = AutoProcessor.from_pretrained(repo_name, **arguments)

# load the model
model = AutoModelForCausalLM.from_pretrained(repo_name, **arguments)


# load image and prompt
inputs = processor.process(
    images=[Image.open("img/lucy.jpg")],
    text="Describe this image.",
)


inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

# generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
output = model.generate_from_batch(
    inputs,
    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
    tokenizer=processor.tokenizer,
)

# only get generated tokens; decode them to text
generated_tokens = output[0, inputs["input_ids"].size(1) :]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

# print the generated text
print(generated_text)
