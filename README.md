# Molmo-7B-O and Molmo-7B-D BnB 4bit quants

Huggingface:

https://huggingface.co/cyan2k/molmo-7B-O-bnb-4bit

## Create venv

```
python -m venv venv
```


```
(mac/linux)
source venv/bin/activate

(win)
 .\venv\Scripts\activate
```


## Install dependencies

Fitting torch for your cuda version
```
(cuda 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

(cuda 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

(cuda 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Additional stuff
```
pip install einops tensorflow-cpu transformers accelerate bitsandbytes
```

## Run

```
python molmo-7B-O-bnb-4bit.py

# A dark gray cat with black stripes and a long tail is perched on a white carpet, its body oriented to the left while its head is turned to face the camera. The cat's striking yellow eyes with black pupils are prominently visible. Its ears are perked up, and its front legs are stretched out in front of it. The cat's tail extends off the right side of the image. Behind the cat, there is a white piano with a stack of books on top. An open book of sheet music is visible on the piano, with the pages spread out. The wall behind the piano is white, and the piano casts a shadow on it. The cat's whiskers are also noticeable, adding to its alert and attentive appearance.

python molmo-7B-D-bnb-4bit.py
```
