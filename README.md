# Molmo-7B-O and Molmo-7B-D BnB 4bit quants

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
pip install einops tensorflow-cpu
```

## Run

```
python molmo-7B-O-bnb-4bit.py

python molmo-7B-D-bnb-4bit.py
```
