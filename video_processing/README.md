## Video processing

`extract_photos.py` is a script that will extract streams of images of passing cars from input videos.

`label_photos.py` is a script that will label car passes (output from `extract_photos.py`) with license plates and match the identical cars into the same folder. It is required to have API point of LMStudio open and a model loaded (for example https://huggingface.co/lmstudio-community/gemma-4-E4B-it-GGUF).
