import os
'''
Set the cache dir instead of the default "$HOME/.cache"
Note: 
* set bash environment variable before importing the transforms module

Reference:
https://huggingface.co/docs/transformers/installation#cache-setup
https://stackoverflow.com/questions/63312859/how-to-change-huggingface-transformers-default-cache-directory/63314437#63314437
'''
os.environ['XDG_CACHE_HOME']='/Users/yingding/MODELS/'
from transformers import pipeline

# input="We are very happy to introduce pipeline to the transformers repository."
# input="chatgpt is a very strange thing."
input="Say something, you should say somthing."

generator_clf = pipeline(task='sentiment-analysis')
# generator = pipeline(model="openai/whisper-large")
result = generator_clf(input)

print(result)