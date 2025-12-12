import os
from ...smp import load_env

INTERNAL = os.environ.get('INTERNAL', 0)


def build_judge(**kwargs):
    from ...api import OpenAIWrapper, SiliconFlowAPI
    model = kwargs.pop('model', None)
    kwargs.pop('nproc', None)
    load_env()
    LOCAL_LLM = os.environ.get('LOCAL_LLM', None)
    if LOCAL_LLM is None:
        model_map = {
            'gpt-4-turbo': 'gpt-4-1106-preview',
            'gpt-4-0613': 'gpt-4-0613',
            'gpt-4-0125': 'gpt-4-0125-preview',
            'gpt-4-0409': 'gpt-4-turbo-2024-04-09',
            'chatgpt-1106': 'gpt-3.5-turbo-1106',
            'chatgpt-0125': 'gpt-3.5-turbo-0125',
            'gpt-4o': 'gpt-4o-2024-05-13',
            'gpt-4o-0806': 'gpt-4o-2024-08-06',
            'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',
            'qwen-7b': 'Qwen/Qwen2.5-7B-Instruct',
            'qwen-72b': 'Qwen/Qwen2.5-72B-Instruct',
            'deepseek': 'deepseek-ai/DeepSeek-V2.5',
        }
        model_version = model_map[model]
    else:
        model_version = LOCAL_LLM

    if model in ['qwen-7b', 'qwen-72b', 'deepseek']:
        model = SiliconFlowAPI(model_version, **kwargs)
    else:
        model = OpenAIWrapper(model_version, **kwargs)
    return model


DEBUG_MESSAGE = """
To debug the OpenAI API, you can try the following scripts in python:
```python
from vlmeval.api import OpenAIWrapper
model = OpenAIWrapper('gpt-4o', verbose=True)
msgs = [dict(type='text', value='Hello!')]
code, answer, resp = model.generate_inner(msgs)
print(code, answer, resp)
```
You cam see the specific error if the API call fails.
"""

EVAL_MESSAGE = """
You will be provided with a <sks>’s textual description—covering appearance and preferences—and a set of question–answer pairs about an image. Your task is to identify whether each pair exhibits any of the following issues:

1. Trivial  
   A question is Trivial if it can be answered directly from the question itself or by using common sense, without needing the image or any extra information.  
   Example:  
     Q: What is <sks> doing in the movie theater?  
     Answer: Watching a movie.  
     Judgement: Trivial  
     (The question already states “movie theater,” and watching a movie in a movie theater is trivial.)

2. Text-Only  
   A question is Text-Only if it can be answered using only the provided textual description (appearance, preferences, etc.), without looking at the image.  
   Example:  
     Q: What is <sks>’s gender?  
     Answer: Male.  
     Judgement: Text-Only  
     (The description explicitly says <sks> is male, so there’s no need to inspect the picture.)

     Q: What is <sks>’s preference in terms of entertainmet?  
     Answer: Watching movie.  
     Judgement: Text-Only  
     (The description explicitly says <sks> likes wathcing movie, so there’s no need to inspect the picture.)
   Examine the answer choices carefully, because we’ve embedded multiple details about ’s preferences and appearance within them.

3. None
    The question itself has no issues, and you need to view both the image and the text to answer it.
    Example:  
     Q: Is <sks> in the image?  
     Answer: Yes  
     Judgement: None  

     Q: Is this image related to <sks>'s preference?  
     Answer: No  
     Judgement: None  

     (To answer this question, you’ll need to find an image of <sks> and its accompanying description.)

"""

EVAL_MESSAGE2 = """
Use exactly one of these labels—Trivial, Text‑Only and None—when annotating. Refer to the definitions and examples above.
Also, please answer the question by selecting one of the options. For example:
“Answer: Text‑only, C”
or
“Answer: None, B.”
"""