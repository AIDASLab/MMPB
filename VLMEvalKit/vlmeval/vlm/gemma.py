# from PIL import Image
# import torch

# from .base import BaseModel
# from ..smp import *

# class Gemma(BaseModel):    

#     def __init__(self, model_path='google/gemma-3-12b-it', **kwargs):
#         try:
#             from transformers import AutoProcessor, Gemma3ForConditionalGeneration
#         except Exception as e:
#             logging.critical('Please install the latest version transformers.')
#             raise e

#         model = Gemma3ForConditionalGeneration.from_pretrained(
#             model_path,            
#             device_map='cpu',            
#         ).eval()
#         self.model = model.cuda()
#         self.processor = AutoProcessor.from_pretrained(model_path)
#         self.kwargs = kwargs
    
#     def chat_inner(self, message, dataset=None):
        





#     # def generate_inner(self, message, dataset=None):
#     #     prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
#     #     image = Image.open(image_path).convert('RGB')

#     #     model_inputs = self.processor(
#     #         text=prompt, images=image, return_tensors='pt'
#     #     ).to('cuda')
#     #     input_len = model_inputs['input_ids'].shape[-1]

#     #     with torch.inference_mode():
#     #         generation = self.model.generate(
#     #             **model_inputs, max_new_tokens=3, do_sample=False
#     #         )
#     #         generation = generation[0][input_len:]
#     #         res = self.processor.decode(generation, skip_special_tokens=True)
#     #     return res