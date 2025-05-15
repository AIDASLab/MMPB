# python run.py --data MMPB --model Claude3-5V_Sonnet --verbose --injection_description_prompt_type image --generic_conversation_n_turn 0 --injection_prompt_n_images 2
# python run.py --data MMPB --model Claude3-7V_Sonnet --verbose --injection_description_prompt_type image --generic_conversation_n_turn 0 --injection_prompt_n_images 2
# python run.py --data MMPB --model Claude3-5V_Sonnet --verbose --injection_description_prompt_type image --generic_conversation_n_turn 10 --injection_prompt_n_images 2
# python run.py --data MMPB --model Claude3-7V_Sonnet --verbose --injection_description_prompt_type image --generic_conversation_n_turn 10 --injection_prompt_n_images 2
python run.py --data MMPB --model Claude3-7V_Sonnet --verbose --injection_description_prompt_type image --generic_conversation_n_turn 10 --wandb_exp_name default_main_results
# python run.py --data MMPB --model Claude3-7V_Sonnet --verbose --injection_description_prompt_type hard_moderate --generic_conversation_n_turn 10 