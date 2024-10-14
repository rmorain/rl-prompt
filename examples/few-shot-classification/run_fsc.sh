# python run_fsc.py \
#     'dataset=[sst-2,yelp-2,mr,cr,agnews,sst-5,yelp-5]' \
#     'dataset_seed=[0,1,2,3,4]' \
#     'prompt_length=[5]' \
#     'task_lm=[distilroberta-base,roberta-base,roberta-large,distilgpt2,gpt2,gpt2-medium,gpt2-large,gpt2-xl]' \
#     'random_seed=[42]'

python run_fsc.py \
    dataset=sst-2 \
    dataset_seed=0 \
    prompt_length=5 \
    task_lm=distilroberta-base \
    random_seed=42