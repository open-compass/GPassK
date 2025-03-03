from itertools import product
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.livemathbench.livemathbench_gen import livemathbench_datasets

from opencompass.models import TurboMindModelwithChatTemplate


k = [4, 8, 16]
n = 48
version = '202412'
temperatures = [1.0]

max_out_len = 8192
top_p = 0.8
top_k = 50
repetition_penalty = 1.0
random_seed = 42

eval_urls = [
    # Put your judge model urls urls here
]
eval_model_name = 'Qwen/Qwen2.5-72B-Instruct'


llm_infos = [
    # model_name_or_path/tp/batch_size
    ('Qwen/Qwen2.5-7B-Instruct', 4, 64),
    ('Qwen/Qwen2.5-Math-7B-Instruct', 4, 64),
    ('meta-llama/Llama-3.1-8B-Instruct', 8, 64),
    ('meta-llama/Llama-3.1-70B-Instruct', 8, 64),
    ('meta-llama/Llama-3.3-70B-Instruct', 8, 64),
    ('mistralai/Mistral-Large-Instruct-2411', 8, 64),
    ('Qwen/Qwen2.5-32B-Instruct', 8, 64),
    ('Qwen/Qwen2.5-72B-Instruct', 8, 64),
    ('01-ai/Yi-1.5-34B-Chat', 8, 64),
    ('deepseek-ai/DeepSeek-V2.5-1210', 8, 16),
    ('google/gemma-2-27b-it', 8, 64),
    ('deepseek-ai/deepseek-math-7b-rl', 8, 64),
    ('internlm/internlm2-math-plus-20b', 8, 64),
    ('Qwen/Qwen2.5-Math-72B-Instruct', 8, 64)
]


models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr=llm_info[0].split('/')[-1] + \
            f'_t{temperature}'
            f'_p{top_p}'
            f'_k{top_k}'
            f'_rp{repetition_penalty}'
            f'_rs{random_seed}'
            f'_l{max_out_len}',
        path=llm_info[0],
        engine_config=dict(tp=llm_info[1]),
        gen_config=dict(
            do_sample=False if temperature < 1e-2 else True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            random_seed=random_seed
        ),
        backend='turbomind',         
        max_out_len=max_out_len,
        batch_size=llm_info[2],
        run_cfg=dict(num_gpus=llm_info[1])
    ) for llm_info, temperature in product(llm_infos, temperatures)
]


livemathbench_dataset = livemathbench_datasets[0]
livemathbench_dataset.update(dict(
    k=k,
    n=n,
    dataset_splits=['CNMO', 'CCEE', 'AMC', 'WLPMC'], # set ['hard'] for hard split
    dataset_languages=['cn', 'en'],
    cot=True,
    version=version,
    abbr=f'LiveMathBench-v{version}_k{"-".join(map(str, [k] if isinstance(k, int) else k))}_n{n}'
))
livemathbench_dataset['eval_cfg']['evaluator'].update(dict(
    model_name=eval_model_name,
    url=eval_urls
))
livemathbench_dataset['infer_cfg']['inferencer'].update(dict(
    max_out_len=max_out_len
))
datasets = [livemathbench_dataset]
