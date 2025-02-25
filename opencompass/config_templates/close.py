from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.livemathbench.livemathbench_gen import livemathbench_datasets

from opencompass.models import OpenAISDK


k = [4, 8, 16]
replication = 3
version = '202412'
temperatures = [1.0]

max_out_len = 8192

eval_urls = [
    # Put your judge model urls urls here
]
eval_model_name = 'Qwen/Qwen2.5-72B-Instruct'
api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    dict(
        abbr='o3-mini-2025-01-31'+ \
            f'_t{temperature}'
            f'_l{max_out_len}',
        type=OpenAISDK,
        path='o3-mini-2025-01-31',
        key='YOU_API_KEY',
        meta_template=api_meta_template,
        query_per_second=8,
        max_seq_len=max_out_len * 4,
        batch_size=8,
        max_completion_tokens=max_out_len,
        retry=20,
        run_cfg=dict(num_gpus=0)
    ) for temperature in temperatures
] + [
    dict(
        abbr='o1-mini-2024-09-12'+ \
            f'_t{temperature}'
            f'_l{max_out_len}',
        type=OpenAISDK,
        path='o1-mini-2024-09-12',
        key='YOU_API_KEY',
        meta_template=api_meta_template,
        query_per_second=8,
        max_seq_len=max_out_len * 4,
        batch_size=8,
        max_completion_tokens=max_out_len,
        retry=20,
        run_cfg=dict(num_gpus=0)
    ) for temperature in temperatures
] + [
    dict(
        abbr='Deepseek-R1'+ \
            f'_t{temperature}'
            f'_l{max_out_len}',
        type=OpenAISDK,
        path='deepseek-reasoner',
        key='YOU_API_KEY',
        openai_api_base='https://api.deepseek.com',
        tokenizer_path='gpt-4o-2024-05-13',
        meta_template=api_meta_template,
        query_per_second=2,
        max_seq_len=max_out_len * 4,
        temperature=temperature,
        batch_size=8,
        max_completion_tokens=max_out_len,
        retry=20,
        run_cfg=dict(num_gpus=0)
    ) for temperature in temperatures
]


livemathbench_dataset = livemathbench_datasets[0]
livemathbench_dataset.update(dict(
    k=k,
    replication=replication,
    dataset_splits=['CNMO', 'CCEE', 'AMC', 'WLPMC'], # set ['hard'] for hard split
    dataset_languages=['cn', 'en'],
    cot=True,
    version=version,
    abbr=f'LiveMathBench-v{version}-k{"_".join(map(str, [k] if isinstance(k, int) else k))}-r{replication}'
))
livemathbench_dataset['eval_cfg']['evaluator'].update(dict(
    model_name=eval_model_name,
    url=eval_urls,
    k=k,
    replication=replication 
))
livemathbench_dataset['infer_cfg']['inferencer'].update(dict(
    max_out_len=max_out_len
))
datasets = [livemathbench_dataset]
