import yaml
from functools import partial
from itertools import product
from typing import Dict, Any, Union, List

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.metrics import Doc
from lighteval.utils.language import Language

from src.lighteval_metric import math_g_pass_at_k_metric


cfg = yaml.safe_load(open('configs/eval_cfg.yaml'))

PROMPT_CN = '''下面是一个{question_type}类型的数学问题，请逐步推理，并把最终答案放置于\\boxed{{}}中。
{question}
'''

PROMPT_EN = '''Here is a {question_type} type math problem, please reasoning step by step, and put your answer in \\boxed{{}}.
{question}
'''


class LiveMathBenchTask(LightevalTaskConfig):
    def __init__(self, subset: str, k: Union[int, List], n: int, judge_kwargs: Dict[str, Any], max_out_len: int):
        language = subset.split('_')[-1]
        assert (max(k) if isinstance(k, List) else k) <= n, 'Maximum value of `k` must less than or equal to `n`'
        return super().__init__(
            name=f"LiveMathBench-{subset}",
            prompt_function=partial(self.prompt_fn, language=language),
            hf_repo='opencompass/LiveMathBench',
            hf_subset=subset.replace(':', '_'),
            metric=[math_g_pass_at_k_metric(k, n, judge_kwargs)],
            hf_avail_splits=['test'],
            evaluation_splits=['test'],
            few_shots_split=None,
            few_shots_select=None,
            suite=['custom'],
            generation_size=max_out_len,
            stop_sequence=None,
        )

    def prompt_fn(self, line, task_name: str = None, language: str = 'en'):
        PROMPT = PROMPT_EN if language == 'en' else PROMPT_CN
        return Doc(
            task_name=task_name,
            query=PROMPT.replace('{question_type}', line['question_type']).replace('{question}', line['question']),
            original_query=line['question'],
            choices=[line['answer']],
            gold_index=0,
            specific={
                'language': Language.ENGLISH if language == 'en' else Language.CHINESE
            }
        )
    

def get_all_tasks(k: Union[int, List], n: int, judge_kwargs: Dict[str, Any], max_out_len: int):
    return [
        # use : to split parent task and subtasks, e.g., [LiveMathBench-v202412]:[CNMO_cn]
        LiveMathBenchTask(f'v202412:{split}_{lang}', k, n, judge_kwargs, max_out_len)
        for split, lang in product(['CNMO', 'CCEE', 'AMC', 'WLPMC',], ['cn', 'en'])
    ]


TASKS_TABLE = get_all_tasks(cfg['k'], cfg['n'], 
                            judge_kwargs=dict(model_name=cfg['eval_model'], url=cfg['eval_urls']),
                            max_out_len=cfg['max_out_len'])