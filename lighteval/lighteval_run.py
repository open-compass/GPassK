import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
import sys
sys.path.append('.')
import yaml
from datetime import timedelta
from itertools import product

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.models.model_input import GenerationParameters
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_accelerate_available

if is_accelerate_available():
    from accelerate import Accelerator, InitProcessGroupKwargs
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None

cfg = yaml.safe_load(open('configs/eval_cfg.yaml'))


def main():
    evaluation_tracker = EvaluationTracker(
        output_dir=f"./outputs/{cfg['model_name_or_path'].split('/')[-1]}",
        save_details=True
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        use_chat_template=True,
        custom_tasks_directory='src/lighteval_task.py'
    )

    model_config = VLLMModelConfig(
        pretrained=cfg['model_name_or_path'],
        dtype='bfloat16',
        tensor_parallel_size=cfg['tp'],
        use_chat_template=True,
        trust_remote_code=True,
        seed=42,
        max_model_length=2048 + cfg['max_out_len'],
        generation_parameters=GenerationParameters(max_new_tokens=cfg['max_out_len'],
                                                   temperature=cfg['temperature'],
                                                   top_p=cfg['top_p'],
                                                   top_k=cfg['top_k'],
                                                   repetition_penalty=cfg['repetition_penalty'],
                                                   seed=42))

    pipeline = Pipeline(
        tasks=','.join([f'custom|LiveMathBench-v202412:{split}_{lang}|0|0' 
                        for split, lang in product(['CNMO', 'CCEE', 'AMC', 'WLPMC',], ['cn', 'en'])]),
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()

if __name__ == "__main__":
    main()