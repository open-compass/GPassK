import re
from typing import Union, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from lighteval.metrics import Doc
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping, MetricCategory, MetricUseCase
from lighteval.utils.language import Language
from opencompass.models import OpenAISDK
from opencompass.registry import MODELS
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import numpy as np
from scipy.stats import hypergeom


api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

JUDGE_PROMPT_CN = '''请你作为一个数学阅卷专家，判断下面的答案是否与标准答案一致，即考生是否回答正确。下面是一些评判标准：
1. 有些答案可能包含多项内容，可能有单选题，多选题，填空题和问答题，只要答案与标准答案一致即可, 对于多选题和多个空的填空题，需要考生对应的选项或空都回答正确才算正确。
2. 有些答案可能通过不同的方式表达，比如有些答案可能是一个数学表达式，有些答案可能是一个文字描述，只要表达的意思一致即可。且有些公式通过不同的方式表达，但等价，也是正确的。
3. 你不需要重新计算问题答案，因为标准答案已经给出，只需要根据问题形式来判断考生的答案是否与标准答案一致，是否正确即可。

请你根据上述标准，判断下面的答案是否与标准答案一致，如果一致，请在最后输出\\boxed{{yes}}, 否则输出\\boxed{{no}}, 如果难以判断，请输出\\boxed{{no}}.
原问题：{question}
标准答案：{gold_answer}
考生答案：{answer}

分析：
'''

JUDGE_PROMPT_EN = '''Please act as an expert in grading mathematics exam papers, and judge whether the following answers match the standard answers, i.e., whether the examinee answered correctly. Here are some evaluation criteria:

1. Some answers may contain multiple parts, such as single-choice questions, multiple-choice questions, fill-in-the-blank questions, and problem-solving questions. As long as the answer matches the standard answer, it is considered correct. For multiple-choice questions and fill-in-the-blank questions with multiple blanks, the examinee must answer all corresponding options or blanks correctly to be considered correct.
2. Some answers may be expressed in different ways; for example, some answers may be mathematical expressions, while others may be textual descriptions. As long as the meaning conveyed is consistent, it is considered correct. Additionally, some formulas may be expressed differently but are equivalent, which is also considered correct.
3. You do not need to recalculate the problem answers, as the standard answers are already provided. You only need to judge whether the examinee's answer matches the standard answer based on the form of the question and whether it is correct.

Please judge whether the following answer matches the standard answer according to the above criteria. If they match, output \\boxed{{yes}}, otherwise output \\boxed{{no}}. If it is difficult to judge, also output \\boxed{{no}}.
Original Question: {question}
Standard Answer: {gold_answer}
Examinee's Answer: {answer}

Analysis:
'''


def compute_pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def _compute_g_pass_at_k(n, c, k, m):
    if m > min(c, k) or k > n or c < 0 or n <= 0 or m < 0:
        return 0.0
    return hypergeom.sf(m - 1, n, c, k)

def compute_g_pass_at_k(n, c, k, t):
    m = max(int(np.ceil(k * t)), 1)
    return _compute_g_pass_at_k(n, c, k, m)

def compute_mg_pass_at_k(n, c, k):
    l, r = int(np.ceil(k * 0.5)), k

    mg_pass_at_k = 0.0
    for i in range(l + 1, r + 1):
        mg_pass_at_k += _compute_g_pass_at_k(n, c, k, i)
    mg_pass_at_k = 2 * mg_pass_at_k / k

    return mg_pass_at_k


class RuleBasedMathVerifier:
    def __init__(self, k: Union[int, List]):
        if isinstance(k, int):
            k = [k]
        self.k = k

    def get_all_metrics(self):
        metrics = []
        for k in self.k:
            for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                metrics.append(f'G-Pass@{k}_{t}')
            metrics.append(f'mG-Pass@{k}')
        return metrics

    def _judge(self, prediction: str, answer: str) -> bool:
        gold_parsed = parse(
            answer,
            extraction_mode='first_match',
            extraction_config=[LatexExtractionConfig()]
        )
        pred_parsed = parse(
            prediction,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed='all',
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode='first_match'
        )
        return verify(gold_parsed, pred_parsed)

    def judge(self, predictions: List[str], formatted_doc: Doc, **kwargs) -> Dict[str, float]:
        results = []
        with ThreadPoolExecutor() as executor:
            for result in tqdm(executor.map(self._judge, predictions, 
                                            [formatted_doc.choices[formatted_doc.gold_index]] * len(predictions)), 
                               total=len(predictions), 
                               desc='Judging'):
                results.append(result)
        
        n = len(predictions)
        if n < max(self.k):
            raise ValueError(f'Totoal Number of Generations must Greater than or Equal '
                              'to Maximum Value of k: n={n}, max_k={max(self.k)}')
        c = sum(results)

        metrics = {}
        for k in self.k:
            for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                metrics[f'G-Pass@{k}_{t}'] = compute_g_pass_at_k(n, c, k, t)
            metrics[f'mG-Pass@{k}'] = compute_mg_pass_at_k(n, c, k)

        return metrics
    

class LLMJudgeMathVerifier:
    def __init__(self, k: Union[int, List], model_name: str, url: Union[str, List[str]]):
        if isinstance(k, int):
            k = [k]
        self.k = k
        
        if isinstance(url, str):
            url = [url]
        self.url = url
        self.models = [
            MODELS.build(
                dict(
                    type=OpenAISDK,
                    path=model_name,
                    openai_api_base=_url,
                    key='EMPTY',
                    query_per_second=30,
                    retry=20,
                    meta_template=api_meta_template,
                    temperature=0.0,
                    max_seq_len=16384,
                )) for _url in url
        ]

    def get_all_metrics(self):
        metrics = []
        for k in self.k:
            for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                metrics.append(f'G-Pass@{k}_{t}')
            metrics.append(f'mG-Pass@{k}')
        return metrics

    def extract_judge_label(self, text: str):
        if isinstance(text, str):
            match = re.findall(r'\\boxed{(.+?)}', text)
            if match:
                return match[-1]

        return None
    
    def get_prompt(self, prediction: str, doc: Doc) -> str:
        answer = doc.choices[doc.gold_index]
        question = doc.original_query
        language = Language.ENGLISH if doc.specific is None else doc.specific['language']

        PROMPT = JUDGE_PROMPT_CN if language == Language.CHINESE else JUDGE_PROMPT_EN
        return PROMPT.replace('{question}', question).replace('{gold_answer}', answer).replace('{answer}', prediction)

    def judge(self, predictions: List[str], formatted_doc: Doc, **kwargs) -> Dict[str, float]:
        results = []
        with ThreadPoolExecutor() as executor:
            tasks = []
            for i in range(len(predictions)):
                task = executor.submit(self.models[i % len(self.models)]._generate, 
                                       self.get_prompt(predictions[i], formatted_doc), 8192, 0.0)
                tasks.append(task)
            for result in tqdm(as_completed(tasks), 
                               total=len(predictions), 
                               desc='Judging'):
                generation = result.result()
                results.append(self.extract_judge_label(generation) == 'yes')

        n = len(predictions)
        if n < max(self.k):
            raise ValueError(f'Totoal Number of Generations must Greater than or Equal '
                             f'to Maximum Value of k: n={n}, max_k={max(self.k)}')
        c = sum(results)

        metrics = {}
        for k in self.k:
            for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                metrics[f'G-Pass@{k}_{t}'] = compute_g_pass_at_k(n, c, k, t)
            metrics[f'mG-Pass@{k}'] = compute_mg_pass_at_k(n, c, k)

        return metrics


def round2(x: List[float]):
    return round(100 * np.mean(x), 2)

def math_g_pass_at_k_metric(k: Union[int, List[int]], n: int, judge_kwargs: Dict = None):
    model_name = judge_kwargs.get('model_name', None)
    url = judge_kwargs.get('url', None)
    if model_name is not None and url is not None:
        judge_model = LLMJudgeMathVerifier(k, model_name, url)
    else:
        judge_model = RuleBasedMathVerifier(k)
    
    return SampleLevelMetricGrouping(
        # use : to split metric_name and num_generations
        metric_name=f'Math_G-Pass@k:{n}',
        higher_is_better={
            metric: True for metric in judge_model.get_all_metrics()
        },
        category=MetricCategory.GENERATIVE_SAMPLING,
        use_case=MetricUseCase.MATH,
        sample_level_fn=judge_model.judge,
        corpus_level_fn={
            metric: round2 for metric in judge_model.get_all_metrics()
        }
    )
