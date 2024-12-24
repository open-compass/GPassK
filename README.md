# GPassK: Are Your LLMs Capable of Stable Reasoning?

<div align="center">

<!-- [🏰[Project Page](https://github.com/open-compass/GPassK/)] -->
[📄[ArXiv Paper](http://arxiv.org/abs/2412.13147)]
[📚[LeaderBoard](https://open-compass.github.io/GPassK/)]
</div>


<div align="center">
 <img src="https://github.com/user-attachments/assets/d91b1b5d-c932-402c-b86d-2846620a68b0" width="800"/>
</div>

<!-- [🏰[Project Page](https://github.com/open-compass/GPassK/)]
[📚[LeaderBoard](https://github.com/open-compass/GPassK/index.html)] -->

## 🚀 News
- **[2024.12.18]** We release the **[ArXiv Paper](http://arxiv.org/abs/2412.13147)** of G-Pass@k. 🎉🎉🎉


## ☀️Introduction

**G-Pass@k** is a novel evaluation metric that provides a continuous assessment of model performance across multiple sampling attempts, quantifying both the model’s peak performance potential and its stability. In addition, it comes with **LiveMathBench**, a dynamic benchmark comprising challenging, contemporary mathematical problems designed to minimize data leakage risks during evaluation. In order to track the latest performance and stability of LLMs, we will continue updating the benchmark with new comptition level mathmatical problems and provide the latest results of the models on the benchmark with G-Pass@k.


## 🌲 Definition of G-Pass@k
$$ \text{G-Pass@}k = \mathbb{E}_{\text{Questions}} \left[ \frac{{c \choose k}}{{n \choose k}} \right] $$ 

where $n$ represents the total number of generations per question, and $c$ denotes the number
of generations resulting in correct solutions.

$$ \text{G-Pass@}k_{\tau} = E_{\text{Questions}} \left[ \sum_{j = \lceil \tau \cdot k \rceil}^{c} \frac{\binom{c}{j} \cdot \binom{n - c}{k - j}}{\binom{n}{k}} \right] $$

where $\lceil \tau \cdot k \rceil$ denotes the smallest integer greater than or equal to $\tau \cdot k$.

$$ \text{mG-Pass@}k_{\tau} = 2\int_{0.5}^{1.0} \text{G-Pass@}k_{\tau} d \tau = \frac{2}{k} \sum_{i= \lceil 0.5 \cdot k \rceil + 1}^{k} \text{G-Pass@}k_{\frac{i}{k}} $$

Intuitively, $\text{mG-Pass@}k$ provides an interpolated estimate of the area under the curve of $\text{mG-Pass@}k_{[0.5:1.0]}$, serving as a comprehensive metric that integrates all $\text{G-Pass@}k_{\tau}$ values where $\tau \in [0.5, 1.0]$. 

## 📚 Main Result
*LiveMathBench-202412 version*

<div align="center">
 <img src="https://github.com/user-attachments/assets/0e5d57c6-7fec-475e-acbe-cfa6aa2088cb" width="800"/>
</div>


## 🖋Use G-Pass@k in OpenCompass
[OpenCompass](https://github.com/open-compass/opencompass) is a toolkit for evaluating the performance of large language models (LLMs). To use GPassK in OpenCompass, you can follow the steps below:

### 1. Prepare Environment
```bash
git clone https://github.com/open-compass/GPassK
cd GPassK
conda create -n livemathbench-eval python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate livemathbench-eval
pip install loguru
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
```


### 2. Prepare Dataset
You can access the LiveMathBench dataset from [huggingface](https://huggingface.co/datasets/opencompass/LiveMathBench).


### 3. Deploy Judge Models
We leverage Qwen2.5-72B-Instruct as the judge model for judging the correctness of generated answers. We recommend to deploy services using deployment tools such as [vllm](https://github.com/vllm-project/vllm) or [lmdeploy](https://github.com/InternLM/lmdeploy) for invocation by different evaluation tasks.

Here is an example using lmdeploy:
```bash
lmdeploy serve api_server Qwen/Qwen2.5-72B-Instruct --server-port 8000 \
    --tp 4 \ # at least 4 A100 or equivalent GPUs are required
    --cache-max-entry-count 0.9 \
    --log-level INFO 
```
Put your urls in definition of `eval_urls` in `opencompass_config_templates/*.py`. You can also modify other parameters, such as `k`， `temperatures`, and `llm_infos`.

### 4. Evaluation
First, you can run the script `save_opencompass_configs.py` to generate all opencompass config files:
```bash
python save_opencompass_configs.py --config_template_file {opencompass_config_templates/nono1.py|opencompass_config_templates/o1.py}
```

After running the script, you can check the opencompass config files in `opencompass_configs/`, such as:
```
.
├── deepseek-math-7b-rl_t0-3_p0-8_k50_rp1-0_rs42_l8192@LiveMathBench-v202412-k4_8_16-r3.py
├── deepseek-math-7b-rl_t0-5_p0-8_k50_rp1-0_rs42_l8192@LiveMathBench-v202412-k4_8_16-r3.py
├── deepseek-math-7b-rl_t0-7_p0-8_k50_rp1-0_rs42_l8192@LiveMathBench-v202412-k4_8_16-r3.py
├── deepseek-math-7b-rl_t1-0_p0-8_k50_rp1-0_rs42_l8192@LiveMathBench-v202412-k4_8_16-r3.py
```

Here, the file name of each opencompass config file follow the pattern:
```
[MODEL_ABBR]_t[TEMPERATUE]_p[TOP_P]_k[TOP_K]_rp[REPETITION_PENALTY]_l[MAX_OUT_LEN]@[DATASET_ABBR]_k[LIST_OF_K]_r[REPLICATION].py
```

Then, you can start evaluation by following commands：
```bash
cd GPassK
conda activate livemathbench-eval
python opencompass/run.py {path/to/config_file} \
      -w ./opencompass_outputs/ \
      --dump-eval-details \
```
You can check the documentations of opencompass for more useful arguments.


# Citation and Tech Report
If you use G-Pass@k in your research, please cite the following paper:
```
@misc{liu2024llmscapablestablereasoning,
      title={Are Your LLMs Capable of Stable Reasoning?}, 
      author={Junnan Liu and Hongwei Liu and Linchen Xiao and Ziyi Wang and Kuikun Liu and Songyang Gao and Wenwei Zhang and Songyang Zhang and Kai Chen},
      year={2024},
      eprint={2412.13147},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2412.13147}, 
}
```
