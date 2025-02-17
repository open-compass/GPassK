# GPassK: Are Your LLMs Capable of Stable Reasoning?

<div align="center">

<!-- [🏰[Project Page](https://github.com/open-compass/GPassK/)] -->
[📄[ArXiv Paper](http://arxiv.org/abs/2412.13147)]
[📚[LeaderBoard](https://open-compass.github.io/GPassK/)]
</div>


<div align="center">
 <img src="assets/pass-at-k-v-s-greedy-g-pass-at-k.png" width="800"/>
</div>

<!-- [🏰[Project Page](https://github.com/open-compass/GPassK/)]
[📚[LeaderBoard](https://github.com/open-compass/GPassK/index.html)] -->

## 🚀 News
- **[2025.2.13]** 🔥 We new results on LiveMathBench, MATH, and AIME24/25.
- **[2025.1.10]** 🔥 We release a small-scale judge model [LiveMath-Judge](https://huggingface.co/jnanliu/LiveMath-Judge).
- **[2025.1.6]** 🔥 **[LiveMathBench](https://huggingface.co/datasets/opencompass/LiveMathBench)** now can be accessed through hugginface, and you can now evaluate your LLMs on it using G-Pass@k in OpenCompass. We have addressed potential errors in LiveMathBench and inconsistencies in the sampling parameters. Please also refer to our updated version of the **[Paper](http://arxiv.org/abs/2412.13147)** for further details.
- **[2024.12.18]** 🎉 We release the **[ArXiv Paper](http://arxiv.org/abs/2412.13147)** of G-Pass@k. 


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

## 📚 Main Results

* ⚽: General Models
* 🏐: Math Models
* 🏀: o1-like Models

### *LiveMathBench-202412*

|LLMs|Greedy|G-Pass@16_0.5|G-Pass@16_0.75|G-Pass@16_1.0|mG-Pass@16|
|--|--|--|--|--|--|
|Llama-3.1-8B-Instruct ⚽|24.0|18.2|11.3|4.5|10.4|
|Qwen2.5-7B-Instruct ⚽|37.0|36.5|27.2|16.0|25.8|
|Llama-3.3-70B-Instruct ⚽|40.3|36.2|28.9|19.1|27.5|
|InternLM3-8B-Instruct ⚽|44.5|43.0|35.4|23.0|33.6|
|Claude-3.5-Sonnet ⚽|46.7|44.1|36.2|26.6|35.3|
|Mistral-Large-Instruct-2411 ⚽|41.6|39.4|37.1|32.9|36.4|
|Qwen2.5-Math-7B-Instruct 🏐|68.4|44.1|38.3|28.1|36.6|
|Qwen2.5-32B-Instruct ⚽|50.8|47.3|39.6|29.0|37.8|
|Qwen2.5-Max ⚽|52.9|52.7|44.3|31.1|42.2|
|Qwen2.5-Math-72B-Instruct 🏐|57.6|52.7|45.4|27.9|42.3|
|DeepSeek-Distill-Llama-8B 🏀|58.4|67.8|56.8|31.9|52.2|
|QwQ-32B-Preview 🏀|72.7|74.9|65.8|40.1|61.2|
|DeepSeek-Distill-Qwen-7B 🏀|65.6|73.0|66.4|48.4|63.1|
|OpenAI-o1-mini 🏀|74.1|76.3|67.3|48.3|64.8|
|DeepSeek-Distill-Qwen-32B 🏀|67.7|81.2|72.3|54.5|69.7|
|DeepSeek-Distill-Llama-70B 🏀|74.8|80.8|73.0|53.0|69.7|
|OpenAI-o3-mini 🏀|84.7|85.7|78.8|65.3|76.8|
|DeepSeek-R1 🏀|81.1|83.6|79.1|69.5|77.6|


### *LiveMathBench-Hard-202412*

|LLMs|Greedy|G-Pass@16_0.5|G-Pass@16_0.75|G-Pass@16_1.0|mG-Pass@16|
|--|--|--|--|--|--|
|Llama-3.1-8B-Instruct ⚽|2.2|0.8|0.0|0.0|0.0|
|Qwen2.5-7B-Instruct ⚽|13.3|6.2|3.2|2.2|3.3|
|Qwen2.5-Math-7B-Instruct 🏐|15.6|8.2|3.3|2.2|3.8|
|QwQ-32B-Preview 🏀|15.6|5.9|4.4|2.4|4.0|
|DeepSeek-Distill-Llama-8B 🏀|8.9|16.1|5.6|2.4|6.2|
|InternLM3-8B-Instruct ⚽|11.1|10.7|8.2|2.7|7.0|
|Qwen2.5-Math-72B-Instruct 🏐|11.1|11.8|7.9|5.9|7.9|
|DeepSeek-Distill-Qwen-7B 🏀|17.8|13.9|8.8|3.3|8.1|
|OpenAI-o1-mini 🏀|18.4|21.0|10.1|0.5|8.5|
|Qwen2.5-32B-Instruct ⚽|13.3|14.1|10.5|3.5|9.1|
|Qwen2.5-72B-Instruct ⚽|17.8|15.3|11.3|5.4|10.5|
|DeepSeek-Distill-Qwen-32B 🏀|22.2|29.9|16.9|3.3|15.1|
|DeepSeek-Distill-Llama-70B 🏀|35.6|33.1|19.0|5.8|17.3|
|OpenAI-o3-mini 🏀|43.3|47.4|32.5|7.7|28.6|
|DeepSeek-R1 🏀|42.2|46.6|33.6|9.8|29.6|

### *MATH500-L5*

|LLMs|Greedy|G-Pass@16_0.5|G-Pass@16_0.75|G-Pass@16_1.0|mG-Pass@16|
|--|--|--|--|--|--|
|Llama-3.1-8B-Instruct ⚽|26.1|17.8|10.7|3.5|9.7|
|InternLM3-8B-Instruct ⚽|51.5|49.9|40.3|26.9|38.3|
|Qwen2.5-7B-Instruct ⚽|56.0|54.9|43.3|28.0|41.5|
|Qwen2.5-72B-Instruct ⚽|63.4|62.5|54.4|44.9|53.1|
|Qwen2.5-Max ⚽|63.4|65.8|57.3|38.9|54.5|
|Qwen2.5-32B-Instruct ⚽|64.2|66.6|59.4|41.0|55.6|
|Qwen2.5-Math-72B-Instruct 🏐|71.6|64.9|59.4|46.0|57.4|
|Qwen2.5-Math-7B-Instruct 🏐|65.7|65.0|62.2|57.6|61.5|
|DeepSeek-Distill-Llama-8B 🏀|65.7|79.5|70.0|39.5|64.5|
|QwQ-32B-Preview 🏀|82.8|87.2|78.8|57.4|75.6|
|DeepSeek-Distill-Qwen-7B 🏀|78.4|87.9|80.5|62.6|77.6|
|DeepSeek-Distill-Qwen-32B 🏀|83.6|89.9|83.8|70.4|81.9|
|DeepSeek-Distill-Llama-70B 🏀|87.3|89.6|85.5|66.8|81.9|

### *AIME2024-45*

|LLMs|Greedy|G-Pass@16_0.5|G-Pass@16_0.75|G-Pass@16_1.0|mG-Pass@16|
|--|--|--|--|--|--|
|Llama-3.1-8B-Instruct ⚽|4.4|2.2|1.6|0.0|1.2|
|Qwen2.5-Math-7B-Instruct 🏐|11.1|4.6|2.6|2.2|3.7|
|Qwen2.5-32B-Instruct ⚽|11.1|7.1|3.4|2.2|3.7|
|InternLM3-8B-Instruct ⚽|11.1|7.2|4.3|1.0|3.7|
|Qwen2.5-7B-Instruct ⚽|11.1|8.9|8.1|4.7|7.5|
|Qwen2.5-Max ⚽|22.2|15.5|9.9|5.3|9.8|
|Qwen2.5-72B-Instruct ⚽|13.3|13.7|12.9|7.5|11.7|
|Qwen2.5-Math-72B-Instruct 🏐|20.0|18.7|16.2|6.7|14.1|
|QwQ-32B-Preview 🏀|44.4|41.0|28.6|8.1|24.7|
|DeepSeek-Distill-Llama-8B 🏀|44.4|53.9|30.4|9.0|28.0|
|DeepSeek-Distill-Qwen-7B 🏀|44.4|56.3|35.4|17.5|33.8|
|OpenAI-o1-mini 🏀|60.3|62.2|53.3|15.6|43.1|
|DeepSeek-Distill-Llama-70B 🏀|62.2|72.9|63.4|32.2|57.6|
|DeepSeek-Distill-Qwen-32B 🏀|62.2|77.0|66.5|31.3|59.3|

### *AIME2025*

|LLMs|Greedy|G-Pass@16_0.5|G-Pass@16_0.75|G-Pass@16_1.0|mG-Pass@16|
|--|--|--|--|--|--|
|Llama-3.1-8B-Instruct ⚽|0.0|8.9|0.0|0.0|0.0|
|InternLM3-8B-Instruct ⚽|13.3|6.7|0.1|0.0|0.8|
|Qwen2.5-32B-Instruct ⚽|20.0|11.5|0.2|0.0|1.4|
|Qwen2.5-7B-Instruct ⚽|6.7|9.7|6.2|0.2|4.7|
|Qwen2.5-72B-Instruct ⚽|20.0|12.2|5.8|0.1|4.9|
|Qwen2.5-Math-7B-Instruct 🏐|20.0|8.7|6.7|6.7|6.8|
|Qwen2.5-Max ⚽|13.3|11.9|6.8|2.9|6.8|
|Qwen2.5-Math-72B-Instruct 🏐|13.3|13.3|13.3|13.3|13.3|
|Gemini-2.0-Flash-Exp ⚽|26.7|26.5|21.5|14.0|21.2|
|QwQ-32B-Preview 🏀|26.7|34.5|32.4|15.6|28.1|
|OpenAI-o1-mini 🏀|46.7|39.9|32.5|14.0|28.4|
|DeepSeek-Distill-Llama-8B 🏀|40.0|40.4|21.2|7.9|21.0|
|DeepSeek-Distill-Qwen-7B 🏀|46.7|46.6|38.3|22.7|36.1|
|DeepSeek-Distill-Llama-70B 🏀|46.7|52.5|38.6|26.8|37.4|
|DeepSeek-R1 🏀|66.7|52.6|46.8|24.3|42.5|
|OpenAI-o3-mini 🏀|53.3|59.0|46.5|29.4|43.6|
|DeepSeek-Distill-Qwen-32B 🏀|46.7|59.7|50.2|29.5|47.3|


## 🖋Use G-Pass@k in OpenCompass
[OpenCompass](https://github.com/open-compass/opencompass) is a toolkit for evaluating the performance of large language models (LLMs). To use GPassK in OpenCompass, you can follow the steps below:

### 1. Prepare Environment
Follow these steps to ensure your environment is ready:

```bash
# Clone the main repository
git clone https://github.com/open-compass/GPassK.git
cd GPassK

# Create and activate a conda environment with specific Python and PyTorch versions
conda create -n livemathbench-eval python=3.10 pytorch torchvision torchaudio pytorch-cuda -c nvidia -c pytorch -y
conda activate livemathbench-eval

# Install additional required packages
pip install loguru

# Clone and install OpenCompass for extended functionality
git clone https://github.com/open-compass/opencompass.git opencompass
cd opencompass
pip install -e .
```


### 2. Prepare Dataset
LiveMathBench dataset can be obtained from HuggingFace. First, you should be granted to access the dataset from the following link: [huggingface](https://huggingface.co/datasets/opencompass/LiveMathBench).
Then, refer to [security-tokens](https://huggingface.co/docs/hub/security-tokens) to set up your HF tokens.


### 3. Deploy Judge Models
We leverage Qwen2.5-72B-Instruct as the judge model for judging the correctness of generated answers. We recommend to deploy services using deployment tools such as [vllm](https://github.com/vllm-project/vllm) or [lmdeploy](https://github.com/InternLM/lmdeploy) for invocation by different evaluation tasks.

Below is an example configuration for deploying the judge model using `lmdeploy`:
```bash
lmdeploy serve api_server Qwen/Qwen2.5-72B-Instruct --server-port 8000 \
    --tp 4 \ # at least 4 A100 or equivalent GPUs are required
    --cache-max-entry-count 0.9 \
    --log-level INFO 
```
After setting up the judge model, define the URLs in the `eval_urls` and `eval_model_name` within `opencompass_config_templates/*.py`. Adjust other parameters such as `k`， `temperatures`, `llm_infos`, and other params according to your needs.

> ❗️Note that omitting `eval_urls` will default to an internal rule-based judge, which might only apply to datasets with numerical answers 

> 💡Now you can use the [LiveMath-Judge](https://huggingface.co/jnanliu/LiveMath-Judge) for judging, which greatly reduces deploy and inference costs.

### 4. Evaluation

To begin the evaluation, first generate the necessary configuration files by running the following script:
```bash
python save_opencompass_configs.py --config_template_file {opencompass_config_templates/nono1.py|opencompass_config_templates/o1.py}
```

Upon execution, verify the generated configuration files located in `opencompass_configs/:

```
.
├── deepseek-math-7b-rl_t0-3_p0-8_k50_rp1-0_rs42_l8192@LiveMathBench-v202412-k4_8_16-r3.py
├── deepseek-math-7b-rl_t0-5_p0-8_k50_rp1-0_rs42_l8192@LiveMathBench-v202412-k4_8_16-r3.py
├── deepseek-math-7b-rl_t0-7_p0-8_k50_rp1-0_rs42_l8192@LiveMathBench-v202412-k4_8_16-r3.py
├── deepseek-math-7b-rl_t1-0_p0-8_k50_rp1-0_rs42_l8192@LiveMathBench-v202412-k4_8_16-r3.py
```

These files follow a naming convention that reflects the model settings and dataset used:
```
[MODEL_ABBR]_t[TEMPERATUE]_p[TOP_P]_k[TOP_K]_rp[REPETITION_PENALTY]_l[MAX_OUT_LEN]@[DATASET_ABBR]_k[LIST_OF_K]_r[REPLICATION].py
```

With the configurations prepared, initiate the evaluation process with the commands below:

```bash
cd GPassK
conda activate livemathbench-eval
python opencompass/run.py {path/to/config_file} \
      -w ./opencompass_outputs/ \
      --dump-eval-details \
```
Refer to the OpenCompass documentation for additional arguments that may enhance your evaluation experience 


## 📄 Citation and Tech Report
If you use G-Pass@k in your research, please cite the following paper:
```
@article{liu2024your,
  title={Are Your LLMs Capable of Stable Reasoning?},
  author={Liu, Junnan and Liu, Hongwei and Xiao, Linchen and Wang, Ziyi and Liu, Kuikun and Gao, Songyang and Zhang, Wenwei and Zhang, Songyang and Chen, Kai},
  journal={arXiv preprint arXiv:2412.13147},
  year={2024}
}
```
