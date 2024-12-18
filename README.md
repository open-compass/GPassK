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
- **[2024.12.18]** We release the **[ArXiv Paper](http://arxiv.org/abs/2412.13147)** of GPassK. 🎉🎉🎉


## ☀️Introduction

**G-Pass@k** is a novel evaluation metric that provides a continuous assessment of model performance across multiple sampling attempts, quantifying both the model’s peak performance potential and its stability. In addition, it comes with **LiveMathBench**, a dynamic benchmark comprising challenging, contemporary mathematical problems designed to minimize data leakage risks during evaluation. In order to track the latest performance and stability of LLMs, we will continue updating the benchmark with new comptition level mathmatical problems and provide the latest results of the models on the benchmark with G-Pass@k.


## 🌲 Definition of GPassk
$$ \text{G-Pass@}k = \mathbb{E}_{\text{Questions}} \left[ \frac{{c \choose k}}{{n \choose k}} \right] $$ 

where $n$ represents the total number of generations per question, and $c$ denotes the number
of generations resulting in correct solutions.

$$ \text{G-Pass@}k_{\tau} = \mathbb{E}_{\text{Questions}} \left[ \sum_{j= \lceil \tau \cdot k \rceil}^{c} \frac{\binom{c}{j} \cdot \binom{n - c}{k - j}}{\binom{n}{k}} \right] $$

where $\lceil \tau \cdot k \rceil$ denotes the smallest integer greater than or equal to $\tau \cdot k$.


## 📚 Main Result
*LiveMathBench-202412 version*

<div align="center">
 <img src="https://github.com/user-attachments/assets/0e5d57c6-7fec-475e-acbe-cfa6aa2088cb" width="800"/>
</div>


## 🖋Use GPassK in OpenCompass
[OpenCompass](https://github.com/open-compass/opencompass) is a toolkit for evaluating the performance of large language models (LLMs). To use GPassK in OpenCompass, you can follow the steps below:
```python
Coming Soon...
```


# Citation and Tech Report
If you use GPassK in your research, please cite the following paper:
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
