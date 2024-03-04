# DM-NonUniform
Official code for Accelerating Diffusion Sampling with Optimized Time Steps (CVPR 2024)
<div align="center">
  <a href="https://arxiv.org/pdf/2402.17376.pdf"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv">
</div>

> [**Accelerating Diffusion Sampling with Optimized Time Steps (CVPR 2024)**](https://arxiv.org/pdf/2402.17376.pdf)<br>
>Shuchen Xue, Zhaoqiang Liu&#8224;, Fei Chen, Shifeng Zhang, Tianyang Hu, Enze Xie, Zhenguo Li
> <br>University of Chinese Academy of Sciences, University of Electronic Science and Technology of China, Huawei Noah’s Ark Lab<br>
---

## Abstract

Discretization of sampling time steps are mainly hand-crafted designed, such as uniform-t scheme, quadratic-t scheme, uniform logSNR scheme and EDM scheme. We propose an optimization-based method to choose appropriate time steps for a specific numerical ODE solver for Diffusion Models.

## Integration with DPM-Solver, UniPC

Add the following code in "get_time_steps" method in DPM-Solver or UniPC

```python
from step_optim import StepOptim
```

```python
elif skip_type == "optimized":
    optimizer = StepOptim(self.noise_schedule)
    t, _ = optimizer.get_ts_lambdas(N, t_0, optimized_type)
    t = t.to(device).to(torch.float32)
    return t
```

For pixel-space diffusion models, we recommend use "optimized_type" as "unif", which means that the optimization algorithm will use uniform-logSNR steps as initialization; for latent-space diffusion models, we recommend use "optimized_type" = "unif_t", which means that the optimization algorithm will use uniform-time steps as initialization.

# Citation

If you find our work useful in your research, please consider citing:

```
@article{xue2024accelerating,
  title={Accelerating Diffusion Sampling with Optimized Time Steps},
  author={Xue, Shuchen and Liu, Zhaoqiang and Chen, Fei and Zhang, Shifeng and Hu, Tianyang and Xie, Enze and Li, Zhenguo},
  journal={arXiv preprint arXiv:2402.17376},
  year={2024}
}
```