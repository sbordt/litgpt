# ⚡ LitGPT with muP and advanced coordinate checks for Pythia models 

This repository is a fork of [LitGPT](https://github.com/Lightning-AI/litgpt) that implements training Pythia models with muP and advanced coordinate checks.

This repository was used for the experiments in the paper [On the Surprising Effectiveness of Large Learning Rates under Standard Width Scaling](https://arxiv.org/abs/2505.22491).
).

The important modifications to the code are in the following files:

- ```pretrain-experiment/pretrain-experiment.py```: Python script to run the experiments from the paper.  
- ```litgpt/monitor.py```: Adds the class ModuleMonitor to monitor the training of a Pytorch module. This is where the coordinate checks are implemented.
- ```litgpt/mup.py```: This is where the relevant primitives for training with muP are implemented (the current implementation only supports Pythia models, but it should be easy to extend this).
- ```litgpt/pretrain.py```: Modified to perform module monitoring during training. We also added some additional parameters to the script to support our experiments.



### License

LitGPT is released under the [Apache 2.0](https://github.com/Lightning-AI/litgpt/blob/main/LICENSE) license.

### Citation

If you use this software in your research, please cite the following works:

```bibtex
@article{haas2025splargelr,
  title={On the Surprising Effectiveness of Large Learning Rates under Standard Width Scaling},
  author={Haas, Moritz and Bordt, Sebastian and von Luxburg, Ulrike and Vankadara, Leena Chennuru},
  journal={arXiv:2505.22491},
  year={2025}
}
```

```bibtex
@misc{litgpt-2023,
  author       = {Lightning AI},
  title        = {LitGPT},
  howpublished = {\url{https://github.com/Lightning-AI/litgpt}},
  year         = {2023},
}
```

&nbsp;
