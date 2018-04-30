# Phoenics

Phoenics is an open source optimization algorithm combining ideas from Bayesian optimization with Bayesian Kernel Density estimation [1]. It performs global optimization on expensive to evaluate objectives, such as physical experiments or demanding computations. Phoenics supports sequential and batch optimizations and allows for the simultaneous optimization of multiple objectives [2].


## Installation

Phoenics can be installed using pip. 

```
	apt-get install python-pip
	pip install phoenics
```

You can also choose to build Phoenics from source by cloning this repository

```
	git clone https://github.com/aspuru-guzik-group/phoenics.git
```


Note: This repository is under construction! We hope to add futther details on the method, instructions and more examples in the near future. 

### References

[1] Häse, F., Roch, L. M., Kreisbeck, C., & Aspuru-Guzik, A. (2018). Phoenics: A universal deep Bayesian optimizer. arXiv preprint [arXiv:1801.01469](https://arxiv.org/abs/1801.01469).  
[2] Häse, F., Roch, L. M., & Aspuru-Guzik, A. (2018). Chimera: enabling hierarchy based multi-objective optimization for self-driving laboratories. chemRxiv preprint [10.26434/chemrxiv.6195176.v1](https://chemrxiv.org/articles/Chimera_Enabling_Hierarchy_Based_Multi-Objective_Optimization_for_Self-Driving_Laboratories/6195176).