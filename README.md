# Phoenics

Phoenics is an open source optimization algorithm combining ideas from Bayesian optimization with Bayesian Kernel Density estimation [1]. It performs global optimization on expensive to evaluate objectives, such as physical experiments or demanding computations. Phoenics supports sequential and batch optimizations and allows for the simultaneous optimization of multiple objectives via the Chimera scalarizing function [2].

Check out the `examples` folder for detailed descriptions and code examples for:

| Example | Link | 
|:--------|:-----|
| Sequential optimization           |  ![examples/optimization_sequential](https://github.com/aspuru-guzik-group/phoenics/tree/master/examples/optimization_sequential)  |
| Parallelizable batch optimization |  ![examples/optimization_parallel](https://github.com/aspuru-guzik-group/phoenics/tree/master/examples/optimization_parallel)  |
| Periodic parameter support        |  ![examples/optimization_periodic_parameters](https://github.com/aspuru-guzik-group/phoenics/tree/master/examples/optimization_periodic_parameters)  | 
| Multi-objective optimization      |  ... coming up soon ...  | 

More elaborate applications of Phoenics and Chimera are listed below

| Application 						  | Link                   | 
|:------------------------------------|:-----------------------|
| Auto-calibration of a virtual robot | ... coming up soon ... | 



## Installation

You can install Phoenics via pip

```
	apt-get install python-pip
	pip install phoenics
```

or by creating a conda environment from the provided environment file

```
	conda env create -f environment.yml
    source activate phoenics
```

Alternatively, you can also choose to build Phoenics from source by cloning this repository

```
	git clone https://github.com/aspuru-guzik-group/phoenics.git
```

##### Requirements

This code has been tested with Python 3.6 and uses
* cython 0.27.3
* json 2.0.9
* numpy 1.13.1
* scipy 0.19.1

Phoenics can construct its probabilistic model with two different probabilistic modeling libraries: PyMC3 and Edward. Depending on your preferences, you will either need 
* pymc3 3.2
* theano 1.0.1

or 
* edward 1.3.5
* tensorflow 1.4.1

Check out the `environment.yml` file for more details. 




### Using Phoenics

Phoenics is designed to suggest new parameter points based on prior observations. The suggested parameters can then be passed on to objective evaluations (experiments or involved computation). As soon as the objective values have been determined for a set of parameters, these new observations can again be passed on to Phoenics to request new, more informative parameters.

```python
from phoenics import Phoenics
    
    # create an instance from a configuration file
    config_file = 'config.json'
    phoenics    = Phoenics(config_file)
    
    # request new parameters from a set of observations
    params      = phoenics.choose(observations = observations)
```
Detailed examples for specific applications are presented in the `examples` folder. 


### Disclaimer

Note: This repository is under construction! We hope to add futther details on the method, instructions and more examples in the near future. 



### References

[1] Häse, F., Roch, L. M., Kreisbeck, C., & Aspuru-Guzik, A. (2018). Phoenics: A universal deep Bayesian optimizer. arXiv preprint [arXiv:1801.01469](https://arxiv.org/abs/1801.01469).  
[2] Häse, F., Roch, L. M., & Aspuru-Guzik, A. (2018). Chimera: enabling hierarchy based multi-objective optimization for self-driving laboratories. chemRxiv preprint [10.26434/chemrxiv.6195176.v1](https://chemrxiv.org/articles/Chimera_Enabling_Hierarchy_Based_Multi-Objective_Optimization_for_Self-Driving_Laboratories/6195176).