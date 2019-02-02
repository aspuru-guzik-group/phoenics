# Phoenics

![](https://img.shields.io/github/release/aspuru-guzik-group/phoenics.svg?style=flat)
![](https://img.shields.io/github/license/aspuru-guzik-group/phoenics.svg?style=flat)

![](https://img.shields.io/github/issues-raw/aspuru-guzik-group/phoenics.svg?style=flat)

Phoenics is an open source optimization algorithm combining ideas from Bayesian optimization with Bayesian Kernel Density estimation [1]. It performs global optimization on expensive to evaluate objectives, such as physical experiments or demanding computations. Phoenics supports sequential and batch optimizations and allows for the simultaneous optimization of multiple objectives via the Chimera scalarizing function [2].

Check out the `examples` folder for detailed descriptions and code examples for:

| Example | Link | 
|:--------|:-----|
| Sequential optimization           |  [examples/optimization_sequential](https://github.com/aspuru-guzik-group/phoenics/tree/master/examples/optimization_sequential)  |
| Parallelizable batch optimization |  [examples/optimization_parallel](https://github.com/aspuru-guzik-group/phoenics/tree/master/examples/optimization_parallel)  |
| Periodic parameter support        |  [examples/optimization_periodic_parameters](https://github.com/aspuru-guzik-group/phoenics/tree/master/examples/optimization_periodic_parameters)  | 
| Multi-objective optimization      |  [examples/optimization_multiple_objectives](https://github.com/aspuru-guzik-group/phoenics/tree/master/examples/optimization_multiple_objectives)  | 

More elaborate applications of Phoenics and Chimera are listed below

| Application 						  | Link                   | 
|:------------------------------------|:-----------------------|
| Auto-calibration of a virtual robot | [examples/application_robot_calibration](https://github.com/aspuru-guzik-group/phoenics/tree/master/examples/application_robot_calibration) | 



# Chimera

Chimera is a general purpose achievement scalarizing function for multi-objective optimization. User preferences regarding the objectives are expected in terms of an importance hierarchy, as well as relative tolerances on each objective indicating what level of degradation is acceptable. Chimera is integrated into Phoenics, but also available for download as a wrapper for other optimization methods (see [chimera](https://github.com/aspuru-guzik-group/phoenics/tree/master/chimera)).


# Installation

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




# Using Phoenics

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


# Using Chimera

Chimera is integrated into Phoenics, but also available as a stand-alone wrapper for other single-objective optimization algorithms. The Chimera wrapper allows to cast a set of objectives for a number of observations into a single objective value for each observation, enabling single-objective optimization algorithms to solve the multi-objective optimization problem. The usage of Chimera is outlined below on an example with four objective functions

```python
from chimera import Chimera

# define tolerances in descending order of importance
tolerances = [0.25, 0.1, 0.25, 0.05]

# create Chimera instance
chimera = Chimera(tolerances)

# cast objectives of shape      [num_observations, num_objectives]
# into single objective vector  [num_observations, 1]
single_objectives = chimera.scalarize_objectives(objectives)

```

**Note**: Phoenics automatically employs Chimera when the configuration contains more than one objective.

### Disclaimer

Note: This repository is under construction! We hope to add further details on the method, instructions and more examples in the near future. 

### Experiencing problems? 

Please create a [new issue](https://github.com/aspuru-guzik-group/phoenics/issues/new/choose) and describe your problem in detail so we can fix it.

### References

[1] Häse, F., Roch, L. M., Kreisbeck, C., & Aspuru-Guzik, A. [Phoenics: A Bayesian Optimizer for Chemistry.](https://pubs.acs.org/doi/abs/10.1021/acscentsci.8b00307) ACS central science 4.6 (2018): 1134-1145.

[2] Häse, F., Roch, L. M., & Aspuru-Guzik, A. [Chimera: enabling hierarchy based multi-objective optimization for self-driving laboratories.](https://pubs.rsc.org/en/content/articlehtml/2018/sc/c8sc02239a) Chemical Science (2018).
