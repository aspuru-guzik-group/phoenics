# Optimization on periodic parameter spaces

Periodic parameter spaces can occur in scenarios, where a parameter describes for example an angle. In this case, evaluating the objective at a particular parameter point x is equivalent to evaluating at another parameter point x+p which is displaced by a multiple of the periodicity p of the parameter. 

For optimizations on periodic parameter spaces we can constrain the parameter space to just one periodic image of the entire space. However, the applied optimization algorithm should then account for the periodic boundary conditions. Phoenics supports optimizations on periodic parameter spaces by computing kernel distributions with periodic distances (see Ref. [1] for details). 

An example for optimization on periodic spaces is provided in `optimize_periodic.py`. 

### Configuration of a periodic parameter space

The periodicity of a parameter in the optimization problem can be indicated by setting the `"periodic"` flag in the definition of the parameter to `"True"`. The range of the periodic parameter must then be set to the period of this parameter. The example below defines two periodic parameters defined on the [0, 1] interval. Note, that you can mix periodic and non-periodic parameters in the configuration file and periodic distances will only be computed with respect to the parameters declared as periodic.

```json
{
    "general": {
        ...
    },
    "variables": [{"params": { "low": 0.0,  "high": 1.0,  "periodic": "True", "type": "float",  "size": 2} }],
    "objectives": [ ... ]
}
```


### References
[1] Häse, F., Roch, L. M., & Aspuru-Guzik, A. (2018). Chimera: enabling hierarchy based multi-objective optimization for self-driving laboratories. chemRxiv preprint [10.26434/chemrxiv.6195176.v1](https://chemrxiv.org/articles/Chimera_Enabling_Hierarchy_Based_Multi-Objective_Optimization_for_Self-Driving_Laboratories/6195176).
