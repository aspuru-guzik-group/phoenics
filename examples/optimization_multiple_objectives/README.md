# Multi-objective optimization

Phoenics supports multi-objective optimization via the Chimera scalarizing function. Chimera processes user preferences provided in the form of an importance hierarchy in the objectives, and relative tolerances indicating the accepted degrees of degradation in each of the objectives. 

The configuration of a multi-objective optimization run is very similar to the configuration of a single-objective optimization run. We simply define multiple objectives in the `"objectives"` section of the configuration file. Note, that the importance hierarchy is defined via the `"hierarchy"` flag in the objective definition. The most important objective should receive a `"hierarchy"` value of zero, the second most important a value of one, etc. The order in which objectives are defined does not necessarily have to be identical to their order in the importance hierarchy. 

```
{
	"general": {
    	...
    },
    "variables": [
    	...
    ],
    "objectives": [
    	{"obj_0": {"type": "minimum", "hierarchy": 0, "tolerance": 0.500}},
        {"obj_1": {"type": "maximum", "hierarchy": 1, "tolerance": 0.250}},
        {"obj_2": {"type": "maximum", "hierarchy": 2, "tolerance": 0.125}},
    ]
}
```

An example for setting up a multi-objective optimization run is provided in `"optimize_multi_objective.py"`, which implements an optimization procedure on the fonseca benchmark set consisting of two objectives. 