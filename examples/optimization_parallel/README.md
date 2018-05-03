# Batch parallel optimization

Phoenics can propose multiple parameter points in one optimization iteration which favor exploration and exploitation of the parameter space to various degrees. This example presents how to use Phoenics for querying multiple parameter sets per optimization iteration. 

The figure below illustrates the sampling behavior of Phoenics when proposing two parameter points in one batch per optimization iteration. Parameter points proposed with bias towards exploration are depicted in red, blue points are proposed with a bias towards exploitation. Phoenics can propose an arbitrary number of points with different sampling strategies. Details on the procedure are provided in the paper ([arXiv:1801.01469](https://arxiv.org/abs/1801.01469))
![alt text](../../images/batch_sampling_strategy.png)

To enable the generation of multiple parameter points in one optimization iteration, we can change two parameter options in the configuration file:

| Parameter              | Description                                                                         |
|:-----------------------|:------------------------------------------------------------------------------------|
| `num_batches`          | Number of parameter batches returned                                                |
| `batch_size`           | Number of parameter sets per batch                                                  |

**Note**: The `"batch_size"` parameter controls the number of parameter points proposed based on *different* sampling strategies, while `"num_batches"` controls the number of parameter batches returned in one iteration, which are generated from the *same* sampling strategies. We recommend to first increase `"batch_size"` to generate more informative parameter sets. Phoenics was successfully tested with `"batch_size"` up to eight.

As Phoenics can propose new parameter points given a set of observations, it does not rely on the complete evaluation of all parameter points suggested in one optimization iteration before another optimization iteration can be started. This provides additional flexibility for the deployment of Phoenics, and allows for different modes of operation presented in the following. 

### Synchronous batch evaluation

In scenarios, where the parallel evaluation of multiple parameter points is possible and the execution time of one evaluation does not depend on the parameters, Phoenics can be operated in synchronous batch evaluation mode. Here, a new optimization iteration is only started after all objectives for all parameter sets proposed in the previous iteration have been obtained. This procedure is outlined in `"opt_synchronous_evaluations.py"` and schematically shown below

```python
observations = []
for num_iter in range(max_iter):
	
    # request new parameters
	params = phoenics.choose(observations = observations)
    
    # evaluate parameters
    new_observations = evaluator(params)
    observations.extend(new_observations)
```

### Asynchronous batch evaluation

In some applications, the execution time of a parameter evaluation can depend on the choice of (some) parameter values. For such scenarios, Phoenics can be operated in an asynchronous batch evaluation mode. Rather than waiting for all parameter points proposed in one batch to be evaluated, new optimization iterations are started as soon as new objective values for a previously submitted set of parameters are received. This procedure is outlined in `"opt_asynchronous_evaluations.py"` and schematically shown below

```python
submitted_evaluations = []
observations          = []
while len(observations) < max_evaluations:

	# check if evaluators are available
    if len(submitted_evaluations) < max_evaluations:
    
    	# get new parameters 
        params = phoenics.choose(observations = observations)
        
        # submit until max evaluators is reached
		submission_index = 0
		while len(submitted_evaluations) < max_evaluations:
        	identifier = generate_submission_identifier()
        	submit_evaluation(param[submission_index], identifier)
            submission_index += 1
            submitted_evaluations.append(identifier)
            
        # check if evaluations have completed
        for identifier in submitted_evaluations:
        	if has_completed(identifier):
            	observation = get_observation(identifier)
                observations.append(observation)
                submitted_evaluations.remove(identifier)
```

### Remarks

The operation modes outlined below are only two examples how Phoenics' batch optimization feature can be applied to various scenarios. The flexibilty of the `"choose"` method allows for other operation modes as well. Note, that Phoenics does not require the evaluation of the proposed parameter points, and also accepts observations for parameter points which have not been suggested by the currently running Phoenics instance. 
