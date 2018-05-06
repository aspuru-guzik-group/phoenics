# Auto-calibration of a virtual robotic sampling sequence for direct-injection

This example illustrates the applicability of Chimera and Phoenics to determining optimal conditions for a robotic system. We consider the problem of auto-calibrating a virtual robotic sampling sequence introduced in Ref. [1]. The response of the robotic systems is emulated by a Bayesian neural network trained on experiments executed with uniformly sampled experimental conditions. Details on the training procedure of the Bayesian neural network as well as the optimization procedure are provided in Ref. [2]. 

The procedure outlined in this example aims to (i) maximize the response of a chemical characterization procedure via high-performance liquid chromatography (HPLC), (ii) to minimize the amount of sample used in the procedure, and (iii) to minimize the execution time of the experiment. As such, the optimization procedure is implemented as a multi-objective optimization problem using both, Phoenics and Chimera

## Finding optimal experimental conditions

The general framework of the optimization procedure is implemented in `run_optimization_calibration.py`. At each optimization iteration, a new set of experimental conditions is proposed by Phoenics accounting for prevously observed experimental responses. The three objectives (HPLC response, amount of sample, and execution time) are cast into a single objective using Chimera. The newly suggested experimental conditions are then passed to the emulator of the experimental procedure to query an experimental response for this set of conditions. 

## Emulating the experimental response




### References
[1] Roch, L. M., Häse, F., Kreisbeck, C., Tamayo-Mendoza, T., Yunker, L. P. E., Hein, J. E. & Aspuru-Guzik, A. (2018). ChemOS: An Orchestration Software to Demoncratize Autonomous Discovery. chemRxiv preprint [10.26434/chemrxiv.6195176](https://chemrxiv.org/articles/ChemOS_An_Orchestration_Software_to_Democratize_Autonomous_Discovery/5953606).

[2] Häse, F., Roch, L. M., & Aspuru-Guzik, A. (2018). Chimera: enabling hierarchy based multi-objective optimization for self-driving laboratories. chemRxiv preprint [10.26434/chemrxiv.6195176.v1](https://chemrxiv.org/articles/Chimera_Enabling_Hierarchy_Based_Multi-Objective_Optimization_for_Self-Driving_Laboratories/6195176).


