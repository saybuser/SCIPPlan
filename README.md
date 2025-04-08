# SCIPPlan

SCIPPlan [1,2,3] is a SCIP-based [4] hybrid planner for domains with i) mixed (i.e., real and/or discrete valued) state and action spaces, ii) nonlinear state transitions (which can be specified as ODEs or directly as solution equations) that are functions of time, and iii) general reward functions. SCIPPlan iteratively i) finds violated constraints (i.e., zero-crossings) by simulating the state transitions, and ii) adds the violated (symbolic) constraints back to its underlying optimisation model, until a valid plan is found.

## Example Domain: Navigation

<img src=./visualisation/scipplan_navigation_1.gif width="32%" height="32%"> <img src=./visualisation/scipplan_navigation_2.gif width="32%" height="32%"> <img src=./visualisation/scipplan_navigation_3.gif width="32%" height="32%">


Figure 1: Visualisation of different plans generated by SCIPPlan [1,2,3] for example navigation domains where the red square represents the agent, the blue shapes represent the obstacles, the gold star represents the goal location and the delta represents time. The agent can control its acceleration and the duration of its control input to modify its speed and location in order to navigate in a two-dimensional maze. The purpose of the domain is to find a path for the agent with minimum makespan such that the agent reaches its the goal without colliding with the obstacles. 

Note that SCIPPlan does not linearise or discretise the domain to find a valid plan.

## Dependencies

i) Solver: SCIP (the current implementation uses the python interface to the SCIP solver, i.e., PySCIPOpt [5]). This version of SCIPPlan has only been tested on PySCIOpt>=4.0.0 using earlier an version of pyscipopt may result in unintended behaviour. 

ii) Symbolic Mathematics: SymPy [6].

## Installing and Running SCIPPlan
In order to Install SCIPPlan you need to ensure you have a working version of the SCIP optimisation suite on your system which can be installed from [the SCIP website](https://www.scipopt.org). For more information about SCIP and PySCIPOpt refer to this [installation guide](https://github.com/scipopt/PySCIPOpt/blob/master/INSTALL.md).

After installing SCIP you will be able to install SCIPPlan using
```bash
pip install scipplan
```
Now you will be able to run some of the example domains which include 
- Navigation (3 instances)

To run one of these examples all you need to do is run
```bash
scipplan -D navigation -I 1
```
which will run the 1st instance of the navigation domain using the ODEs as the transition function with the help of SymPy [6]. Similarly, the command
```bash
scipplan -D navigation -I 1 --provide-sols
```
will run the the 1st instance of the navigation domain using the solution equations as the transition function. For more information regarding the available tags and what they mean run `scipplan --help`.

Alternatively you can import scipplan classes to run it using python.
```py
from scipplan.scipplan import SCIPPlan
from scipplan.config import Config
from scipplan.helpers import write_to_csv
```
this will import the only 2 classes and function needed to run SCIPPlan. Then to set the configuration either create an instance of the Config class by setting the params or by retrieving the cli input
```py
# Set params
config = Config(domain="navigation", instance=1)
# Retrieve cli args
config = Config.get_config()
```
after which you are able to solve problem by either using the solve or optimize methods
```py
# The optimize method just optimises the problem for the given horizon
plan = SCIPPlan(config)
plan.optimize()
# Class method which takes input the config, solves the problem 
# with auto incrementing the horizon until a solution is found then 
# returns the plan as well as the time taken to solve the problem
plan, solve_time = SCIPPlan.solve(config)  
```
In order to save the generated constraints for the horizon solved as well as the results, use the following code
```py
write_to_csv("new_constraints", plan.new_constraints, config)
write_to_csv("results", plan.results_table, config)
```

## Citation

If you are using SCIPPlan, please cite the papers [1,2,3] and the underlying SCIP solver [4].

## References
[1] Buser Say and Scott Sanner. [Metric Nonlinear Hybrid Planning with Constraint Generation](http://icaps18.icaps-conference.org/fileadmin/alg/conferences/icaps18/workshops/workshop06/docs/proceedings.pdf#page=23). In PlanSOpt, pages 19-25, 2018.

[2] Buser Say and Scott Sanner. [Metric Hybrid Factored Planning in Nonlinear Domains with Constraint Generation](https://link.springer.com/chapter/10.1007/978-3-030-19212-9_33). In CPAIOR, pages 502-518, 2019.

[3] Buser Say. [Robust Metric Hybrid Planning in Stochastic Nonlinear Domains Using Mathematical Optimization](https://ojs.aaai.org/index.php/ICAPS/article/view/27216). In ICAPS, pages 375-383, 2023.

[4] [SCIP](https://www.scipopt.org/)

[5] [PySCIPOpt](https://github.com/SCIP-Interfaces/PySCIPOpt)

[6] [SymPy](https://www.sympy.org/en/index.html)
