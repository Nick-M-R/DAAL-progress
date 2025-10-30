# import time
import numpy as np
from IPython.display import display
import pandas as pd
import traceback

import matplotlib.pyplot as plt
import matplotlib as mpl
# plt.style.use("https://raw.githubusercontent.com/cnativid/MPL-Styles/main/default.mplstyle")

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
# from pymoo.operators.crossover.expx import ExponentialCrossover
from pymoo.operators.mutation.pm import PM
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem as pymoo_Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem
from pymoo.config import Config
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV

from pysamoo.algorithms.gpsaf import GPSAF
from pysamoo.core.surrogate import Surrogate
from ezmodel.models.kriging import Kriging
from pysamoo.core.target import Target
from pysamoo.core.algorithm import MyNormalization
from pymoo.core.population import Population


from dask.distributed import Client, LocalCluster, as_completed
# from dask.distributed.protocol import serialize
# from dask.distributed import progress

# from shapely.geometry import Polygon

import os
from os import listdir

import pickle

from rich.progress import Progress, MofNCompleteColumn, TextColumn, TimeRemainingColumn

np.random.seed(1)
pd.set_option('display.max_columns', None)
Config.warnings['not_compiled'] = False

class Problem:
    def __init__(self, 
                ## PARAMETERIZATION ##
                variables = None,
                init_pop = None,
                
                ## OBJECTIVE ##
                evaluator = None,
                objectives = None,

                ## SUBJECT TO ##
                constraints = [],
                
                ## USE THE METHODS ##
                genetic_algorithm = None,
                crossover = SBX(prob=1, eta=2),
                surrogate_assisted = False,
                
                ## OPTIONS ##
                pop_size=200, n_offsprings=200, 
                n_gen_max = 40, n_gen_restart = 5, 
                repair_invalid = False,
                hv_ref = None, remove_dominated = False,
                verbose = 1, cluster = None):
        
        """ MOOFOIL 
        verbose : = 0, minimal outputs
                  = 1, basic monitoring
                  = 2, debugging
        """
        
        
        self.variables = variables
        self.n_var = 0
        self.varsplit = [0]
        self.lower_bound = []
        self.upper_bound = []
        
        for key in variables.keys():
            variable = variables[key]
            var_n_var = variable.n_var
            self.n_var += var_n_var
            self.varsplit += [self.n_var]
            self.lower_bound += [np.ones(var_n_var) * variable.lower_bound]
            self.upper_bound += [np.ones(var_n_var) * variable.upper_bound]

        self.lower_bound = np.concatenate(self.lower_bound)
        self.upper_bound = np.concatenate(self.upper_bound)
        
        self.evaluator = evaluator
        
        self.objectives = objectives
        self.n_obj = sum([len(objective.cases) for objective in objectives])
        
        self.constraints = constraints
        self.n_ieq_constr = sum([len(constraint.cases) for constraint in constraints])
        
        self.pop_size = pop_size
        self.n_offsprings = n_offsprings
        
        self.crossover = crossover

        self.init_pop = init_pop
        
        self.hv_ref = hv_ref
        self.remove_dominated = remove_dominated
        
        print("INITPOP",len(self.init_pop))
        
        self.n_gen_max = n_gen_max
        self.repair_invalid = repair_invalid
        self.n_gen_restart = n_gen_restart
        self.verbose = verbose
        if cluster:
            self.cluster = cluster
        else:
            self.cluster = LocalCluster()
        
        self.Evaluator = Evaluator()
        
        self.surrogate_assisted = surrogate_assisted
                    
    def run(self, restart = None, workers = None):
        mpl.use('agg')
        ## INITIALIZE ##
        evaluator = self.evaluator
        objectives = self.objectives
        constraints = self.constraints
        n_obj = self.n_obj
        n_ieq_constr = self.n_ieq_constr
        hv_ref = self.hv_ref
        remove_dominated = self.remove_dominated
        
        # start the client
        client = Client(self.cluster)
        print(client.dashboard_link)
        
        # create all required folders
        try:
            os.system("mkdir gen/_current -p")
            os.system("mkdir gen/_dominated -p")
        except:
            pass

        problem = pymoo_Problem(
            n_var=self.n_var, n_obj=self.n_obj, n_constr=self.n_ieq_constr, 
            xl=self.lower_bound, xu=self.upper_bound
        )
        
        if restart:
            algorithm = pickle.load(open(restart, 'rb'))
        else:
            algorithm = NSGA2(
                pop_size=self.pop_size, n_offsprings = self.n_offsprings,
                sampling = self.init_pop, 
                crossover = self.crossover,
                mutation= PM(prob=1,eta=20),
                eliminate_duplicates=True, termination=NoTermination(),
                save_history = True
            )
            
            if self.surrogate_assisted:
                xl, xu = problem.bounds()

                targets = []

                models = dict(
                    default_kriging=Kriging(regr="linear", ARD=True, norm_X=MyNormalization(xl, xu))
                )
                for m in range(problem.n_obj):
                    target = Target(("F", m), models)
                    targets.append(target)

                for g in range(problem.n_ieq_constr):
                    target = Target(("G", g), models)
                    targets.append(target)

                for h in range(problem.n_eq_constr):
                    target = Target(("H", h), models)
                    targets.append(target)

                # create the surrogate model
                surrogate = Surrogate(problem, targets)
                
                algorithm = GPSAF(algorithm,
                    surrogate = surrogate,
                    sampling = self.init_pop, 
                    alpha=10,
                    beta=50,
                    n_max_doe=100,
                    n_max_infills=np.inf,
                    eliminate_duplicates=True,
                    )

            algorithm.setup(problem)
            algorithm.n_gen = 1 

        # define eval function
        def eval(individual):
            cpu_path = f"./run/{individual.name}"
            os.system(f"mkdir {cpu_path} -p")
            
            try:
                result, individual = evaluator(individual, cpu_path)
                f = np.concatenate([objective.eval(result) for objective in objectives])
                g = np.concatenate([constraint.eval(result) for constraint in constraints])
                return f, g, individual, 1
            except Exception:
                # print(e)
                os.system(f"mv {cpu_path} ./run/_{individual.name}")
                os.system(f'echo "{individual.name:<7} | {traceback.format_exc()}" >> logs/out.err')
                os.system(f'echo "{individual.name:<7} | {traceback.format_exc()}" >> ./run/_{individual.name}/out.err')
                return np.inf*np.ones(n_obj), np.inf*np.ones(n_ieq_constr), individual, 0

        ## EVALUATE ##
        try: # allow for CTRL C event
            while algorithm.n_gen < self.n_gen_max + 2:
                # clear any residual files in run
                os.system("rm -rf ./run/*")
                n_gen = algorithm.n_gen - 1
                
                # ask the algorithm for the next solution to be evaluated
                offspring = algorithm.ask()
                                                 
                # name the offspring for identification, save variables
                for (e, individual) in enumerate(offspring):
                    individual.name = f"g{n_gen}-e{e}"
                    individual.var = {}
                    for i, key in enumerate(self.variables.keys()):
                        individual.var |= {key:individual.x[self.varsplit[i]:self.varsplit[i+1]]}
                
                # submit evaluations to client
                futures = [client.submit(eval, individual) for individual in offspring]
                
                # track it for logging
                with Progress(TextColumn("DAAL MOOFOIL | {task.description}"),MofNCompleteColumn(),TimeRemainingColumn(elapsed_when_finished = True)) as progress:
                    task = progress.add_task(f"[red]Generation {n_gen:g}",total = len(futures))
                    for completed in as_completed(futures):
                        progress.update(task,advance = 1)
                
                # receive the solutions
                FG = [future.result() for future in futures]
                F = np.row_stack([fg[0] for fg in FG])
                G = np.row_stack([fg[1] for fg in FG])
                
                # update individuals
                for i, _ in enumerate(offspring):
                    offspring[i] = FG[i][2]
                
                # only pass valid offspring
                invalid_offspring = []
                for i, _ in enumerate([fg[2] for fg in FG]):
                    if FG[i][-1] == 0:
                        invalid_offspring.append(i)
                offspring = np.delete(offspring, invalid_offspring, axis = 0)
                F = np.delete(F, invalid_offspring, axis = 0)
                G = np.delete(G, invalid_offspring, axis = 0)
                
                n_successful = sum([fg[3] for fg in FG])
                print(f"{n_successful}/{len(FG)} successful evaluations")
                
                # Submit results to GA
                static = StaticProblem(problem, F=F, G=G)
                self.Evaluator.eval(static, offspring)
                algorithm.tell(infills=offspring)
                
                ## POST EVAL ##
                try:
                    dr_knee = np.amin(np.linalg.norm(algorithm.opt.get("F"), axis = 1)) - r_knee
                except:
                    dr_knee = 0
                    
                r_knee = np.amin(np.linalg.norm(algorithm.opt.get("F"), axis = 1))
                
                gen_dict = dict(
                    Generation = n_gen,
                    n_eval = self.Evaluator.n_eval,
                    n_nds = len(np.unique(algorithm.opt.get("name"))),
                    r_knee = r_knee,
                    dr_knee = dr_knee,
                )
                
                min_F = np.min(F, axis = 0)
                
                if self.surrogate_assisted:
                    algo = algorithm.algorithm
                    perf = algorithm.surrogate.performance("mae")
                    gen_dict.update(
                        dict(
                            norm_mae_f1 = perf[("F", 0)]/min_F[0],
                            norm_mae_f2 = perf[("F", 1)]/min_F[1],
                        )
                    )
                else:
                    algo = algorithm

                for i, min_f in enumerate(min_F):
                    gen_dict.update(
                        {f"min(obj_{i})" : min_f}
                         )
                    
                min_G = np.min(G, axis = 0)
                for i, min_g in enumerate(min_G):
                    gen_dict.update(
                        {f"min(constr_{i})" : min_g}
                         )
                
                
                # # do not ask me about this algorithm... (moves files in an out of the non-dominated set)
                # if any(algo.opt): # does a current pareto exist? 
                #     # if it does exist, check its difference with the last pareto
                #     if len(algo.history) > 1: # see if a previous pareto even exists
                        
                #         current_opt = algo.history[-1].opt.get("name")
                #         prev_opt = algo.history[-2].opt.get("name")
                        
                #         to_add = set(current_opt) - set(prev_opt)
                #         to_remove = set(prev_opt) - set(current_opt)

                #         assert len(to_add) - len(to_remove) == len(algo.history[-1].opt) - len(algo.history[-2].opt)
                #     else:
                #         to_add = algo.history[-1].opt.get("name")
                #         to_remove = []
                #         assert len(to_add) == len(algo.history[-1].opt)
                    
                #     if any(to_add):
                #         os.system(f"mv {" ".join("./run/"+i for i in to_add)} ./gen/_current")
                #     if any(to_remove):
                #         os.system(f"mv {" ".join("./gen/_current/"+i for i in to_remove)} ./gen/_dominated")
                
                if len(algo.history) > 1: # only need to check diffs if on gen 1 (2 gen eval already)
                    # first get all previous offspring
                    prev_all_offspring = Population.create()
                    for generation in algo.history[:-1]:
                        prev_all_offspring = generation.off.merge(prev_all_offspring, generation.off)
    
                    # then get all existing offspring
                    curr_all_offspring = generation.off.merge(prev_all_offspring, algo.history[-1].off)
                    
                    # extract pareto soln from both
                    prev_nds = self.extract_pareto(prev_all_offspring)
                    curr_nds = self.extract_pareto(curr_all_offspring)

                    # do the set math to find which airfoils needs to be moved                    
                    prev_nds_names = prev_nds.get("name")
                    curr_nds_names = curr_nds.get("name")
                    
                    to_add = set(curr_nds_names) - set(prev_nds_names)
                    to_remove = set(prev_nds_names) - set(curr_nds_names)

                    assert len(to_add) - len(to_remove) == len(curr_nds) - len(prev_nds)
                else:
                    curr_nds = self.extract_pareto(algo.history[-1].off)
                    to_add = curr_nds.get("name")
                    to_remove = []
                    assert len(to_add) == len(curr_nds)
                
                if any(to_add):
                    os.system(f"mv {" ".join("./run/"+i for i in to_add)} ./gen/_current")
                if any(to_remove):
                    if remove_dominated:
                        os.system(f"rm -rf {" ".join("./gen/_current/"+i for i in to_remove)}")
                    else:
                        os.system(f"mv {" ".join("./gen/_current/"+i for i in to_remove)} ./gen/_dominated")
                    
                
                # if there are no solutions, don't do anything
                
                # compute convergence metrics
                gen_dict["hv"] = HV(ref_point = hv_ref)(curr_nds.get("F"))
                gen_dict["n_pareto"] = len(curr_nds)
                
                # callback
                gen_df = pd.DataFrame([gen_dict])
                display(gen_df)

                ## SAVE TO RESTART FILE ##
                pickle.dump(algo.history, open("./gen/_current/history.pkl", 'wb'))
                pickle.dump(algo, open("./gen/_current/algorithm.pkl", 'wb'))
                
                if n_gen%self.n_gen_restart == 0 and n_gen > 0:
                    # save history
                    os.system(f"mkdir -p ./gen/{n_gen:g}")
                    # pickle.dump(algorithm.history, open(f"./gen/{n_gen:g}/history.pkl", 'wb'))
                    
                    # save data files
                    try:
                        os.system(f"mkdir -p ./gen/{n_gen:g}/dat")
                        os.system(f"mkdir -p ./gen/{n_gen:g}/png")
                    except:
                        pass
                    
                    # save the algorithm to file for restart
                    # pickle.dump(algorithm, open(f"./gen/{n_gen:g}/pkl.pkl", 'wb'))
                    
                    # save airfoils
                    # client.map(lambda i: i.Airfoil.save(f"./gen/{n_gen:g}/dat"), algorithm.opt)
                    
                    # for individual in algorithm.opt:
                    #     t = np.linspace(1, -1, 301)
                    #     x = (0.5 - 0.5*np.cos(np.pi*t))
                    #     _, z = basis.generate(x, individual.x)
                        
                    #     with open(f"./gen/{n_gen:g}/dat/{individual.name}.dat", "w") as file:
                    #         file.write(individual.name)
                    #         file.writelines([f"\n{x} {z}" for (x, z) in zip(x, z)])
                    #         file.write('\n')

                    # restart the client
                    print("Restarting Client")
                    client.restart(wait_for_workers = True,)
                    
        except KeyboardInterrupt:
            print("Keyboard Interrupt Detected. Exiting...")
            print("Exit successful.")
            
            
        ## POST PROCESSING ##
        client.shutdown()
        # obtain the result objective from the algorithm
        res = algorithm.result()

        # calculate a hash to show that all executions end with the same result
        print(res.F)
        print("hash", res.F.sum())
        print("best", np.min(res.F,axis = 0))
        print("knee", res.F[np.argmin(np.linalg.norm(algorithm.opt.get("F"), axis = 1))], r_knee )

        self.plot_pareto(res.F)
        
        return res
    
    ## UTITLITY ##
    def vprint(self, str):
        if self.verbose >= 1:
            print(str)
            
    def plot_pareto(self, F):
        plt.figure()
        plt.scatter(F[:,0], F[:,1])
        plt.savefig("gen/_current/pareto.png")

    @staticmethod
    def extract_pareto(pop):
        # extract feasible soln
        feasible_idx = np.where(np.all(pop.get("G") <= 0 ,axis = 1))[0]
        feasible_pop = pop[feasible_idx]
        # find the pareto of the feasible pop
        return feasible_pop[NonDominatedSorting().do(feasible_pop.get("F"), only_non_dominated_front = True)]
        
## ISSUES ##
"""

2024-03-27 15:50:57,341 - distributed.utils_perf - WARNING - full garbage collections took 13% CPU time recently (threshold: 10%)
fixed

"""
