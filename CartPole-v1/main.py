import argparse
import pickle
from SimulatedAnnealing import SimulatedAnnealing
from evaluation import Evaluate, Environment, Imitation, DAgger
import copy
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-seed', action='store', dest='seed', default=0,
                        help='Set seed for reproducible results.')

    parser.add_argument('-search', action='store', dest='search_algorithm', 
                        default='SimulatedAnnealing', 
                        help='Search Algorithm (SimulatedAnnealing, BottomUpSearch, UCT)')
    
    parser.add_argument('-approach', action='store', dest='approach', default='0',
                        help='0: Reward. 1: Score ')

    parser.add_argument('-bound', action='store', dest='bound',
                        help='Bound for Bottom-Up Search')
    
    parser.add_argument('-e', action='store', dest='eval_function',
                        default='Environment',
                        help='Environment, Imitation, DAgger, DAggerQ')

    parser.add_argument('--bo', action='store_true', dest='bayes_opt', default=False,
                        help='Bayesian Optimization toggle (default=False)')

    parser.add_argument('-ip', action='store', dest='init_program', default=None,
                        help='Initial Program (default=None)')

    parser.add_argument('-oracle', action='store', dest='oracle', default=None,
                        help='Oracle directory containing: Policy.pth, ReLUs.pkl, Actions.npy, Observations.npy')

    parser.add_argument('-c', action='store', dest='capacity', default=None,
                        help='Buffer capacity for imitation style learning (default=None)')

    parser.add_argument('--aug_dsl', action='store_true', dest='augment_dsl', default=False,
                        help='Augment DSL using ReLU units provided by oracle?')

    parser.add_argument('-n', action='store', dest='number_games', default=100,
                        help='Number of games played in each evaluation')
    
    parser.add_argument('-time', action='store', dest='time_limit', default=1200,
                        help='Time limit in seconds')
    
    parser.add_argument('-temperature', action='store', dest='initial_temperature', default=50,
                        help='SA\'s initial temperature')
    
    parser.add_argument('-alpha', action='store', dest='alpha', default=1.2,
                        help='SA\'s alpha value')
    
    parser.add_argument('-beta', action='store', dest='beta', default=100,
                        help='SA\'s beta value')

    parser.add_argument('-file_name', action='store', dest='file_name', default="",
                        help='File in which results will be saved')
    
    parser.add_argument('--hole-node', action='store_true', default=False,
                        dest='use_hole_node',
                        help='Allow the use of hole nodes in the production rules of the AST.')
    
    parser.add_argument('--detect-equivalence', action='store_true', default=False,
                        dest='detect_equivalence',
                        help='Detect observational equivalence in Bottom-Up Search.')
    
    parser.add_argument('--triage', action='store_true', default=False,
                        dest='use_triage',
                        help='Use a 3-layer triage for evaluating programs.')

    parser.add_argument('--double-programs', action='store_false', default=True,
                        dest='use_double_programs',
                        help='The program will have two instructions, one for yes-no decisions and another for column decisions')
    
    parser.add_argument('--iterated-best-response', action='store_true', default=False,
                        dest='run_ibr',
                        help='It will run Iterated Best Response')
    
    parser.add_argument('--reuse-tree', action='store_true', default=False,
                    dest='reuse_tree',
                    help='UCT reuses its tree and SA starts with previous solution in between iterations of IBR')

    parameters = parser.parse_args()


    # Casting to integers
    number_games = int(parameters.number_games)
    seed = int(parameters.seed)
    time_limit = int(parameters.time_limit)
    parameters.capacity = None if parameters.capacity is None else int(parameters.capacity)

    # Specify folder and file names
    folder_name = str(parameters.eval_function)[0] + str(int(parameters.bayes_opt)) + str(int(parameters.augment_dsl))\
                   + parameters.approach + ("" if parameters.init_program is None else ("_" + parameters.init_program[0:4]))\
                    + '/' + str(parameters.oracle)

    file_name = '_n-' + str(number_games) + '_c-' + str(parameters.capacity) + '_run-' + str(seed) + parameters.file_name

    # Load from Oracle path
    if parameters.oracle is not None:
        # load model
        model = PPO.load("./Oracle/" + parameters.oracle + '/model')
        # Load ReLUs
        accepted_relus = pickle.load(open("./Oracle/" + parameters.oracle + "/ReLUs.pkl", "rb"))
        # Arguments for evaluation function
        oracle = {"oracle": model, "inputs": [], "actions": [], "capacity": parameters.capacity}
    else:
        oracle = {}
        
    # Constructors
    eval_function = globals()[parameters.eval_function](oracle, number_games, seed)
    algorithm = globals()[parameters.search_algorithm](folder_name, file_name, seed)

    # Initialize inputs
    eval_function.initialize_history(25)


    if isinstance(algorithm, SimulatedAnnealing):
        
        from DSL import Ite, \
                                Lt, \
                                AssignAction, \
                                Addition, \
                                Multiplication, \
                                Observation, \
                                ReLU, \
                                Num,\
                                Gt0, Gt, Lt0
        
        terminals = [AssignAction]
        OPERATIONS = [AssignAction, Ite, Lt, Gt, Num, Observation, Addition, Multiplication]

        # Add ReLU node to DSL and load ReLU programs
        if parameters.augment_dsl:
            print("HELLO")
            OPERATIONS.append(ReLU)
        else:
            accepted_relus = None

        algorithm.search(parameters.approach,
                         OPERATIONS,
                         [-2.0, -1.0, 0.0, 1.0, 2.0],
                         [0, 1, 2, 3],
                         [0, 1],
                         accepted_relus,
                         eval_function,
                         parameters.use_triage,
                         parameters.use_double_programs,
                         float(parameters.initial_temperature),
                         float(parameters.alpha),
                         float(parameters.beta),
                         time_limit,
                         None,
                         parameters.init_program,
                         bool(parameters.bayes_opt))

if __name__ == "__main__":
    main()
