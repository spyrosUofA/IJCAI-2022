from DSL import *

import numpy as np
import random
import time
from os.path import join
import os
import pickle
import copy


def ite_same_same(p, changes):
    if isinstance(p, Ite):
        if isinstance(p.children[1], AssignAction) and isinstance(p.children[2], AssignAction):
            if p.children[1].children[0] == p.children[2].children[0]:
                p = p.children[2]
                changes += 1
    return p, changes


def num_lt_num(p, changes):
    if isinstance(p, Ite):
        child = p.children[0]
        if isinstance(child, Lt):
            if isinstance(child.children[0], Num) and isinstance(child.children[1], Num):
                if child.children[0].children[0] < child.children[1].children[0]:
                    p = p.children[1]
                    changes += 1
                    return p, changes
                else:
                    p = p.children[2]
                    changes += 1
                    return p, changes
    return p, changes


def num_plus_num(p, changes):
    if isinstance(p, Addition):
        if isinstance(p.children[0], Num) and isinstance(p.children[1], Num):
            p = Num.new(p.children[0].children[0] + p.children[1].children[0])
            changes += 1
    return p, changes


def num_times_num(p, changes):
    if isinstance(p, Multiplication):
        if isinstance(p.children[0], Num) and isinstance(p.children[1], Num):
            p = Num.new(p.children[0].children[0] * p.children[1].children[0])
            changes += 1
            return p, changes
    return p, changes


def plus_zero(p, changes):
    if isinstance(p, Addition):
        if isinstance(p.children[0], Num):
            if p.children[0].children[0] == 0:
                p = p.children[1]
                changes += 1
                return p, changes
        elif isinstance(p.children[1], Num):
            if p.children[1].children[0] == 0:
                p = p.children[0]
                changes += 1
                return p, changes
    return p, changes


def times_zero(p, changes):
    if isinstance(p, Multiplication):
        if isinstance(p.children[0], Num):
            if p.children[0].children[0] == 0:
                p = Num.new(0.0)
                changes += 1
                return p, changes
        elif isinstance(p.children[1], Num):
            if p.children[1].children[0] == 0:
                p = Num.new(0.0)
                changes += 1
                return p, changes
    return p, changes


def relu_lt_0(p, changes):
    if isinstance(p, Ite):
        child = p.children[0]
        if isinstance(child, Lt):
            if isinstance(child.children[0], ReLU) and isinstance(child.children[1], Num):
                if child.children[1].children[0] <= 0:
                    p = p.children[2] # false case
                    changes += 1
                    return p, changes
    return p, changes


def negative_lt_relu(p, changes):
    if isinstance(p, Ite):
        child = p.children[0]
        if isinstance(child, Lt):
            if isinstance(child.children[0], Num) and isinstance(child.children[1], ReLU):
                if child.children[0].children[0] < 0:
                    p = p.children[1] # true case
                    changes += 1
                    return p, changes
    return p, changes


def simplify_node(p, changes):
    c0 = changes
    program = copy.deepcopy(p)
    program, changes = ite_same_same(program, changes)
    program, changes = times_zero(program, changes)
    program, changes = plus_zero(program, changes)
    program, changes = num_plus_num(program, changes)
    program, changes = num_times_num(program, changes)
    program, changes = num_lt_num(program, changes)
    program, changes = relu_lt_0(program, changes)
    program, changes = negative_lt_relu(program, changes)
    return program, changes - c0


def simplify_program(p, changes):

    for i in range(p.get_number_children()):
        simplified, change = simplify_node(p.children[i], changes)
        changes += change
        p.replace_child(simplified, i)
        # simplify children
        if isinstance(p.children[i], Node):
            changes = simplify_program(p.children[i], changes)
            #changes += simplify_program(p.children[i], changes)
            #changes += simplify_program(p.children[i], 0)
    return changes


def simplify_program1(p):
    while True:
        changes = simplify_program(p, 0)
        if changes == 0:
            return p

class SimulatedAnnealing():

    def __init__(self, folder_name, file_name, seed):
        ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default=1))

        self.log_folder = 'logs/' + folder_name + '/'
        self.program_folder = 'programs/' + folder_name + '/'
        self.binary_programs = 'binary_programs/' + folder_name + '/'

        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

        if not os.path.exists(self.program_folder):
            os.makedirs(self.program_folder)

        if not os.path.exists(self.binary_programs):
            os.makedirs(self.binary_programs)

        self.log_file = 'sa_cpus-' + str(ncpus) + file_name
        self.program_file = 'sa_cpus-' + str(ncpus) + file_name
        self.binary_program_file = self.binary_programs + 'sa_cpus-' + str(ncpus) + file_name + '.pkl'

        # Set seed
        np.random.seed(seed)
        random.seed(seed)

    def update_log_file(self, id_log, best_reward, best_score, time_start):
        with open(join(self.log_folder + self.log_file), 'a') as results_file:
            results_file.write(("{:d}, {:f}, {:f}, {:d}, {:f} \n".format(id_log,
                                                                         best_reward,
                                                                         best_score,
                                                                         self.eval_function.get_games_played(),
                                                                         time.time() - time_start)))

    def update_program_file(self, id_log, best_reward_program):
        with open(join(self.program_folder + self.program_file), 'a') as results_file:
            results_file.write(("{:d} \n".format(id_log)))
            results_file.write(best_reward_program.to_string())
            results_file.write('\n')

    def update_binary_file(self, best_reward_program):
        with open(self.binary_program_file, 'wb') as file_program:
            pickle.dump(best_reward_program, file_program)

    def mutate_inner_nodes_ast(self, p, index):
        self.processed += 1

        if not isinstance(p, Node):
            return False

        for i in range(p.get_number_children()):

            if index == self.processed:
                # Accepted rules for the i-th child
                types = p.accepted_rules(i)

                # Generate instance of a random accepted rule
                if isinstance(p, AssignAction) or isinstance(p, Observation) or isinstance(p, Num) or isinstance(p, ReLU) or isinstance(p, Linear):
                    child = list(types)[random.randrange(len(types))]
                else:
                    child = Node.factory(list(types)[random.randrange(len(types))])

                # Randomly generate the child
                if isinstance(child, Node):
                    self.fill_random_program(child, 0, 4)

                # Replacing previous child with the randomly generated one
                p.replace_child(child, i)
                return True

            #print(i, " ", index, p.children[i])
            mutated = self.mutate_inner_nodes_ast(p.children[i], index)

            if mutated:

                # Fixing the size of all nodes in the AST along the modified branch 
                modified_size = 1
                for j in range(p.get_number_children()):
                    if isinstance(p.children[j], Node):
                        modified_size += p.children[j].get_size()
                    else:
                        modified_size += 1
                p.set_size(modified_size)

                return True

        return False

    def mutate(self, p):
        try:
            index = random.randrange(p.get_size())
        except:
            index = 0

        # Mutating the root of the AST
        if index == 0:

            if self.use_double_program:
                p = StartSymbol()
            else:
                initial_types = Node.accepted_rules(0)
                p = Node.factory(list(initial_types)[random.randrange(len(initial_types))])
            self.fill_random_program(p, self.initial_depth_ast, self.max_mutation_depth)

            return p

        self.processed = 0
        self.mutate_inner_nodes_ast(p, index)

        return p

    def return_terminal_child(self, p, types):
        terminal_types = []

        for t in types:
            child = p.factory(t)

            if child.get_number_children() == 0 or isinstance(child, Num) or isinstance(child, Observation) or isinstance(
                    child, AssignAction) or isinstance(child, ReLU) or isinstance(child, Linear):
                terminal_types.append(child)

        if len(terminal_types) == 0:
            for t in types:
                child = p.factory(t)

                if child.get_number_children() == 1:
                    terminal_types.append(child)

        if len(terminal_types) > 0:
            return terminal_types[random.randrange(len(terminal_types))]

        return p.factory(list(types)[random.randrange(len(types))])

    def fill_random_program(self, p, depth, max_depth):

        size = p.get_size()

        for i in range(p.get_number_children()):

            if isinstance(p, AssignAction) or isinstance(p, Observation) or isinstance(p, Num) or isinstance(p, ReLU) or isinstance(p, Linear):
                types = p.accepted_rules(i)
                child = list(types)[random.randrange(len(types))]
                p.add_child(child)
                size += 1
            elif depth >= max_depth:
                types = p.accepted_rules(i)
                child = self.return_terminal_child(p, types)
                p.add_child(child)
                child_size = self.fill_random_program(child, depth + 1, max_depth)
                size += child_size
            else:
                types = p.accepted_rules(i)
                child = p.factory(list(types)[random.randrange(len(types))])
                p.add_child(child)
                child_size = self.fill_random_program(child, depth + 1, max_depth)
                size += child_size

        p.set_size(size)
        return size

    def random_program(self):
        if self.use_double_program:
            p = StartSymbol()
        else:
            initial_types = list(Node.accepted_initial_rules()[0])
            p = Node.factory(initial_types[random.randrange(len(initial_types))])

        self.fill_random_program(p, self.initial_depth_ast, self.max_mutation_depth)

        return p

    def accept_function(self, current_score, next_score):
        return np.exp(self.beta * (next_score - current_score) / self.current_temperature)

    def decrease_temperature(self, i):
        #         self.current_temperature = self.initial_temperature * self.alpha ** i
        self.current_temperature = self.initial_temperature / (1.0 + self.alpha * (i))

    def search(self,
           search_type,
           operations,
           numeric_constant_values,
           observation_values,
               affine_values,
               action_values,
           relu_values,
           eval_function,
           use_triage,
           use_double_program,
           initial_temperature,
           alpha,
           beta,
           time_limit,
           winrate_target=None,
           initial_program=None,
           bayes_opt=False):

        # Objective function: score
        # Evaluation criteria: reward
        if search_type == "0":
            self.search_0(operations,
               numeric_constant_values,
               observation_values,
                affine_values,
               action_values,
               relu_values,
               eval_function,
               use_triage,
               use_double_program,
               initial_temperature,
               alpha,
               beta,
               time_limit,
               winrate_target,
               initial_program,
               bayes_opt)

        # Objective function: score
        # Evaluation criteria: score
        elif search_type == "1":
            self.search_1(operations,
                         numeric_constant_values,
                         observation_values,
                          affine_values,
                         action_values,
                         relu_values,
                         eval_function,
                         use_triage,
                         use_double_program,
                         initial_temperature,
                         alpha,
                         beta,
                         time_limit,
                         winrate_target,
                         initial_program,
                         bayes_opt)
        else:
            raise Exception("No version of search algorithm was specified.")

    # Objective function: score
    # Evaluation criteria: reward
    def search_0(self,
               operations,
               numeric_constant_values,
               observation_values,
               affine_values,
               action_values,
               relu_values,
               eval_function,
               use_triage,
               use_double_program,
               initial_temperature,
               alpha,
               beta,
               time_limit,
               winrate_target=None,
               initial_program=None,
               bayes_opt=False):

        time_start = time.time()

        self.winrate_target = winrate_target
        self.use_double_program = use_double_program

        Node.filter_production_rules(operations,
                                     numeric_constant_values,
                                     observation_values,
                                     action_values)

        self.max_mutation_depth = 6
        self.initial_depth_ast = 0
        self.initial_temperature = initial_temperature
        self.alpha = alpha
        self.beta = beta
        self.slack_time = 60

        Num.accepted_types = [set(numeric_constant_values)]
        AssignAction.accepted_types = [set(action_values)]
        Observation.accepted_types = [set(observation_values)]
        #Affine.accepted_types = [[[0,0,0,0], [0,0,0,0]]] #[[[0.0] * (len(observation_values) + 1)]]
        #Affine.accepted_types = [affine_values]
        Linear.accepted_types = [affine_values]



        ReLU.accepted_types = [relu_values]

        self.operations = operations
        self.numeric_constant_values = numeric_constant_values
        self.eval_function = eval_function
        nb_evaluations = self.eval_function.nb_evaluations

        id_log = 1

        # Initialize Program
        if initial_program is not None:
            current_program = copy.deepcopy(initial_program)
        else:
            current_program = self.random_program()
            if bayes_opt:
                self.eval_function.optimize(current_program)
        #current_program = simplify_program1(current_program)
        # Gather data
        #self.eval_function.update_trajectory0(current_program, 1)

        # Evaluate initial program
        best_reward_program = copy.deepcopy(current_program)
        best_reward = self.eval_function.collect_reward(best_reward_program, nb_evaluations)

        # Save
        print(best_reward, self.eval_function.evaluate(current_program), best_reward_program.to_string())
        self.update_log_file(0, best_reward, 0.0, time_start)
        self.update_program_file(0, best_reward_program)
        self.update_binary_file(best_reward_program)

        while time.time() - time_start < time_limit - self.slack_time:

            # RESET SA
            iteration_number = 1
            self.current_temperature = self.initial_temperature

            # re-optimize best policy
            current_program = copy.deepcopy(best_reward_program)
            if bayes_opt:
                self.eval_function.optimize(current_program)
            best_score_program = current_program

            best_score = self.eval_function.evaluate(current_program)
            current_score = best_score

            # START SA
            while self.current_temperature > 1:

                # If time is up, save best program and exit
                if time.time() - time_start > time_limit - self.slack_time:
                    self.update_log_file(id_log, best_reward, best_score, time_start)
                    self.update_program_file(id_log, best_reward_program)
                    #self.update_binary_file(best_reward_program)
                    return best_reward, best_reward_program

                # Mutate current program
                mutation = self.mutate(copy.deepcopy(current_program))

                # optimize and evaluate program
                if bayes_opt:
                    self.eval_function.optimize(mutation)
                #mutation = simplify_program1(mutation)
                next_score = self.eval_function.evaluate(mutation)

                # Improved score?
                if next_score > best_score:
                    best_score = next_score
                    best_score_program = mutation
                    print(best_score, best_score_program.to_string())

                # Accept a new program? Note: if next_score > current_score, we accept.
                prob_accept = min(1, self.accept_function(current_score, next_score))
                prob = random.uniform(0, 1)
                if prob < prob_accept:
                    current_program = mutation
                    current_score = next_score

                iteration_number += 1
                print(iteration_number)
                self.decrease_temperature(iteration_number)
            # END SA

            # evaluate new policy
            current_reward = self.eval_function.collect_reward(best_score_program, nb_evaluations)
            print("Current:", current_reward, best_score, best_score_program.to_string())
            print("Best:", best_reward, self.eval_function.evaluate(best_reward_program), best_reward_program.to_string())
            #print(self.eval_function.collect_reward(best_reward_program, nb_evaluations))

            if current_reward > best_reward:
                best_reward = current_reward
                best_reward_program = best_score_program # simplify_program1(best_score_program)
                # Update Files
                self.update_log_file(id_log, best_reward, best_score, time_start)
                self.update_program_file(id_log, best_reward_program)
                self.update_binary_file(best_reward_program)
                id_log += 1

            # DAgger step to update histories
            self.eval_function.update_trajectory0(best_reward_program, 25)
            print("NB TIMESTEPS:", len(self.eval_function.actions))

        return best_reward, best_reward_program

    # Objective function: score
    # Evaluation criteria: score
    def search_1(self,
               operations,
               numeric_constant_values,
               observation_values,
               affine_values,
               action_values,
               relu_values,
               eval_function,
               use_triage,
               use_double_program,
               initial_temperature,
               alpha,
               beta,
               time_limit,
               winrate_target=None,
               initial_program=None,
               bayes_opt=False):

        time_start = time.time()

        self.winrate_target = winrate_target
        self.use_double_program = use_double_program

        Node.filter_production_rules(operations,
                                     numeric_constant_values,
                                     observation_values,
                                     action_values)

        self.max_mutation_depth = 4
        self.initial_depth_ast = 0
        self.initial_temperature = initial_temperature
        self.alpha = alpha
        self.beta = beta
        self.slack_time = 60

        Num.accepted_types = [set(numeric_constant_values)]
        AssignAction.accepted_types = [set(action_values)]
        Observation.accepted_types = [set(observation_values)]
        ReLU.accepted_types = [relu_values]
        Affine.accepted_types = [[[0.0] * (len(observation_values) +1)]]


        self.operations = operations
        self.numeric_constant_values = numeric_constant_values
        self.eval_function = eval_function

        best_score = self.eval_function.worst_score
        best_program = None

        id_log = 1

        # Initialize current program
        if initial_program is not None:
            current_program = copy.deepcopy(initial_program)
        else:
            current_program = self.random_program()
        # BayesOpt for current program
        if bayes_opt:
            self.eval_function.optimize(current_program)

        while True:
            self.current_temperature = self.initial_temperature

            # BayesOpt for current program
            if bayes_opt:
                self.eval_function.optimize(current_program)

            if use_triage:
                current_score = self.eval_function.eval_triage(current_program, best_score)
            else:
                current_score = self.eval_function.evaluate(current_program)

            iteration_number = 1

            if best_program is None or current_score > best_score:
                best_score = current_score
                best_program = current_program

                self.update_log_file(id_log, 0.0, best_score, time_start)
                self.update_program_file(id_log, best_program)
                self.update_binary_file(best_program)
                id_log += 1

            while self.current_temperature > 1:

                time_end = time.time()

                if time_end - time_start > time_limit - self.slack_time:
                    self.update_log_file(id_log, best_score, best_score, time_start)
                    self.update_program_file(id_log, best_program)
                    self.update_binary_file(best_program)
                    return best_score, best_program

                copy_program = copy.deepcopy(current_program)

                #print('\nCurrent: ')
                #print(current_program.to_string())
                mutation = self.mutate(copy_program)
                #print('Mutated: ')
                #print(mutation.to_string(), '\n')

                # BayesOpt for mutated program
                if bayes_opt:
                    self.eval_function.optimize(mutation)

                if use_triage:
                    next_score, number_matches_played = self.eval_function.eval_triage(mutation, best_score)
                else:
                    next_score = self.eval_function.evaluate(mutation)

                if self.winrate_target is not None and next_score >= self.winrate_target:
                    return next_score, mutation

                # Better program?
                if best_program is None or next_score > best_score:

                    best_score = next_score
                    best_program = mutation

                    self.update_log_file(id_log, best_score, best_score, time_start)
                    self.update_program_file(id_log, best_program)
                    self.update_binary_file(best_program)  # changed from current program
                    id_log += 1

                # Accept new program?
                prob_accept = min(1, self.accept_function(current_score, next_score))
                prob = random.uniform(0, 1)
                if prob < prob_accept:
                    current_program = mutation
                    current_score = next_score

                iteration_number += 1
                self.decrease_temperature(iteration_number)

            if initial_program is not None:
                current_program = copy.deepcopy(initial_program)
            else:
                if best_score == self.eval_function.worst_score:
                    current_program = self.random_program()
                else:
                    current_program = copy.deepcopy(best_program)

            # update DAgger
            best_score = self.eval_function.update_trajectory1(current_program, best_score)

        return best_score, best_program


