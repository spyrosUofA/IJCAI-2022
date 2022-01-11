import itertools
import numpy as np
import copy


def create_interval(value, delta):
    interval = (value - delta, value + delta)
    return interval


class Node:
    def __init__(self):
        self.size = 0

        # NEW >>
        self.number_children = 0
        self.current_child = 0
    
    def getSize(self):
        return self.size
    
    def toString(self):
        raise Exception('Unimplemented method: toString')
    
    def interpret(self):
        raise Exception('Unimplemented method: interpret')
    
    def grow(self, plist, new_plist):
        pass

    # NEW >>>
    def get_number_children(self):
        return self.number_children

    # <<< NEW

    @classmethod
    def name(cls):
        return cls.__name__

    def get_Num_range(self):
        dict_ranges = {}
        originals = []
        i = 1
        j = 1
        # BFS
        q = []
        q.append(self)
        while len(q) > 0:
            node = q.pop(0)
            if type(node) is Num:
                name = "Num"+str(i)
                i+=1
                originals.append(node.value)
                interval = (-1., 1.) #create_interval(node.value, 2.5) #
                dict_ranges[name] = copy.deepcopy(interval)
            elif type(node) is Ite:
                q.append(node.condition)
                q.append(node.true_case)
                q.append(node.false_case)
            elif type(node) is Lt:
                q.append(node.left)
                q.append(node.right)
            #elif type(node) is AssignAction:
            #    q.append(node.value)
            elif type(node) is Addition:
                q.append(node.left)
                q.append(node.right)
            elif type(node) is Linear:
                for k, w_k in enumerate(node.weight):
                    name = "w" + str(j) + '.' + str(k)
                    #print(w_k)
                    originals.append(w_k)
                    interval = (-1., 1.)  #create_interval(w_k, 2) #
                    dict_ranges[name] = copy.deepcopy(interval)
                j += 1
        return dict_ranges, originals

    def set_Num_value(self, values):
        # BFS to traverse tree, whenever we find Num node we set the value of the node according to [name].
        q = []
        i = 1
        j = 1
        q.append(self)
        while len(q) > 0:
            node = q.pop(0)
            if type(node) is Num:
                name = "Num"+str(i)
                if type(values) is not list:
                    node.value = values[name]
                    i += 1
                else:
                    node.value = values.pop(0)
            elif type(node) is Ite:
                q.append(node.condition)
                q.append(node.true_case)
                q.append(node.false_case)
            elif type(node) is Lt:
                q.append(node.left)
                q.append(node.right)
            #elif type(node) is AssignAction:
            #    q.append(node.value)
            elif type(node) is Addition:
                q.append(node.left)
                q.append(node.right)
            elif type(node) is Linear:
                #print("val", values)
                w = []
                for k in range(node.obs_size):
                    name = "w" + str(j) + '.' + str(k)
                    node.weight[k] = values[name]
                    #print("w_i", values[name])
                    #w.append(values[name])
                #node.weights = w
                #node = Linear(w)
                #print(node.toString())
                #print("w", w)
                j += 1
        return


class Lt(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        
        self.size = left.getSize() + right.getSize() + 1        

    def setSize(self, size):
        self.size = size

    def toString(self):
        return "(" + self.left.toString() + " < " + self.right.toString() + ")"
    
    def interpret(self, env):
        return self.left.interpret(env) < self.right.interpret(env)

    def __eq__(self, other):
        if type(other) != Lt:
            return False
        if self.left == other.left and self.right == other.right:
            return True
        return False
    
    def grow(plist, size):
        new_programs = []
        # defines which nodes are accepted in the AST
        accepted_nodes = set([Num.name(), ReLU.name(), Linear.name(), Observation.name(), Addition.name()])
        
        # generates all combinations of cost of size 2 varying from 1 to size - 1
        combinations = list(itertools.product(range(1, size - 1), repeat=2))

        for c in combinations:
            # skip if the cost combination exceeds the limit
            if c[0] + c[1] + 1 != size:
                continue
            
            # retrive bank of programs with costs c[0] and c[1]
            program_set1 = copy.deepcopy(plist.get_programs(c[0]))
            program_set2 = copy.deepcopy(plist.get_programs(c[1])) #plist.get_programs(c[1])
            
            for t1, programs1 in program_set1.items():                
                # skip if t1 isn't a node accepted by Lt
                if t1 not in accepted_nodes:
                    continue
                
                for p1 in programs1:
                    for t2, programs2 in program_set2.items():
                        # skip if t1 isn't a node accepted by Lt
                        if t2 not in accepted_nodes:
                            continue

                        # Boolean conditions have to have at least one observation
                        #if not (t1 == Observation.name() or t2 == Observation.name()):
                        #    continue

                        # Boolean conditions should not be Num1 < Num2
                        if t1 == Num.name() and t2 == Num.name():
                            continue

                        # Left and Right shoudn't be the same
                        if t1 == t2:
                            continue

                        # Symmetry
                        if t1 == Num.name():
                            continue
                        
                        # p1 and all programs in programs2 satisfy constraints; grow the list
                        for p2 in programs2:
                            lt = Lt(p1, p2)
                            new_programs.append(lt)
                            
                            yield lt
        return new_programs 


class Ite(Node):
    def __init__(self, condition, true_case, false_case):
        self.condition = condition
        self.true_case = true_case
        self.false_case = false_case
        
        self.size = condition.getSize() + true_case.getSize() + false_case.getSize() + 1
        
    def toString(self):
        return "(if" + self.condition.toString() + " then " + self.true_case.toString() + " else " + self.false_case.toString() + ")"
    
    def interpret(self, env):
        if self.condition.interpret(env):
            return self.true_case.interpret(env)
        else:
            return self.false_case.interpret(env)

    def __eq__(self, other):
        if type(other) != Ite:
            return False
        if self.condition == other.condition and self.true_case == other.true_case and self.false_case == other.false_case:
            return True
        return False

    def getBooleans(self):
        # BFS
        q = [self]
        bools = []

        while len(q) > 0:
            node = q.pop(0)
            if type(node) is Ite:
                # Always a Lt expression
                bools.append(node.condition)

                # Add Lt expressions to boolean list
                if type(node.true_case) is Lt:
                    bools.append(node.true_case)
                if type(node.false_case) is Lt:
                    bools.append(node.false_case)

                # If true/false condition is Ite, add to q and keep traversing
                if type(node.true_case) is Ite:
                    q.append(node.true_case)
                if type(node.false_case) is Ite:
                    q.append(node.false_case)

            # Add Lt expressions to boolean list
            elif type(node) is Lt:
                bools.append(node)
        return bools

    def getReLUs(self):
        # BFS
        q = [self]
        bools = []

        while len(q) > 0:
            node = q.pop(0)
            if type(node) is Ite:
                # Always a Lt expression
                bools.append(node.condition)

                # Add Lt expressions to boolean list
                if type(node.true_case) is Lt:
                    bools.append(node.true_case)
                if type(node.false_case) is Lt:
                    bools.append(node.false_case)

                # If true/false condition is Ite, add to q and keep traversing
                if type(node.true_case) is Ite:
                    q.append(node.true_case)
                if type(node.false_case) is Ite:
                    q.append(node.false_case)

            # Add Lt expressions to boolean list
            elif type(node) is Lt:
                bools.append(node)
        return bools

    def grow(plist, size):
        new_programs = []
        # defines the set of nodes accepted as conditions for an Ite
        accepted_condition_nodes = set([Lt.name()])
        # defines the set of nodes accepted as cases for an Ite
        accepted_case_nodes = set([AssignAction.name(), Ite.name()])

        combinations = list(itertools.product(range(1, size - 1), repeat=3))
        
        for c in combinations:
            
            if c[0] + c[1] + c[2] + 1 != size:
                continue
            
            # retrive bank of programs with costs c[0] and c[1]
            program_set1 = copy.deepcopy(plist.get_programs(c[0])) #plist.get_programs(c[0])
            program_set2 = copy.deepcopy(plist.get_programs(c[1])) #plist.get_programs(c[1])
            program_set3 = copy.deepcopy(plist.get_programs(c[2])) #plist.get_programs(c[2])
            
            for t1, programs1 in program_set1.items():                
                # skip if t1 isn't a node accepted as a condition node for Ite
                if t1 not in accepted_condition_nodes:
                    continue
                
                for p1 in programs1:
                    for t2, programs2 in program_set2.items():
                        # skip if t2 isn't a case node
                        if t2 not in accepted_case_nodes:
                            continue
                        
                        # p1 and all programs in programs2 satisfy constraints; grow the list
                        for p2 in programs2:
                            for t3, programs3 in program_set3.items():
                            # skip if t3 isn't a case node
                                if t3 not in accepted_case_nodes:
                                    continue
            
                                # produces a new program with Ite, p1, p2, and p3
                                for p3 in programs3:
                                    if p2 == p3:
                                        continue

                                    # Remove duplicate actions and consider symmetry
                                    if t2 == AssignAction.name() and t3 == AssignAction.name():
                                        if p2.value.value >= p3.value.value:
                                            continue
                                    ite = Ite(p1, p2, p3)
                                    new_programs.append(ite)
                                    
                                    yield ite
        return new_programs


class Num(Node):
    def __init__(self, value):
        self.value = value
        self.size = 1
        
    def toString(self):
        return str(self.value)
    
    def interpret(self, env):
        return self.value

    def __eq__(self, other):
        if type(other) != Num:
            return False
        if self.value == other.value:
            return True
        return False


class AssignActionDiscrete(Node):
    def __init__(self, value):
        self.value = value
        self.size = 1
        
    def toString(self):
        return 'act = ' + str(self.value)
    
    def interpret(self, env):
        env['act'] = self.value

    def __eq__(self, other):
        if type(other) != AssignActionDiscrete:
            return False
        if self.value == other.value:
            return True
        return False


class AssignAction(Node):
    def __init__(self, value):
        self.value = value
        self.size = 1 # self.value.getSize()

    def toString(self):
        return 'act = ' + str(self.value.toString())

    def interpret(self, env):
        env['act'] = self.value.interpret(env)

    def __eq__(self, other):
        if type(other) != AssignAction:
            return False
        if self.value == other.value:
            return True
        return False

    def grow(plist, size):
        #print("grow action")

        new_programs = []
        # defines which nodes are accepted in the AST
        accepted_nodes = set([Num.name()]) #, Observation.name(), Addition.name(), Multiplication.name()])

        # generates all combinations of cost of size 2 varying from 1 to size - 1
        combinations = list(range(1, size))  # [1], [1, 2], ..., [1, 2, 3, 4, 5, ..., 15]

        for c in combinations:
            # skip if the cost combination exceeds the limit
            if c + 1 != size:
                continue

            # retrive bank of programs with costs c[0] and c[1]
            program_set1 = copy.deepcopy(plist.get_programs(c[c])) #plist.get_programs(c)

            #print("p1", program_set1)

            # need this loop
            for t1, programs1 in program_set1.items():
                # skip if t1 isn't a node accepted by AA
                if t1 not in accepted_nodes:
                    continue

                # need this loop
                for p1 in programs1:
                    aa = AssignAction(p1)
                    new_programs.append(aa)

                    yield aa

        return new_programs


class Observation(Node):
    def __init__(self, index):
        self.index = index
        self.size = 1
        
    def toString(self):
        return 'obs[' + str(self.index) + ']'
    
    def interpret(self, env):
        return env['obs'][self.index]

    def __eq__(self, other):
        if type(other) != Observation:
            return False
        if self.index == other.index:
            return True
        return False


class ReLU(Node):
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

        self.size = 1

    def toString(self):
        return 'max(0, ' + str(self.weight) + " * obs + " + str(self.bias) + ")"

    def interpret(self, env):
        return max(0.0, np.dot(self.weight, env['obs']) + self.bias)

    def __eq__(self, other):
        if type(other) != ReLU:
            return False
        if (self.weight == other.weight).all() and (self.bias == other.bias):
            return True
        return False


class Linear(Node):
    def __init__(self, weight):
        self.weight = weight
        self.size = 1
        self.obs_size = len(weight)

    def toString(self):
        return '( ' + str(self.weight) + " *dot* obs[:] )"

    def interpret(self, env):
        return np.dot(self.weight, env['obs'])

    def __eq__(self, other):
        if type(other) != ReLU:
            return False
        if (self.weight == other.weight).all():
            return True
        return False

class Addition(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

        self.size = left.getSize() + right.getSize() + 1

    def toString(self):
        return "(" + self.left.toString() + " + " + self.right.toString() + ")"

    def interpret(self, env):
        return self.left.interpret(env) + self.right.interpret(env)

    def __eq__(self, other):
        if type(other) != Addition:
            return False
        if self.left == other.left and self.right == other.right:
            return True
        if self.left == other.right and self.right == other.left:
            return True
        return False

    def grow(plist, size):
        new_programs = []
        # defines which nodes are accepted in the AST
        accepted_nodes = set([Num.name(), Observation.name(), Multiplication.name(), ReLU.name()])

        # generates all combinations of cost of size 2 varying from 1 to size - 1
        combinations = list(itertools.product(range(1, size - 1), repeat=2))

        for c in combinations:
            # skip if the cost combination exceeds the limit
            if c[0] + c[1] + 1 != size:
                continue

            # retrive bank of programs with costs c[0] and c[1]
            program_set1 = copy.deepcopy(plist.get_programs(c[0])) #plist.get_programs(c[0])
            program_set2 = copy.deepcopy(plist.get_programs(c[1])) #plist.get_programs(c[1])

            for t1, programs1 in program_set1.items():
                # skip if t1 isn't a node accepted by Lt
                if t1 not in accepted_nodes:
                    continue

                for p1 in programs1:
                    for t2, programs2 in program_set2.items():
                        # skip if t1 isn't a node accepted
                        if t2 not in accepted_nodes:
                            continue

                        if (t1 == Num.name() and t2 == Num.name()):
                            continue

                        # p1 and all programs in programs2 satisfy constraints; grow the list
                        for p2 in programs2:
                            st = Addition(p1, p2)
                            new_programs.append(st)

                            yield st
        return new_programs


class Multiplication(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

        self.size = left.getSize() + right.getSize() + 1

    def toString(self):
        return "(" + self.left.toString() + " * " + self.right.toString() + ")"

    def interpret(self, env):
        return self.left.interpret(env) * self.right.interpret(env)

    def __eq__(self, other):
        if type(other) != Multiplication:
            return False
        if self.left == other.left and self.right == other.right:
            return True
        if self.left == other.right and self.right == other.left:
            return True
        return False

    def grow(plist, size):
        new_programs = []
        # defines which nodes are accepted in the AST
        accepted_nodes = set([Num.name(), Observation.name(), Addition.name(), ReLU.name()])

        # generates all combinations of cost of size 2 varying from 1 to size - 1
        combinations = list(itertools.product(range(1, size - 1), repeat=2))

        for c in combinations:
            # skip if the cost combination exceeds the limit
            if c[0] + c[1] + 1 != size:
                continue

            # retrive bank of programs with costs c[0] and c[1]
            program_set1 = copy.deepcopy(plist.get_programs(c[0])) #plist.get_programs(c[0])
            program_set2 = copy.deepcopy(plist.get_programs(c[1])) #plist.get_programs(c[1])

            for t1, programs1 in program_set1.items():
                # skip if t1 isn't a node accepted by Lt
                if t1 not in accepted_nodes:
                    continue

                for p1 in programs1:
                    for t2, programs2 in program_set2.items():
                        # skip if t1 isn't a node accepted
                        if t2 not in accepted_nodes:
                            continue

                        #if (t1 == Observation.name() and t2 == Observation.name()):
                        #    continue

                        # p1 and all programs in programs2 satisfy constraints; grow the list
                        for p2 in programs2:
                            mp = Multiplication(p1, p2)
                            new_programs.append(mp)

                            yield mp
        return new_programs
