import numpy as np
import copy


def create_interval(value, delta):
    interval = (value - delta, value + delta)
    return interval


class Node:
    def __init__(self):
        self.size = 1  # changed from 0
        self.number_children = 0
        self.current_child = 0
        self.children = []

    def get_Num_range(self):
        dict_ranges = {}
        originals = []
        i = 1
        j = 1
        a = 1
        # BFS
        q = []
        q.append(self)
        while len(q) > 0:
            node = q.pop(0)
            if type(node) is Num:
                name = "Num" + str(i)
                i += 1
                originals.append(node.children[0])
                interval = (-2, 2) #create_interval(node.children[0], 2)
                dict_ranges[name] = copy.deepcopy(interval)
            elif type(node) is Ite:
                q.append(node.children[0])
                q.append(node.children[1])
                q.append(node.children[2])
            elif type(node) is Lt:
                q.append(node.children[0])
                q.append(node.children[1])
            elif type(node) is Gt:
                q.append(node.children[0])
                q.append(node.children[1])
            #elif type(node) is AssignAction:
            #    name = "a" + str(a)
            #    a += 1
            #    originals.append(node.children[0])
            #    interval = create_interval(0, 2)
            #    dict_ranges[name] = copy.deepcopy(interval)
            elif type(node) is Addition:
                q.append(node.children[0])
                q.append(node.children[1])
            elif type(node) is Multiplication:
                q.append(node.children[0])
                q.append(node.children[1])
            elif type(node) is Lt0 or type(node) is Gt0:
                q.append(node.children[0])
            elif type(node) is StartSymbol:
                q.append(node.children[0])
            elif type(node) is Affine:
                #print(type(node), node.children[0])
                for k in range(9):
                    name = "w" + str(j) + '.' + str(k)
                    originals.append(node.children[0][k])
                    interval = (-2.0, 2.0) #create_interval(node.children[0][k], 1)
                    dict_ranges[name] = copy.deepcopy(interval)
                j += 1
            elif type(node) is Linear:
                for k in range(node.size):
                    name = "w" + str(j) + '.' + str(k)
                    originals.append(node.children[0][k])
                    interval = (-2.0, 2.0) #create_interval(node.children[0][k], 1)
                    dict_ranges[name] = copy.deepcopy(interval)
                j += 1

        return dict_ranges, originals

    def set_Num_value(self, values):
        # BFS to traverse tree, whenever we find Num node we set the value of the node according to [name].
        q = []
        i = 1
        j = 1
        a = 1
        q.append(self)
        while len(q) > 0:
            node = q.pop(0)
            if type(node) is Num:
                name = "Num" + str(i)
                i += 1
                if type(values) is not list:
                    node.children[0] = values[name]
                    #print("broke")
                else:
                    node.children[0] = values.pop(0)
            elif type(node) is Ite:
                q.append(node.children[0])
                q.append(node.children[1])
                q.append(node.children[2])
            elif type(node) is Lt:
                q.append(node.children[0])
                q.append(node.children[1])
            elif type(node) is Gt:
                q.append(node.children[0])
                q.append(node.children[1])
            #elif type(node) is AssignAction:
            #    name = "a" + str(a)
            #    a += 1
            #    if type(values) is not list:
            #        node.children[0] = values[name]
            #    else:
            #        node.children[0] = values.pop(0)
            elif type(node) is Addition:
                q.append(node.children[0])
                q.append(node.children[1])
            elif type(node) is Multiplication:
                q.append(node.children[0])
                q.append(node.children[1])
            elif type(node) is Lt0 or type(node) is Gt0:
                q.append(node.children[0])
            elif type(node) is Affine:
                w = []
                for k in range(9):
                    name = "w" + str(j) + '.' + str(k)
                    w.append(values[name])
                node.children[0] = w
                j += 1
            elif type(node) is Linear:
                w = []
                for k in range(8):
                    name = "w" + str(j) + '.' + str(k)
                    w.append(values[name])
                node.children[0] = w
                j += 1
            elif type(node) is StartSymbol:
                q.append(node.children[0])
        return


    def getSize(self):
        return self.size
    
    def to_string(self):
        raise Exception('Unimplemented method: toString')
    
    def interpret(self):
        raise Exception('Unimplemented method: interpret')
    
    def grow(self, plist, new_plist):
        pass

    def add_child(self, child):
        if len(self.children) + 1 > self.number_children:
            raise Exception('Unsupported number of children')

        self.children.append(child)
        self.current_child += 1

        if child is None or not isinstance(child, Node):
            self.size += 1
        else:
            self.size += child.size

    def get_current_child(self):
        return self.current_child

    def get_number_children(self):
        return self.number_children

    def get_size(self):
        return self.size

    def set_size(self, size):
        self.size = size

    def replace_child(self, child, i):

        if len(self.children) < i + 1:
            self.add_child(child)
        else:
            if isinstance(self.children[i], Node):
                self.size -= self.children[i].size
            else:
                self.size -= 1

            if isinstance(child, Node):
                self.size += child.size
            else:
                self.size += 1

            self.children[i] = child

    @classmethod
    def accepted_rules(cls, child):
        return cls.accepted_types[child]

    @classmethod
    def class_name(cls):
        return cls.__name__

    @staticmethod
    def factory(classname):
        if classname not in globals():
            return classname
        return globals()[classname]()

    @classmethod
    def accepted_initial_rules(cls):
        return cls.accepted_types

    @staticmethod
    def filter_production_rules(operations,
                                numeric_constant_values,
                                observation_values,
                                action_values):
        rules = set()
        for op in operations:
            rules.add(op.class_name())

        if len(numeric_constant_values) > 0:
            rules.add(Num.class_name())

        if len(observation_values) > 0:
            rules.add(Observation.class_name())

        rules.add(None)

        list_all_productions = [Node, #StartSymbol,
                                Ite,
                                Lt,
                                Gt,
                                Lt0,
                                Gt0,
                                Addition,
                                Multiplication]

        for op in list_all_productions:

            op_to_remove = []

            for types in op.accepted_types:
                for op in types:
                    if op not in rules:
                        op_to_remove.append(op)

                for op in op_to_remove:
                    if op in types:
                        types.remove(op)

    @staticmethod
    def restore_original_production_rules():
        StartSymbol.accepted_nodes = [Ite.class_name(), AssignAction.class_name()]
        StartSymbol.accepted_types = [StartSymbol.accepted_nodes]

        Multiplication.accepted_nodesL = [Num.class_name(),
                                    Observation.class_name(),
                                    ReLU.class_name(),
                                    Addition.class_name(),
                                    Multiplication.class_name()]
        Multiplication.accepted_nodesR = [Observation.class_name(),
                                          ReLU.class_name(),
                                          Addition.class_name(),
                                          Multiplication.class_name()]
        Multiplication.accepted_types = [Multiplication.accepted_nodesL,  Multiplication.accepted_nodesR]

        Addition.accepted_nodes = [Num.class_name(),
                                    Observation.class_name(),
                                    ReLU.class_name(),
                                    Addition.class_name(),
                                    Multiplication.class_name()]
        #Addition.accepted_types = [Addition.accepted_nodes, Addition.accepted_nodes]
        Addition.accepted_types = [[ReLU.class_name(), Observation.class_name()], [ReLU.class_name(), Observation.class_name()]]

        Subtraction.accepted_nodes = [Num.class_name(),
                                   Observation.class_name(),
                                   ReLU.class_name(),
                                   Addition.class_name(),
                                   Multiplication.class_name()]
        #Subtraction.accepted_types = [Subtraction.accepted_nodes, Subtraction.accepted_nodes]
        Subtraction.accepted_types = [[ReLU.class_name(), Observation.class_name()], [ReLU.class_name(), Observation.class_name()]]

        Ite.accepted_nodes_bool = [Lt.class_name(), Gt.class_name(), Lt0.class_name(), Gt0.class_name()]
        Ite.accepted_nodes_block = [AssignAction.class_name(), Ite.class_name()]
        Ite.accepted_types = [Ite.accepted_nodes_bool, Ite.accepted_nodes_block, Ite.accepted_nodes_block]

        Lt.accepted_nodesL = [ReLU.class_name(), Linear.class_name(), Affine.class_name()]
        Lt.accepted_nodesR = [Num.class_name()]
        Lt.accepted_types = [Lt.accepted_nodesL, Lt.accepted_nodesR]

        Gt.accepted_nodes = [#Num.class_name(),
            #Observation.class_name(),
            ReLU.class_name(),
            Addition.class_name(),
            Multiplication.class_name()]
        Gt.accepted_types = [[Num.class_name(), ReLU.class_name()], Gt.accepted_nodes]

        Gt0.accepted_nodes = [Observation.class_name(),
                                     ReLU.class_name(),
                                     Affine.class_name(),
                                     Addition.class_name(),
                                     Subtraction.class_name(),
                                     Multiplication.class_name()]
        Gt0.accepted_types = [Gt0.accepted_nodes]

        Lt0.accepted_nodes = [Observation.class_name(),
                                     ReLU.class_name(),
                                     Affine.class_name(),
                                     Addition.class_name(),
                                     Subtraction.class_name(),
                                     Multiplication.class_name()]
        Lt0.accepted_types = [Lt0.accepted_nodes]

        Node.accepted_types = [[Ite.class_name(), AssignAction.class_name()]]

    @classmethod
    def name(cls):
        return cls.__name__


class StartSymbol(Node):
    def __init__(self):
        super(StartSymbol, self).__init__()
        self.size = 0
        self.number_children = 1

    @classmethod
    def new(cls, yes_no):
        inst = cls()
        inst.add_child(yes_no)
        return inst

    def to_string(self):
        return self.children[0].to_string()

    def interpret(self, env):
        return self.children[0].interpret(env)


class Num(Node):

    def __init__(self):
        super(Num, self).__init__()
        self.number_children = 1
        self.size = 0

    @classmethod
    def new(cls, var):
        inst = cls()
        inst.add_child(var)

        return inst

    def to_string(self):
        if len(self.children) == 0:
            raise Exception('VarScalar: Incomplete Program')

        return str(self.children[0])

    def interpret(self, env):
        if len(self.children) == 0:
            raise Exception('VarScalar: Incomplete Program')

        return self.children[0]


class AssignAction(Node):
    def __init__(self):
        super(AssignAction, self).__init__()
        self.number_children = 1
        self.size = 0

    def to_string(self):
        if len(self.children) == 0:
            raise Exception('AssignAction: Incomplete Program')

        return 'act = ' + str(self.children[0])

    def interpret(self, env):
        if len(self.children) == 0:
            raise Exception('AssignAction: Incomplete Program')

        env['act'] = self.children[0]

    @classmethod
    def new(cls, var):
        inst = cls()
        inst.add_child(var)

        return inst


class Observation(Node):
    def __init__(self):
        super(Observation, self).__init__()
        self.number_children = 1 #?
        self.size = 0 #?

    @classmethod
    def new(cls, var):
        inst = cls()
        inst.add_child(var)

        return inst

    def to_string(self):
        if len(self.children) == 0:
            raise Exception('Observation: Incomplete Program')

        return 'obs[' + str(self.children[0]) + ']'

    def interpret(self, env):
        if len(self.children) == 0:
            raise Exception('Observation: Incomplete Program')

        return env['obs'][self.children[0]]

    def __eq__(self, other):
        if type(other) != Observation:
            return False
        if self.index == other.index:
            return True
        return False


class ReLU(Node):
    def __init__(self):
        super(ReLU, self).__init__()
        self.number_children = 1
        self.size = 1

    @classmethod
    def new(cls, weight_bias):
        inst = cls()
        inst.add_child(weight_bias)

        return inst

    def to_string(self):
        return 'max(0, ' + str(np.around(self.children[0][0], 3)) + " *dot* obs[:] + " + str(np.round(self.children[0][1], 3)) + ")"

    def interpret(self, env):
        return max(0.0, np.dot(self.children[0][0], env['obs']) + self.children[0][1])

    def __eq__(self, other):
        if type(other) != ReLU:
            return False
        if (self.weight == other.weight).all() and (self.bias == other.bias).all():
            return True
        return False


class ReLU_w(Node):
    def __init__(self):
        super(ReLU_w, self).__init__()
        self.number_children = 1
        self.size = 1

    @classmethod
    def new(cls, weight_bias):
        inst = cls()
        inst.add_child(weight_bias)

        return inst

    def to_string(self):
        return '(' + str(np.around(self.children[0][0], 3)) + " *dot* obs[:] + " + str(np.round(self.children[0][1], 3)) + ")"

    def interpret(self, env):
        return np.dot(self.children[0][0], env['obs']) + self.children[0][1]

    def __eq__(self, other):
        if type(other) != ReLU:
            return False
        if (self.weight == other.weight).all() and (self.bias == other.bias).all():
            return True
        return False


class Affine(Node):
    def __init__(self):
        super(Affine, self).__init__()
        self.number_children = 1
        self.size = 8

    @classmethod
    def new(cls, weight_bias):
        inst = cls()
        inst.add_child(weight_bias)
        return inst

    def to_string(self):
        return '(' + str(np.round(self.children[0][0], 3)) + " + " + str(np.around(self.children[0][1:], 3)) + " *dot* obs[:] " + ")"

    def interpret(self, env):
        return self.children[0][0] + np.dot(self.children[0][1:], env['obs'])

    def __eq__(self, other):
        if type(other) != ReLU:
            return False
        if (self.weight == other.weight).all() and (self.bias == other.bias).all():
            return True
        return False


class Linear(Node):
    def __init__(self):
        super(Linear, self).__init__()
        self.number_children = 1
        self.size = 7

    @classmethod
    def new(cls, weight_bias):
        inst = cls()
        inst.add_child(weight_bias)
        return inst

    def to_string(self):
        return '(' + str(np.around(self.children[0], 3)) + " *dot* obs[:] )"

    def interpret(self, env):
        return np.dot(self.children[0], env['obs'])


class Lt(Node):
    def __init__(self):
        super(Lt, self).__init__()
        self.number_children = 2

    @classmethod
    def new(cls, left, right):
        inst = cls()
        inst.add_child(left)
        inst.add_child(right)

        return inst

    def to_string(self):
        return "(" + self.children[0].to_string() + " < " + self.children[1].to_string() + ")"

    def interpret(self, env):
        return self.children[0].interpret(env) < self.children[1].interpret(env)


class Lt0(Node):
    def __init__(self):
        super(Lt0, self).__init__()
        self.number_children = 1

    @classmethod
    def new(cls, left):
        inst = cls()
        inst.add_child(left)

        return inst

    def to_string(self):
        return "(" + self.children[0].to_string() + " < 0)"

    def interpret(self, env):
        return self.children[0].interpret(env) < 0


class Gt0(Node):
    def __init__(self):
        super(Gt0, self).__init__()
        self.number_children = 1

    @classmethod
    def new(cls, left):
        inst = cls()
        inst.add_child(left)

        return inst

    def to_string(self):
        return "(" + self.children[0].to_string() + " > 0)"

    def interpret(self, env):
        return self.children[0].interpret(env) > 0


class Gt(Node):
    def __init__(self):
        super(Gt, self).__init__()
        self.number_children = 2

    @classmethod
    def new(cls, left, right):
        inst = cls()
        inst.add_child(left)
        inst.add_child(right)

        return inst

    def to_string(self):
        return "(" + self.children[0].to_string() + " > " + self.children[1].to_string() + ")"

    def interpret(self, env):
        return self.children[0].interpret(env) > self.children[1].interpret(env)



class Ite(Node):
    def __init__(self):
        super(Ite, self).__init__()

        self.number_children = 3

    @classmethod
    def new(cls, bool_expression, true_block, false_block):
        inst = cls()
        inst.add_child(bool_expression)
        inst.add_child(true_block)
        inst.add_child(false_block)

        return inst

    def to_string(self):
        return '(if ' + self.children[0].to_string() + ' then: ' + self.children[1].to_string() + ' else: ' + \
               self.children[2].to_string() + ")"

    def interpret(self, env):
        if self.children[0].interpret(env):
            return self.children[1].interpret(env)
        else:
            return self.children[2].interpret(env)


class Addition(Node):
    def __init__(self):
        super(Addition, self).__init__()

        self.number_children = 2

    @classmethod
    def new(cls, left, right):
        inst = cls()
        inst.add_child(left)
        inst.add_child(right)

        return inst

    def to_string(self):
        return "(" + self.children[0].to_string() + " + " + self.children[1].to_string() + ")"

    def interpret(self, env):
        return self.children[0].interpret(env) + self.children[1].interpret(env)


class Subtraction(Node):
    def __init__(self):
        super(Subtraction, self).__init__()
        self.number_children = 2

    @classmethod
    def new(cls, left, right):
        inst = cls()
        inst.add_child(left)
        inst.add_child(right)

        return inst

    def to_string(self):
        return "(" + self.children[0].to_string() + " - " + self.children[1].to_string() + ")"

    def interpret(self, env):
        return self.children[0].interpret(env) - self.children[1].interpret(env)


class Multiplication(Node):
    def __init__(self):
        super(Multiplication, self).__init__()
        self.number_children = 2

    @classmethod
    def new(cls, left, right):
        inst = cls()
        inst.add_child(left)
        inst.add_child(right)

        return inst

    def to_string(self):
        if len(self.children) < 2:
            raise Exception('Times: Incomplete Program')

        return "(" + self.children[0].to_string() + " * " + self.children[1].to_string() + ")"

    def interpret(self, env):
        if len(self.children) < 2:
            raise Exception('Times: Incomplete Program')

        return self.children[0].interpret(env) * self.children[1].interpret(env)


Node.restore_original_production_rules()


