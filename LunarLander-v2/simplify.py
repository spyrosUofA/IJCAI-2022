import copy
import pickle


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
        #print(p.to_string() + '\n')
        if changes == 0:
            return p


from DSL import *
from evaluation import *
import time

#policy = pickle.load(open("../binary_programs/D000/Oracle-15/sa_cpus-16_n-25_c-None_run-8.pkl", "rb"))
#policy = pickle.load(open("../binary_programs/D010/Oracle-13/sa_cpus-16_n-25_c-None_run-3.pkl", "rb"))
#policy = pickle.load(open("../binary_programs/E010_D010_old/Oracle-12/sa_cpus-16_n-25_c-None_run-3.pkl", "rb"))
#policy = pickle.load(open("../binary_programs/D010/Oracle-4/sa_cpus-16_n-25_c-None_run-25.pkl", "rb"))
#policy = pickle.load(open("../binary_programs/D010/Oracle-9/sa_cpus-16_n-25_c-None_run-9.pkl", "rb"))

#policy = pickle.load(open("../binary_programs/E010_D010_old/Oracle-8/sa_cpus-16_n-25_c-None_run-BEST.pkl", "rb"))
#policy = pickle.load(open("./binary_programs/D010/Oracle-64x64/1/sa_cpus-1_n-100_c-None_run-100.pkl", "rb"))


policy = Ite.new(Gt0.new(Addition.new(Observation.new(2), Observation.new(3))), AssignAction.new(1), AssignAction.new(0))

#(if (max(0, [-0.014  0.099  0.235  0.273] *dot* obs[:] + 0.1) < 0.10763176744760765) then: act = 0 else: act = 1)


print(policy.to_string() + '\n')
p_new = simplify_program1(policy)
print(p_new.to_string() + '\n')
print("------------------")
print(Environment({}, 200, 1).evaluate(policy))
exit()


exit()


rew = 0
for i in range(1, 16, 1):
    policy = pickle.load(open("../binary_programs/E010_D010_old/Oracle-" + str(i) + "/sa_cpus-16_n-25_c-None_run-BEST.pkl", "rb"))

    print(policy.to_string() + '\n')
    policy = simplify_program(policy)
    policy = simplify_program(policy)
    policy = simplify_program(policy)
    policy = simplify_program(policy)
    print(policy.to_string() + '\n')

    rew += Environment({}, 200, 1).evaluate(policy)

print(rew / 15.0)
# E010_D010_old: 163.48063343676603

