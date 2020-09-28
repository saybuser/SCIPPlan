import math

from pyscipopt import Model, sqrt, log, exp
from pyscipopt.scip import Expr, VarExpr, SumExpr, ProdExpr, quicksum #, GenExpr, ExprCons, Term

def encode_scipplan(domain, instance, horizon, epsilon, gap):

    bigM = 1000.0
    
    model = Model(domain + "_" + instance + "_" + str(horizon))
    
    initials = readConstraintFiles("./translation/initial_" + domain + "_" + instance+".txt")
    instantaneous_constraints = readConstraintFiles("./translation/instantaneous_constraints_" + domain + "_" + instance+".txt")
    temporal_constraints = readConstraintFiles("./translation/temporal_constraints_" + domain + "_" + instance+".txt")
    goals = readConstraintFiles("./translation/goal_" + domain + "_" + instance+".txt")
    transitions = readTransitions("./translation/transitions_" + domain + "_" + instance+".txt")
    reward = readReward("./translation/reward_" + domain + "_" + instance+".txt")
    
    A, S, Aux, A_type, S_type, Aux_type = readVariables("./translation/pvariables_"+domain+"_"+instance+".txt")
    
    model, x, y, v, d = initialize_original_variables(model, A, S, Aux, A_type, S_type, Aux_type, horizon)

    model = encode_initial_constraints(model, S, y, initials)
    model, d, v, Aux = encode_global_instantaneous_constraints(model, A, S, Aux, x, y, v, d, instantaneous_constraints, horizon, bigM)
    model, d, v, Aux = encode_global_temporal_constraints(model, A, S, Aux, x, y, v, d, temporal_constraints, horizon, bigM)
    model, d = encode_transitions(model, A, S, Aux, x, y, v, d, transitions, horizon)
    model, d = encode_goal_constraints(model, S, Aux, y, v, d, goals, horizon)
    model = encode_reward(model, A, S, Aux, x, y, v, d, reward, horizon)
    
    for t in range(horizon):
        model.addCons(x[("dt",t)] <= bigM)
        
    model.setRealParam("limits/gap", gap)
    
    while True:
        model.optimize()
        if len(model.getSols()) == 0:
            print("Problem is infeasible for the given horizon.")
            return False
        violated_t, interval, violated_c_index = checkTemporalConstraintViolation(model, A, S, Aux, x, y, v, d, temporal_constraints, horizon, epsilon)
        #print(violated_t, interval, violated_c_index)
        if violated_t == -1:
            break
        zero_crossing_coef = ((interval[1] + interval[0]) / 2.0) / model.getVal(x[("dt",violated_t)])
        model.freeTransform()
        model = encode_violated_global_temporal_constraint(model, A, S, Aux, x, y, v, d, temporal_constraints, horizon, violated_t, zero_crossing_coef, violated_c_index)
    
    print("Plan:")
    for t in range(horizon):
        for index, a in enumerate(A):
            print(a + " at time " + str(t) + " by value " + str(model.getVal(x[(a,t)])))
    return True

def checkTemporalConstraintViolation(model, A, S, Aux, x, y, v, d, temporal_constraints, horizon, epsilon):

    bigM = 1000.0
    
    constraints = []
    or_index = {}
    constraint_index = {}
    c_index = 0
    for index, constraint in enumerate(temporal_constraints):
        if "OR" in constraint:
            temp_constraints = (",".join(constraint)).split(",OR,")
            for temp_index, temp_constraint in enumerate(temp_constraints):
                constraints.append(temp_constraint.split(","))
                or_index[c_index] = [temp_index, len(temp_constraints)-1]
                constraint_index[c_index] = index
                c_index += 1
        else:
            constraints.append(constraint)
            or_index[c_index] = [0, 0]
            constraint_index[c_index] = index
            c_index += 1

    interval = [-1.0*epsilon, -1.0*epsilon]
    violated_constraint_index = -1

    for t in range(horizon):
        dt_val = model.getVal(x[("dt",t)])
        time = 0.0
        violationFound = False
        while time <= dt_val:
            c_index = 0
            num_unsat = 0
            while c_index < len(constraints):
                if interval[0] >= 0.0 and violated_constraint_index != constraint_index[c_index]:
                    c_index += 1
                    continue
                if or_index[c_index][0] == 0:
                    num_unsat = 0
                constraint = constraints[c_index]
                terms = constraint[:-2]
                LHS_val = 0.0
                for t_index, term in enumerate(terms):
                    coef = "1.0"
                    variables = term.split("*")
                    if variables[0] not in A + S + Aux:
                        coef = variables[0]
                        variables = variables[1:]
                    if set(A).isdisjoint(variables) or t < horizon:
                        term_val = float(coef)
                        for var in variables:
                            if var == "dt":
                                term_val *= time
                            else:
                                if var in A:
                                    term_val *= model.getVal(x[(var,t)])
                                elif var in S:
                                    term_val *= model.getVal(y[(var,t)])
                                else:
                                    term_val *= model.getVal(v[(var,t)])
                        LHS_val += term_val
            
                if ("<=" == constraint[len(constraint)-2] and LHS_val > float(constraint[len(constraint)-1])) or (">=" == constraint[len(constraint)-2] and LHS_val < float(constraint[len(constraint)-1])) or ("==" == constraint[len(constraint)-2] and (LHS_val < float(constraint[len(constraint)-1]) or LHS_val > float(constraint[len(constraint)-1]))):
                    if or_index[c_index][1] > 0:
                        num_unsat += 1
                    if num_unsat - 1 == or_index[c_index][1]:
                        if interval[0] <= -1.0*epsilon:
                            interval[0] = time
                            violated_constraint_index = constraint_index[c_index]
                        interval[1] = time
                elif interval[1] >= 0.0:
                    return t, interval, violated_constraint_index

                c_index += 1
            time += epsilon

    return -1, [-1.0*epsilon, -1.0*epsilon], -1

def encode_violated_global_temporal_constraint(model, A, S, Aux, x, y, v, d, temporal_constraints, horizon, violated_t, zero_crossing_coef, violated_c_index):

    bigM = 1000.0
    
    constraints = []
    
    if "OR" in temporal_constraints[violated_c_index]:
        temp_constraints = (",".join(temporal_constraints[violated_c_index])).split(",OR,")
        for temp_index, temp_constraint in enumerate(temp_constraints):
            split_constraint = temp_constraint.split(",")
            bool_name = "OR_Temp_" + str(violated_c_index) + "_" + str(temp_index)
            if "<=" == split_constraint[len(split_constraint)-2]:
                split_constraint.insert(len(split_constraint)-2, str(-1.0*bigM) + "*" + bool_name)
            elif ">=" == split_constraint[len(split_constraint)-2]:
                split_constraint.insert(len(split_constraint)-2, str(bigM) + "*" + bool_name)
            constraints.append(split_constraint)
    else:
        constraints.append(temporal_constraints[violated_c_index])

    #print(constraints)

    for c_index, constraint in enumerate(constraints):
        terms = constraint[:-2]
        con_expr = SumExpr()
        empty = True
        for t_index, term in enumerate(terms):
            coef = 1.0
            variables = term.split("*")
            if variables[0] not in A + S + Aux:
                coef = float(variables[0])
                variables = variables[1:]
            if set(A).isdisjoint(variables) or violated_t < horizon:
                for var in variables:
                    if var == "dt":
                        coef *= zero_crossing_coef
                        
                if len(variables) > 1:
                    con_expr += coef * VarExpr(d[("Temp",violated_c_index,t_index,violated_t)])
                else:
                    if variables[0] in A:
                        con_expr += coef * VarExpr(x[(variables[0],violated_t)])
                    elif variables[0] in S:
                        con_expr += coef * VarExpr(y[(variables[0],violated_t)])
                    else:
                        con_expr += coef * VarExpr(v[(variables[0],violated_t)])
                empty = False
        
        if not empty:
            #print(con_expr)
            if "<=" == constraint[len(constraint)-2]:
                model.addCons(con_expr <= float(constraint[len(constraint)-1]))
            elif ">=" == constraint[len(constraint)-2]:
                model.addCons(con_expr >= float(constraint[len(constraint)-1]))
            else:
                model.addCons(con_expr == float(constraint[len(constraint)-1]))

    return model

def readConstraintFiles(directory):
    
    import os
    
    listOfConstraints = []
    
    if os.path.exists(directory):
        file = open(directory,"r")
        constraints = file.read().splitlines()
        
        for constraint in constraints:
            listOfConstraints.append(constraint.split(","))
    else:
        print("No file provided.")
    
    return listOfConstraints

def readTransitions(directory):

    import os

    listOfTransitions = []

    if os.path.exists(directory):
        file = open(directory,"r")
        transitions = file.read().splitlines()
        
        for transition in transitions:
            listOfTransitions.append(transition.split(","))
    else:
        print("No file provided.")
    
    return listOfTransitions

def readReward(directory):
    
    import os
    
    listOfReward = []
    
    if os.path.exists(directory):
        file = open(directory,"r")
        reward = file.read().splitlines()
        
        for rew in reward:
            listOfReward.append(rew.split(","))
    else:
        print("No file provided.")
    
    return listOfReward

def readVariables(directory):
    
    A = []
    S = []
    Aux = []
    A_type = []
    S_type = []
    Aux_type = []
    
    variablesFile = open(directory,"r")
    data = variablesFile.read().splitlines()

    for dat in data:
        variables = dat.split(",")
        for var in variables:
            if "action_continuous:" in var or "action_boolean:" in var or "action_integer:" in var:
                if "action_continuous:" in var:
                    A.append(var.replace("action_continuous: ",""))
                    A_type.append("C")
                elif "action_boolean:" in var:
                    A.append(var.replace("action_boolean: ",""))
                    A_type.append("B")
                else:
                    A.append(var.replace("action_integer: ",""))
                    A_type.append("I")
            elif "state_continuous:" in var or "state_boolean:" in var or "state_integer:" in var:
                if "state_continuous:" in var:
                    S.append(var.replace("state_continuous: ",""))
                    S_type.append("C")
                elif "state_boolean:" in var:
                    S.append(var.replace("state_boolean: ",""))
                    S_type.append("B")
                else:
                    S.append(var.replace("state_integer: ",""))
                    S_type.append("I")
            else:
                if "auxiliary_continuous:" in var:
                    Aux.append(var.replace("auxiliary_continuous: ",""))
                    Aux_type.append("C")
                elif "auxiliary_boolean:" in var:
                    Aux.append(var.replace("auxiliary_boolean: ",""))
                    Aux_type.append("B")
                else:
                    Aux.append(var.replace("auxiliary_integer: ",""))
                    Aux_type.append("I")

    return A, S, Aux, A_type, S_type, Aux_type

def initialize_original_variables(model, A, S, Aux, A_type, S_type, Aux_type, horizon):
    
    # Create vars for each action a, time step t
    x = {}
    for index, a in enumerate(A):
        for t in range(horizon):
            x[(a,t)] = model.addVar(a + "_" + str(t), vtype=A_type[index])

    # Create vars for each state s, time step t
    y = {}
    for index, s in enumerate(S):
        for t in range(horizon+1):
            y[(s,t)] = model.addVar(s + "_" + str(t), vtype=S_type[index])

    # Create vars for each auxilary variable aux, time step t
    v = {}
    for index,aux in enumerate(Aux):
        for t in range(horizon+1):
            v[(aux,t)] = model.addVar(aux + "_" + str(t), vtype=Aux_type[index])

    # Create vars for each nonlinear term n, time step t
    d = {}

    return model, x, y, v, d

def encode_initial_constraints(model, S, y, initials):
    
    for init in initials:
        variables = init[:-2]
        init_expr = Expr()
        for var in variables:
            coef = "1.0"
            if "*" in var:
                coef, var = var.split("*")
            init_expr += float(coef) * y[(var,0)]

        model.addCons(init_expr == float(init[len(init)-1]))

    return model

def encode_global_instantaneous_constraints(model, A, S, Aux, x, y, v, d, instantaneous_constraints, horizon, bigM): #variables for each i) nonlinear term and ii) boolean expression
    
    constraints = []
    for c_index, constraint in enumerate(instantaneous_constraints):
        if "OR" in constraint:
            temp_constraints = (",".join(constraint)).split(",OR,")
            for temp_index, temp_constraint in enumerate(temp_constraints):
                split_constraint = temp_constraint.split(",")
                Aux.append("OR_Inst_" + str(c_index) + "_" + str(temp_index))
                if "<=" == split_constraint[len(split_constraint)-2]:
                    split_constraint.insert(len(split_constraint)-2, str(-1.0*bigM) + "*" + Aux[len(Aux)-1])
                elif ">=" == split_constraint[len(split_constraint)-2]:
                    split_constraint.insert(len(split_constraint)-2, str(bigM) + "*" + Aux[len(Aux)-1])
                for t in range(horizon+1):
                    v[(Aux[len(Aux)-1],t)] = model.addVar(Aux[len(Aux)-1] + "_" + str(t), vtype="B")
                constraints.append(split_constraint)
            constraints.append(Aux[-1*len(temp_constraints):] + ["<=",str(len(temp_constraints)-1)])
        else:
            constraints.append(constraint)

    for t in range(horizon+1):
        for c_index, constraint in enumerate(constraints):
            terms = constraint[:-2]
            con_expr = SumExpr()
            empty = True
            for t_index, term in enumerate(terms):
                coef = "1.0"
                variables = term.split("*")
                if variables[0] not in A + S + Aux:
                    coef = variables[0]
                    variables = variables[1:]
                if set(A).isdisjoint(variables) or t < horizon:
                    prod_expr = ProdExpr()
                    for var in variables:
                        if var in A:
                            prod_expr *= VarExpr(x[(var,t)])
                        elif var in S:
                            prod_expr *= VarExpr(y[(var,t)])
                        else:
                            prod_expr *= VarExpr(v[(var,t)])
                
                    if len(variables) > 1:
                        d[("Inst",c_index,t_index,t)] = model.addVar("cons_inst_" + str(c_index) + "_" + str(t_index) + "_" + str(t))
                        model.addCons(prod_expr == d[("Inst",c_index,t_index,t)])

                        con_expr += float(coef) * d[("Inst",c_index,t_index,t)]
                    else:
                        con_expr += float(coef) * prod_expr
                    empty = False
            
            if not empty:
                #print(con_expr)
                if "<=" == constraint[len(constraint)-2]:
                    model.addCons(con_expr <= float(constraint[len(constraint)-1]))
                elif ">=" == constraint[len(constraint)-2]:
                    model.addCons(con_expr >= float(constraint[len(constraint)-1]))
                else:
                    model.addCons(con_expr == float(constraint[len(constraint)-1]))

    return model, d, v, Aux

def encode_global_temporal_constraints(model, A, S, Aux, x, y, v, d, temporal_constraints, horizon, bigM): #variables for each i) nonlinear term and ii) boolean expression
    
    constraints = []
    for c_index, constraint in enumerate(temporal_constraints):
        if "OR" in constraint:
            temp_constraints = (",".join(constraint)).split(",OR,")
            for temp_index, temp_constraint in enumerate(temp_constraints):
                split_constraint = temp_constraint.split(",")
                Aux.append("OR_Temp_" + str(c_index) + "_" + str(temp_index))
                if "<=" == split_constraint[len(split_constraint)-2]:
                    split_constraint.insert(len(split_constraint)-2, str(-1.0*bigM) + "*" + Aux[len(Aux)-1])
                elif ">=" == split_constraint[len(split_constraint)-2]:
                    split_constraint.insert(len(split_constraint)-2, str(bigM) + "*" + Aux[len(Aux)-1])
                for t in range(horizon+1):
                    v[(Aux[len(Aux)-1],t)] = model.addVar(Aux[len(Aux)-1] + "_" + str(t), vtype="B")
                constraints.append(split_constraint)
            constraints.append(Aux[-1*len(temp_constraints):] + ["<=",str(len(temp_constraints)-1)])
        else:
            constraints.append(constraint)
    
    for t in range(horizon+1):
        for c_index, constraint in enumerate(constraints):
            terms = constraint[:-2]
            con_expr = SumExpr()
            empty = True
            for t_index, term in enumerate(terms):
                coef = "1.0"
                variables = term.split("*")
                if variables[0] not in A + S + Aux:
                    coef = variables[0]
                    variables = variables[1:]
                if set(A).isdisjoint(variables) or t < horizon:
                    prod_expr = ProdExpr()
                    for var in variables:
                        if var in A:
                            prod_expr *= VarExpr(x[(var,t)])
                        elif var in S:
                            prod_expr *= VarExpr(y[(var,t)])
                        else:
                            prod_expr *= VarExpr(v[(var,t)])
                
                    if len(variables) > 1:
                        d[("Temp",c_index,t_index,t)] = model.addVar("cons_temp_" + str(c_index) + "_" + str(t_index) + "_" + str(t))
                        model.addCons(prod_expr == d[("Temp",c_index,t_index,t)])
                        
                        con_expr += float(coef) * d[("Temp",c_index,t_index,t)]
                    else:
                        con_expr += float(coef) * prod_expr
                    empty = False
                                
            if not empty:
                #print(con_expr)
                if "<=" == constraint[len(constraint)-2]:
                    model.addCons(con_expr <= float(constraint[len(constraint)-1]))
                elif ">=" == constraint[len(constraint)-2]:
                    model.addCons(con_expr >= float(constraint[len(constraint)-1]))
                else:
                    model.addCons(con_expr == float(constraint[len(constraint)-1]))

    return model, d, v, Aux

def encode_transitions(model, A, S, Aux, x, y, v, d, transitions, horizon): #variables for each i) nonlinear term and ii) boolean expression
    
    for t in range(horizon):
        for tran_index, transition in enumerate(transitions):
            tran_expr = SumExpr()
            #print(transition[0][:-1])
            tran_expr += VarExpr(y[(transition[0][:-1],t+1)])
            
            terms = transition[1:-2]
            #empty = True
            for t_index, term in enumerate(terms):
                coef = "1.0"
                variables = term.split("*")
                if variables[0] not in A + S + Aux:
                    coef = variables[0]
                    variables = variables[1:]
                if set(A).isdisjoint(variables) or t < horizon:
                    prod_expr = ProdExpr()
                    for var in variables:
                        if var in A:
                            prod_expr *= VarExpr(x[(var,t)])
                        elif var in S:
                            prod_expr *= VarExpr(y[(var,t)])
                        else:
                            prod_expr *= VarExpr(v[(var,t)])
                
                    if len(variables) > 1:
                        d[("Tran",tran_index,t_index,t)] = model.addVar("tran_" + str(tran_index) + "_" + str(t_index) + "_" + str(t))
                        model.addCons(prod_expr == d[("Tran",tran_index,t_index,t)])
                        
                        tran_expr += float(coef) * d[("Tran",tran_index,t_index,t)]
                    else:
                        tran_expr += float(coef) * prod_expr
                    #empty = False
                                
            #if not empty:
                #print(con_expr)
            if "<=" == transition[len(transition)-2]:
                model.addCons(tran_expr <= float(transition[len(transition)-1]))
            elif ">=" == transition[len(transition)-2]:
                model.addCons(tran_expr >= float(transition[len(transition)-1]))
            else:
                model.addCons(tran_expr == float(transition[len(transition)-1]))

    return model, d

def encode_goal_constraints(model, S, Aux, y, v, d, goals, horizon): #variables for each i) nonlinear term and ii) boolean expression
    
    for g_index, goal in enumerate(goals):
        terms = goal[:-2]
        sum_expr = SumExpr()
        empty = True
        for t_index, term in enumerate(terms):
            coef = "1.0"
            variables = term.split("*")
            if variables[0] not in S + Aux:
                coef = variables[0]
                variables = variables[1:]
            prod_expr = ProdExpr()
            for var in variables:
                if var in S:
                    prod_expr *= VarExpr(y[(var,horizon)])
                else:
                    prod_expr *= VarExpr(v[(var,horizon)])
    
            if len(variables) > 1:
                d[(g_index,t_index,horizon)] = model.addVar("goal_" + str(g_index) + "_" + str(t_index) + "_" + str(horizon))
                model.addCons(prod_expr == d[(g_index,t_index,horizon)])
                    
                sum_expr += float(coef) * d[(g_index,t_index,horizon)]
            else:
                sum_expr += float(coef) * prod_expr
            empty = False
            
        if not empty:
            #print(sum_expr)
            if "<=" == goal[len(goal)-2]:
                model.addCons(sum_expr <= float(goal[len(goal)-1]))
            elif ">=" == goal[len(goal)-2]:
                model.addCons(sum_expr >= float(goal[len(goal)-1]))
            else:
                model.addCons(sum_expr == float(goal[len(goal)-1]))
                    
    return model, d

def encode_reward(model, A, S, Aux, x, y, v, d, reward, horizon):
    
    sum_expr = Expr()
    empty = True
    for t in range(horizon):
        for t_index, term in enumerate(reward[0]):
            coef = "1.0"
            variables = term.split("*")
            if variables[0] not in A + S + Aux:
                coef = variables[0]
                variables = variables[1:]
            if set(A).isdisjoint(variables) or t < horizon:
                prod_expr = ProdExpr()
                for var in variables:
                    if var in A:
                        prod_expr *= VarExpr(x[(var,t)])
                    elif var in S:
                        prod_expr *= VarExpr(y[(var,t)])
                    else:
                        prod_expr *= VarExpr(v[(var,t)])
            
                d[(t_index,t)] = model.addVar("rew_" + str(t_index) + "_" + str(t))
                model.addCons(prod_expr == d[(t_index,t)])
                sum_expr += float(coef) * d[(t_index,t)]
                empty = False
        
    if not empty:
        #print(sum_expr)
        model.setObjective(sum_expr, "maximize")
    
    return model

def get_args():
    
    import sys
    argv = sys.argv
    
    myargs = {}
    
    for index, arg in enumerate(argv):
        if arg[0] == '-':
            myargs[arg] = argv[index+1]

    return myargs

if __name__ == '__main__':
    
    import os
    myargs = get_args()
    
    setDomain = False
    setInstance = False
    setHorizon = False
    setEpsilon = False
    setGap = False
    
    for arg in myargs:
        if arg == "-d":
            domain = myargs[(arg)]
            setDomain = True
        elif arg == "-i":
            instance = myargs[(arg)]
            setInstance = True
        elif arg == "-h":
            horizon = myargs[(arg)]
            setHorizon = True
        elif arg == "-e":
            epsilon = myargs[(arg)]
            setEpsilon = True
        elif arg == "-g":
            gap = myargs[(arg)]
            setGap = True

    if setDomain and setInstance:
        if not setEpsilon:
            epsilon = "0.01"
            print 'Epsilon is not provided, and is set to 0.01.'
        if not setHorizon:
            horizon = "1"
            print 'Horizon is not provided, and is set to 1.'
        if not setGap:
            gap = "0.001"
            print 'Optimality gap is not provided, and is set to 0.001.'
        import time
        start_time = time.time()
        while not encode_scipplan(domain, instance, int(horizon), float(epsilon), float(gap)):
            horizon = str(int(horizon) + 1)
        print 'Total Time: %.4f seconds' % (time.time() - start_time)
    elif not setDomain:
        print 'Domain is not provided.'
    else:
        print 'Instance is not provided.'
