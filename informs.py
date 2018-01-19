import cplex
from cplex.exceptions import CplexError
import sys
import numpy as np



def portfolio(sector,bench,asset,MCAPQ,beta,alpha,qmat,Q_con,multiple,time_init):
    c = cplex.Cplex()
    t = c.variables.type


    c.variables.add(names=["d"+str(i) for i in asset],obj=alpha,lb=[-1*bench[j] for j in asset])
    c.variables.add(names=["q"+ str(j)  for j in asset ], types=[t.binary for i in asset])
    c.variables.add(names=["assum"], lb=[-99999])

    c.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=["d" + str(i) for i in asset], val=alpha)], senses=["E"],
        rhs=[0.0], names=["sum"])
    c.linear_constraints.set_linear_components("sum" , [["assum"], [-1.0]])

    for i in asset:
        c.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["d"+str(i)], val = [1.0])],senses=["G"],rhs=[-1*bench[i]],names=[str(i)+"di_wi"])
        # c.linear_constraints.set_linear_components(str(i)+"di_wi", [["w" + str(i)], [1.0]])
        c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d" + str(i)], val=[1.0])], senses=["L"],
                                 rhs=[0.05], names=["st_5_1"])
        c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d" + str(i)], val=[1.0])], senses=["G"],
                                 rhs=[-0.05], names=["st_5_2"])

    c.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["d"+str(i) for i in asset], val = [1.0]*len(asset))],senses=["E"],rhs=[0],names=["st_4"])

    for j in sector:
        c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d" + str(i) for i in sector[j]], val=[1.0]*len(sector[j]))], senses=["L"],
                                 rhs=[0.1], names=["st_6_1"])
        c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d" + str(i) for i in sector[j]], val=[1.0]*len(sector[j]))], senses=["G"],
                                 rhs=[-0.1], names=["st_6_2"])


    for j in MCAPQ:
        c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d" + str(i) for i in MCAPQ[j]], val=[1.0]*len(MCAPQ[j]))], senses=["L"],
                                 rhs=[0.1], names=["st_7_1"])
        c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d" + str(i) for i in MCAPQ[j]], val=[1.0]*len(MCAPQ[j]))], senses=["G"],
                                 rhs=[-0.1], names=["st_7_2"])


    c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d" + str(i) for i in asset], val=[beta[i] for i in asset])], senses=["L"],
                             rhs=[0.1], names=["st_8_1"])
    c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d" + str(i) for i in asset], val=[beta[i] for i in asset])], senses=["G"],
                             rhs=[-0.1], names=["st_8_2"])


    for i in asset:
        c.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=["q" + str(i)], val=[1.0])], senses=["G"],
            rhs=[bench[i]], names=["st_q"+str(i)])
        c.linear_constraints.set_linear_components("st_q"+str(i), [["d" + str(i)], [-1.0]])

        c.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=["q" + str(i)], val=[1.0])], senses=["L"],
            rhs=[0.999+bench[i]], names=["st_qq" + str(i)])
        c.linear_constraints.set_linear_components("st_qq" + str(i), [["d" + str(i)], [-1.0]])


    c.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=["q" + str(i) for i in asset], val=[1.0]*len(asset))], senses=["G"],
        rhs=[60], names=["st_9_1"])


    c.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=["q" + str(i) for i in asset], val=[1.0]*len(asset))], senses=["L"],
        rhs=[70], names=["st_9_2"])

    Q3 = cplex.SparseTriple(ind1=Q_con[0], ind2=Q_con[1], val=Q_con[2])
    c.quadratic_constraints.add(rhs=0.01*multiple, quad_expr=Q3, name="st_11_1", sense="L")

    c.objective.set_quadratic(qmat)

    c.objective.set_sense(c.objective.sense.minimize)
    c.parameters.threads.set(1)
    c.parameters.timelimit.set(time_init)

    c.solve()
    total0=[]
    total1=[]
    total2=[]
    total3=[]
    w_dics0={}
    d_dics0={}
    w_dics1={}
    d_dics1={}
    w_dics2={}
    d_dics2={}
    w_dics3={}
    d_dics3={}
    
    

    numa = 0
    print(c.solution.get_status())
    
    pool_list = []
    
    for i in asset:
        w_dics0.update({str(i):bench[i] + c.solution.pool.get_values(0, "d" + str(i))})
        d_dics0.update({str(i):c.solution.pool.get_values(0, "d" + str(i))})
        w_dics1.update({str(i):bench[i] + c.solution.pool.get_values(1, "d" + str(i))})
        d_dics1.update({str(i):c.solution.pool.get_values(1, "d" + str(i))})
        w_dics2.update({str(i):bench[i] + c.solution.pool.get_values(2, "d" + str(i))})
        d_dics2.update({str(i):c.solution.pool.get_values(2, "d" + str(i))})
        w_dics3.update({str(i):bench[i] + c.solution.pool.get_values(3, "d" + str(i))})
        d_dics3.update({str(i):c.solution.pool.get_values(3, "d" + str(i))})
        
        numa += 1
    total0.append(w_dics0)
    total0.append(d_dics0)
    total0.append(c.solution.pool.get_objective_value(0))
    total1.append(w_dics1)
    total1.append(d_dics1)
    total1.append(c.solution.pool.get_objective_value(1))
    total2.append(w_dics2)
    total2.append(d_dics2)
    total2.append(c.solution.pool.get_objective_value(2))
    total3.append(w_dics3)
    total3.append(d_dics3)
    total3.append(c.solution.pool.get_objective_value(3))
    
    pool_list.append(total0)
    pool_list.append(total1)
    pool_list.append(total2)
    pool_list.append(total3)
    
    return pool_list