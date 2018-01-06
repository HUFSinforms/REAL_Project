import cplex
from cplex.exceptions import CplexError
import sys
import numpy as np



def portfolio(sector,bench,asset,MCAPQ,beta,alpha,qmat,Q_con,returns,sols,w_upsums,big_w_dic,multiple):
    c = cplex.Cplex()
    t = c.variables.type


    c.variables.add(names=["d"+str(i) for i in asset],obj=alpha,lb=[-1*bench[j] for j in asset])

    c.variables.add(names=["assum"], lb=[-99999])

    # print(c.variables.get_lower_bounds())


    # c.variables.add(names=["y"+str(i) for i in asset],lb=[-99999999 for j in asset])
    #
    # c.variables.add(names=["c"+ str(j) + str(i) for j in asset for i in range(2)], types=[t.binary for i in asset for j in range(2)])

    c.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=["d" + str(i) for i in asset], val=alpha)], senses=["E"],
        rhs=[0.0], names=["sum"])
    c.linear_constraints.set_linear_components("sum" , [["assum"], [-1.0]])

    # 0.112046486647
    # -0.07957269716079549

    bigeer_bench_sum = 0
    for i in asset:
        bigeer_bench_sum = bigeer_bench_sum + bench[i]*big_w_dic[i]

    c.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=["d" + str(i) for i in asset], val=[big_w_dic[i] for i in asset])], senses=["E"],
        rhs=[w_upsums - bigeer_bench_sum], names=["w_big"])

    # c.linear_constraints.add(
    #     lin_expr=[cplex.SparsePair(ind=["assum"], val=[1.0])], senses=["E"],
    #     rhs=[returns], names=["sum"])

    for i in asset:
        c.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["d"+str(i)], val = [1.0])],senses=["G"],rhs=[-1*bench[i]],names=[str(i)+"di_wi"])
        # c.linear_constraints.set_linear_components(str(i)+"di_wi", [["w" + str(i)], [1.0]])
        c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d" + str(i)], val=[1.0])], senses=["L"],
                                 rhs=[0.05], names=["st_5_1"])
        c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=["d" + str(i)], val=[1.0])], senses=["G"],
                                 rhs=[-0.05], names=["st_5_2"])

    #
    # bench_sum = 0
    #
    # for i in bench:
    #     bench_sum += bench[i]

    # print(bench_sum)

    # c.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["d"+str(i) for i in asset], val = [1.0]*len(asset))],senses=["E"],rhs=[0],names=["st_4"])

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


    # for i in asset:
    #     c.linear_constraints.add(
    #         lin_expr=[cplex.SparsePair(ind=["q" + str(i)], val=[1.0])], senses=["G"],
    #         rhs=[bench[i]], names=["st_q"+str(i)])
    #     c.linear_constraints.set_linear_components("st_q"+str(i), [["d" + str(i)], [-1.0]])
    #
    #     c.linear_constraints.add(
    #         lin_expr=[cplex.SparsePair(ind=["q" + str(i)], val=[1.0])], senses=["L"],
    #         rhs=[0.999+bench[i]], names=["st_qq" + str(i)])
    #     c.linear_constraints.set_linear_components("st_qq" + str(i), [["d" + str(i)], [-1.0]])

        # c.linear_constraints.add(
        #     lin_expr=[cplex.SparsePair(ind=["y" + str(i)], val=[1.0])], senses=["L"],
        #     rhs=[bench[i]], names=["st_y1" + str(i)])
        # c.linear_constraints.set_linear_components("st_y1" + str(i), [["d" + str(i)], [-1.0]])
        #
        # c.linear_constraints.add(
        #     lin_expr=[cplex.SparsePair(ind=["y" + str(i)], val=[1.0])], senses=["L"],
        #     rhs=[bench[i]], names=["st_y2" + str(i)])
        #
        # c.linear_constraints.add(
        #     lin_expr=[cplex.SparsePair(ind=["y" + str(i)], val=[1.0])], senses=["G"],
        #     rhs=[bench[i]-999.0], names=["st_y3" + str(i)])
        # c.linear_constraints.set_linear_components("st_y3" + str(i), [["d" + str(i)], [-1.0]])
        # c.linear_constraints.set_linear_components("st_y3" + str(i), [["c" + str(i) + str(0)], [-999.0]])
        #
        # c.linear_constraints.add(
        #     lin_expr=[cplex.SparsePair(ind=["y" + str(i)], val=[1.0])], senses=["G"],
        #     rhs=[bench[i]-999.0], names=["st_y4" + str(i)])
        # c.linear_constraints.set_linear_components("st_y4" + str(i), [["c"+ str(i) + str(1)], [-999.0]])
        #
        # c.linear_constraints.add(
        #     lin_expr=[cplex.SparsePair(ind=["c" + str(i)+str(0)], val=[1.0])], senses=["E"],
        #     rhs=[1.0], names=["st_c" + str(i)])
        # c.linear_constraints.set_linear_components("st_c" + str(i), [["c" + str(i) + str(1)], [1.0]])



    nums = 0
    for i in asset:
        if nums in sols:
            c.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=["d" + str(i)], val=[1.0])], senses=["G"],
                rhs=[-1*bench[i]+0.00101], names=["st_select"+str(i)])

        else:
            c.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=["d" + str(i)], val=[1.0])], senses=["E"],
                rhs=[-1*bench[i]], names=["st_select" + str(i)])
        nums += 1








    # c.linear_constraints.add(
    #     lin_expr=[cplex.SparsePair(ind=["q" + str(i) for i in asset], val=[1.0]*len(asset))], senses=["G"],
    #     rhs=[50], names=["st_9_1"])
    #
    #
    # c.linear_constraints.add(
    #     lin_expr=[cplex.SparsePair(ind=["q" + str(i) for i in asset], val=[1.0]*len(asset))], senses=["L"],
    #     rhs=[70], names=["st_9_2"])

    # c.linear_constraints.add(
    #     lin_expr=[cplex.SparsePair(ind=["y" + str(i) for i in asset], val=[1.0] * len(asset))], senses=["G"],
    #     rhs=[0], names=["st_10_1"])
    #
    # c.linear_constraints.add(
    #     lin_expr=[cplex.SparsePair(ind=["y" + str(i) for i in asset], val=[1.0] * len(asset))], senses=["L"],
    #     rhs=[0.4], names=["st_10_2"])



    # for i in asset:
    #     Q = cplex.SparseTriple(ind1=["q"+str(i)], ind2=["d"+str(i),"z"+str(i)],
    #                        val=[-1.0,1.0])
    #     c.quadratic_constraints.add(rhs=bench[i], quad_expr=Q, name="Q"+str(i))
    # #
    # Q2 = cplex.SparseTriple(ind1=["q" + str(i) for i in asset], ind2=["d" + str(i) for i in asset],
    #                        val=[-1.0]*len(asset))
    # c.quadratic_constraints.add(rhs=0.6, quad_expr=Q2, name="st_10_2", sense="G")


    Q3 = cplex.SparseTriple(ind1=Q_con[0], ind2=Q_con[1], val=Q_con[2])
    c.quadratic_constraints.add(rhs=0.01*multiple, quad_expr=Q3, name="st_11_1", sense="L")


    # c.linear_constraints.add(
    #     lin_expr=[cplex.SparsePair(ind=["z" + str(i) for i in asset], val=[-1.0 for i in asset])], senses=["G"],
    #     rhs=[-0.4], names=["st_10_2"])



    c.objective.set_quadratic(qmat)

    c.objective.set_sense(c.objective.sense.minimize)
    c.solve()
    abcd=[]
    ab={}
    cd={}

    numa = 0
    for i in asset:

        # print("d" + str(i)  + " : " + str(c.solution.get_values("d" + str(i))))
        #if bench[i] + c.solution.get_values("d" + str(i)) > 0:
           # print("w" + str(i)  + " : " + str(bench[i] + c.solution.get_values("d" + str(i))))
            # print(numa)
        # print("q" + str(i) + " : " + str(c.solution.get_values("q" + str(i))))
        ab.update({str(i):bench[i] + c.solution.get_values("d" + str(i))})

        cd.update({str(i):c.solution.get_values("d" + str(i))})
        numa += 1
#    print("assum"+ " : " + str( c.solution.get_values("assum" )))
#    print(c.solution.get_objective_value())
#    print(c.solution.get_quadratic_slacks())
    abcd.append(ab)
    abcd.append(cd)
    abcd.append(c.solution.get_objective_value())
    return abcd