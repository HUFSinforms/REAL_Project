

import networkx as nx
import numpy as np
import math
import cplex
from scipy.stats import norm
import pickle
import collections
import numba
from pyomo.environ import *
import inspec


class Prob:

    def __init__(self,
                 dataDir,
                 B=10,
                 mid_node_range=1,
                 veh_range=10,
                 reachability_fn=None,
                 OD_num_limit=-1,
                 node_weight_limit=-1,
                 dview=None,
                 probability_precision=4):

        self.dataDir = dataDir

        nodes = [n.split() for n in file(dataDir + '/NODES.txt', 'r')]
        arcs = [[int(v) for v in a.split()] for a in file(dataDir + '/ARCS.txt', 'r')]

        G_orig = nx.Graph()

        for n in nodes:
            G_orig.add_node(int(n[0]), weight=int(n[1]), pos_x=float(n[2]), pos_y=float(n[3]))

        for a in arcs:
            G_orig.add_edge(a[2], a[3], length=a[1])

        self.G_orig = G_orig
        self.veh_range = veh_range
        self.mid_node_range = mid_node_range

        self.N_c = set()
        self.B = B

        self.OD_num_limit = OD_num_limit
        self.node_weight_limit = node_weight_limit


        if reachability_fn is None:
            # deterministic case - use step function
            self.reachability_fn = Reachability.DeterministicCDF
            self.range_uncertainty = False
            self.probability_limit = 1
            self.probability_precision = 0
        else:
            self.reachability_fn = reachability_fn
            self.range_uncertainty = True
            self.probability_limit = 0.5
            self.probability_precision = probability_precision

        self.K, self.Demands = self.MakeDemand()
        self.ExpandedNetworks = self.MakeAllExpandedNetworks(dview)

        self.c_i = {i:1 for i in self.N_c}




    def MakeDemand(self):
        G = self.G_orig

        Demands = {}

        K = []

        for i in G:
            for j in (j for j in G if j > i):
                if self.node_weight_limit > 0 and \
                    (G.node[i]['weight'] < self.node_weight_limit or G.node[j]['weight'] < self.node_weight_limit):
                    continue

                if G.node[i]['weight'] * G.node[j]['weight'] > 0:
                    D = int((G.node[i]['weight'] * G.node[j]['weight'])
                                    / pow(nx.dijkstra_path_length(G, i, j, weight='length'), 1.5))
                    k = (i,j)
                    K.append(k)
                    Demands[k] = D


        if self.OD_num_limit > 0:
            Demands = {d[0]:d[1] for d in sorted(Demands.items(), key=lambda x: x[1], reverse=True)[:self.OD_num_limit]}
            K = Demands.keys()


        return K, Demands

    def MakeAllExpandedNetworks(self, dview = None):

        K = self.K
        Demands = self.Demands
        ExpandedNetworks = {}

        if dview is None:
            for k in K:
                s = k[0]
                t = k[1]
                #                 ExpandedNetworks[k] = self.MakeExpandedNetwork(s, t)
                _, ExpandedNetworks[k], N_c = MakeExpandedNetwork(
                    self.G_orig,
                    s,
                    t,
                    self.veh_range,
                    self.mid_node_range,
                    self.reachability_fn,
                    self.probability_limit,
                    self.probability_precision
                )
                for nn in N_c:
                    if nn != 's' and nn != 't':
                        self.N_c.add(nn)

        else:
            dview['MakeExpandedNetwork'] = MakeExpandedNetwork
            n = len(self.K)
            networks = dview.map_sync(MakeExpandedNetwork,
                                          [self.G_orig] * n,
                                          [k[0] for k in self.K],
                                          [k[1] for k in self.K],
                                          [self.veh_range] * n,
                                          [self.mid_node_range] * n,
                                          [self.reachability_fn] * n,
                                          [self.probability_limit] * n,
                                          [self.probability_precision] * n
                                      )
            for k, en, N_c in networks:
                ExpandedNetworks[k] = en
                for nn in N_c:
                    if nn != 's' and nn != 't':
                        self.N_c.add(nn)

        return ExpandedNetworks
    #
    # def MakeExpandedNetwork(self, i, j):
    #
    #     G = self.G_orig
    #
    #     veh_range = self.veh_range
    #     mid_node_range = self.mid_node_range
    #
    #     def ConnectIfReachable(G, i, j, dist_multiplier):
    #         if i != j:
    #             try:
    #                 dist = nx.dijkstra_path_length(G, i, j, weight='length')
    #                 probability = self.reachability_fn(dist, self.veh_range)
    #                 roundtrip_probability = self.reachability_fn(dist * dist_multiplier, self.veh_range)
    #                 if roundtrip_probability >= self.probability_limit:
    #                     EN.add_edge(i, j, length=dist, probability=probability)
    #             except:
    #                 pass
    #
    #
    #     EN = nx.DiGraph()
    #
    #     path = nx.dijkstra_path(G, i, j, weight='length')
    #
    #     EN.add_nodes_from(path, type='normal')
    #
    #
    #     for n1,n2 in zip(path[:-1], path[1:]):
    #         length = G[n1][n2]['length']
    #         if length > mid_node_range:
    #             # num_midnode = int(math.ceil(length/mid_node_range)) - 1
    #             # prev_node = n1
    #             # for t in range(num_midnode):
    #             #     t_name = '{%d~%d@%d}' % (min(n1,n2), max(n1,n2), t)
    #             #     EN.add_node(t_name, type='mid')
    #             #     probability = self.reachability_fn(mid_node_range, self.veh_range)
    #             #     EN.add_edge(prev_node, t_name, length=mid_node_range, probability=probability)
    #             #     prev_node = t_name
    #
    #             num_midnode = 0
    #             prev_node = n1
    #             dist = mid_node_range
    #             while dist < length:
    #                 t_name = '{%d~%d@%d}' % (min(n1,n2), max(n1,n2), num_midnode)
    #                 EN.add_node(t_name, type='mid')
    #                 probability = self.reachability_fn(mid_node_range, self.veh_range)
    #                 EN.add_edge(prev_node, t_name, length=mid_node_range, probability=probability)
    #                 prev_node = t_name
    #                 dist += mid_node_range
    #                 num_midnode += 1
    #
    #
    #
    #             EN.add_edge(prev_node, n2, length=length - num_midnode*mid_node_range,
    #                         probability=self.reachability_fn(length - num_midnode * mid_node_range, self.veh_range))
    #
    #         else:
    #             EN.add_edge(n1, n2, length=length, probability=self.reachability_fn(length, self.veh_range))
    #
    #     for n1 in EN.node:
    #         for n2 in EN.node:
    #             if n1 != n2:
    #                 ConnectIfReachable(EN, n1, n2, 1)
    #
    #
    #     EN.add_node('s', type='source')
    #     EN.add_node('t', type='sink')
    #
    #     EN.add_edge('s', path[0], length=0, probability=1)
    #     EN.add_edge(path[-1], 't', length=0, probability=1)
    #
    #
    #     for n in EN.node:
    #         if n != 's':
    #             ConnectIfReachable(EN, 's', n, 2)
    #
    #         if n != 't':
    #             ConnectIfReachable(EN, n, 't', 2)
    #
    #     for n in EN.node:
    #         if n != 's' and n != 't':
    #             self.N_c.add(n)
    #
    #
    #     for i,j,e in EN.edges_iter(data=True):
    #         e['log_probability'] = np.around(-np.log(e['probability']) * 10.0 ** self.probability_precision, decimals=0)
    #
    #     return EN



    def CalPathProbability(self, k, path, reachability_fn=None):

        G = self.ExpandedNetworks[k]

        if path is None or len(path) <= 1:
            return 0

        if reachability_fn is None:
            prob = 1
            for idx in range(1, len(path)):
                prob *= G[path[idx-1]][path[idx]]['probability']
            return prob
        else:
            prob = 1
            for idx in range(1, len(path)):
                prob *= reachability_fn(G[path[idx-1]][path[idx]]['length'])
            return prob


def MakeExpandedNetwork(G, i, j, veh_range, mid_node_range, reachability_fn, probability_limit, probability_precision):

    def ConnectIfReachable(G, i, j, dist_multiplier):
        if i != j:
            try:
                dist = nx.dijkstra_path_length(G, i, j, weight='length')
                probability = reachability_fn(dist, veh_range)
                roundtrip_probability = reachability_fn(dist * dist_multiplier, veh_range)
                if roundtrip_probability >= probability_limit:
                    EN.add_edge(i, j, length=dist, probability=probability)
            except:
                pass


    EN = nx.DiGraph()

    path = nx.dijkstra_path(G, i, j, weight='length')

    EN.add_nodes_from(path, type='normal')


    for n1,n2 in zip(path[:-1], path[1:]):
        length = G[n1][n2]['length']
        if length > mid_node_range:
            # num_midnode = int(math.ceil(length/mid_node_range)) - 1
            # prev_node = n1
            # for t in range(num_midnode):
            #     t_name = '{%d~%d@%d}' % (min(n1,n2), max(n1,n2), t)
            #     EN.add_node(t_name, type='mid')
            #     probability = self.reachability_fn(mid_node_range, self.veh_range)
            #     EN.add_edge(prev_node, t_name, length=mid_node_range, probability=probability)
            #     prev_node = t_name

            num_midnode = 0
            prev_node = n1
            dist = mid_node_range
            while dist < length:
                t_name = '{%d~%d@%d}' % (min(n1,n2), max(n1,n2), num_midnode)
                EN.add_node(t_name, type='mid')
                probability = reachability_fn(mid_node_range, veh_range)
                EN.add_edge(prev_node, t_name, length=mid_node_range, probability=probability)
                prev_node = t_name
                dist += mid_node_range
                num_midnode += 1



            EN.add_edge(prev_node, n2, length=length - num_midnode*mid_node_range,
                        probability=reachability_fn(length - num_midnode * mid_node_range, veh_range))

        else:
            EN.add_edge(n1, n2, length=length, probability=reachability_fn(length, veh_range))

    for n1 in EN.node:
        for n2 in EN.node:
            if n1 != n2:
                ConnectIfReachable(EN, n1, n2, 1)


    EN.add_node('s', type='source')
    EN.add_node('t', type='sink')

    EN.add_edge('s', path[0], length=0, probability=1)
    EN.add_edge(path[-1], 't', length=0, probability=1)


    for n in EN.node:
        if n != 's':
            ConnectIfReachable(EN, 's', n, 2)

        if n != 't':
            ConnectIfReachable(EN, n, 't', 2)

    N_c = set()
    for n in EN.node:
        if n != 's' and n != 't':
            N_c.add(n)

    for _,_,e in EN.edges_iter(data=True):
        e['log_probability'] = max(np.around(-np.log(e['probability']) * 10.0 ** probability_precision, decimals=0), 0.0)


    return (i,j), EN, N_c

def GaussianCDF(x, veh_range):
    return 1 if x==0 else (1-norm(loc=veh_range, scale=veh_range/5.0).cdf(x))

def DeterministicCDF(x, veh_range):
    return 1 if x <= veh_range else 0

class Reachability:

    GaussianCDF = staticmethod(GaussianCDF)

    DeterministicCDF = staticmethod(DeterministicCDF)


class Solver:

    def __init__(self):
        pass

    def Solve(self):

        results = self.InternalSolve()

        p = self.p

        results['Problem'] = {
            'dataDir': p.dataDir,
            'reachability_fn': None if p.reachability_fn is None else inspect.getsource(p.reachability_fn),
            'veh_range': p.veh_range,
            'mid_node_range': p.mid_node_range,
            'budget': p.B,
            'OD_num_limit': p.OD_num_limit

        }

        return results


    def InternalSolve(self):
        raise NotImplementedError

    def GetSolutionPaths(self, reachability_fn=None):
        raise NotImplementedError



class MIPSolver(Solver):

    def __init__(self, p):
        self.p = p

        if not p.range_uncertainty:
            self.FormulateFlowMIP_Det(p)
        else:
            self.FormulateFlowMINLP(p)




    def FormulateFlowMIP_Det(self, p):

        cpx = cplex.Cplex()

        Demand = p.Demands
        K = p.K
        N_c = p.N_c
        B = p.B
        c_i = p.c_i

        z_k = lambda k: 'z_%s' % str(k)
        y_i = lambda i: 'y_%s' % str(i)
        x_a_k = lambda a,k: 'x_(%s~%s)_%s' % (str(a[0]), str(a[1]), str(k))


        cpx.objective.set_sense(cpx.objective.sense.maximize)

        # demand cover variables
        cpx.variables.add(
            obj = [Demand[k] for k in K],
            lb = [0] * len(K),
            ub = [1] * len(K),
            types = ['C'] * len(K),
            names = [z_k(k) for k in K]
        )

        # design variables
        cpx.variables.add(
            obj = [0] * len(N_c),
            lb = [0] * len(N_c),
            ub = [1] * len(N_c),
            types = ['B'] * len(N_c),
            names = [y_i(i) for i in N_c]
        )


        for k in K:

            G = p.ExpandedNetworks[k]
            A = G.edges()
            N = G.nodes()

            # flow variables
            cpx.variables.add(
                obj = [0] * len(A),
                lb = [0] * len(A),
                ub = [1] * len(A),
                types = ['C'] * len(A),
                names = [x_a_k(a,k) for a in A]
            )

            for i in N:
                # flow balance constraints
                cpx.linear_constraints.add(
                    lin_expr=[
                        cplex.SparsePair(
                            ind = [x_a_k(a,k) for a in G.out_edges(i)] + [x_a_k(a,k) for a in G.in_edges(i)] + [z_k(k)],
                            val = [1] * len(G.out_edges(i)) + [-1] * len(G.in_edges(i)) + [-1 if i=='s' else 1 if i=='t' else 0]
                        )
                    ],
                    senses=['E'],
                    rhs=[0],
                    names=['fb_%s_%s' % (i, k)]
                )


                if i in N_c:
                    cpx.linear_constraints.add(
                        lin_expr=[
                            cplex.SparsePair(
                                ind = [x_a_k(a,k) for a in G.out_edges(i)] + [y_i(i)],
                                val = [1] * len(G.out_edges(i)) + [-1]
                            )
                        ],
                        senses=['L'],
                        rhs=[0],
                        names=['capa_%s_%s' % (i, k)]
                    )


            # budget constraint
            cpx.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(
                        ind = [y_i(i) for i in N_c],
                        val = [c_i[i] for i in N_c]
                    )
                ],
                senses=['L'],
                rhs=[B],
                names=['budget']
            )

        self.cpx = cpx
        self.x_a_k = x_a_k
        self.y_i = y_i
        self.z_k = z_k


    def FormulateFlowMINLP(self, p):

        m = ConcreteModel()


        ToArcStr = lambda a: '%s->%s' % (a[0], a[1])
        self.ToArcStr = ToArcStr

        m.K = Set(initialize=range(len(p.K)))

        def N_init(model, k):
            return p.ExpandedNetworks[p.K[k]].nodes()
        def A_init(model, k):
            E = p.ExpandedNetworks[p.K[k]].edges()
            return [ToArcStr(e) for e in E]


        m.Nk = Set(m.K, initialize=N_init)
        m.Ak = Set(m.K, initialize=A_init)



        def KA_init(model, k):
            return [a for a in m.Ak[k]]

        m.KA = Set(m.K, initialize=KA_init)

        m.Nc = Set(initialize=p.N_c)


        def KNc_init(model, k):
            return [i for i in p.N_c if i in m.Nk[k]]

        m.KNc = Set(m.K, initialize=KNc_init)


        m.x = Var([(k,a) for k in m.K for a in m.KA[k]], domain=Binary)

        m.y = Var([i for i in m.Nc], domain=Binary)

        m.z = Var([k for k in m.K], domain=Binary)


        m.p = Var([k for k in m.K], domain=NonNegativeReals, bounds=(0,1))

        m.w = Var([(k, a) for k in m.K for a in m.KA[k]], bounds=(0,1))

        # m.v = Var([(k, a) for k in m.K for a in m.KA[k]], domain=Binary)


        def Budget_rule(model):
            return sum([model.y[i] for i in model.Nc]) == p.B

        m.Budget = Constraint(rule=Budget_rule)


        def Capa_rule(model, k, i):
            return sum([model.x[k,ToArcStr(a)] for a in p.ExpandedNetworks[p.K[k]].out_edges(i)]) - model.y[i] <= 0

        m.Capa = Constraint([(k, i) for k in m.K for i in m.KNc[k]], rule=Capa_rule)


        def FB_rule(model, k, i):
            inflow = sum([model.x[k,ToArcStr(a)] for a in p.ExpandedNetworks[p.K[k]].in_edges(i)])
            outflow = sum([model.x[k,ToArcStr(a)] for a in p.ExpandedNetworks[p.K[k]].out_edges(i)])
            if i == 's':
                return inflow - outflow + model.z[k] == 0
            elif i == 't':
                return inflow - outflow - model.z[k] == 0
            else:
                return inflow - outflow == 0

        m.FB = Constraint([(k,i) for k in m.K for i in m.Nk[k]], rule=FB_rule)




        pp = lambda x: math.exp((-round(-math.log(x) * 10.0 ** p.probability_precision) / (10.0**p.probability_precision)))

        m.probability = {(k, ToArcStr(a)): pp(p.ExpandedNetworks[p.K[k]][a[0]][a[1]]['probability']) for k in m.K for a in p.ExpandedNetworks[p.K[k]].edges()}

        def W1_rule(model, k, a):
            return model.w[k,a] <= model.probability[k,a] * model.x[k,a] + 1.0 - model.x[k,a]

        # def W2_rule(model, k, a):
        #     return model.w[k,a] <= model.probability[k,a] * model.x[k,a] + model.v[k,a]
        # def W3_rule(model, k, a):
        #     return model.w[k,a] >= 1 - model.x[k,a]
        # def W4_rule(model, k, a):
        #     return model.w[k,a] <= 1 - model.x[k,a] + 1 - model.v[k,a]

        m.W1 = Constraint([(k, a) for k in m.K for a in m.KA[k]], rule=W1_rule)
#        m.W2 = Constraint([(k, a) for k in m.K for a in m.KA[k]], rule=W2_rule)
#        m.W3 = Constraint([(k, a) for k in m.K for a in m.KA[k]], rule=W3_rule)
#        m.W4 = Constraint([(k, a) for k in m.K for a in m.KA[k]], rule=W4_rule)


        def Prob_rule(model, k):

            pp = 1
            for a in model.KA[k]:
                pp *= model.w[k,a]

            return model.p[k] <= pp


        m.Prob = Constraint(m.K, rule=Prob_rule)


        def Obj_rule(model):
            return sum(p.Demands[p.K[k]] * model.p[k] * model.z[k] for k in m.K)

        m.Obj = Objective(rule=Obj_rule, sense=maximize)



        self.opt = SolverFactory('couenne')
        self.MINLP_model = m


    def GetSolutionPaths(self, reachability_fn=None):

        sol_paths = []

        p = self.p

        if self.p.range_uncertainty:
            model = self.MINLP_model

            sol_paths = []
            for k, val in model.z.items():

                d = p.Demands[p.K[k]]
                if val.value > 0.5:
                    G = p.ExpandedNetworks[p.K[k]]

                    path = []
                    i = 's'
                    path.append(i)
                    probability = 1
                    while i != 't':
                        for a in G.out_edges(i):
                            if model.x[k, self.ToArcStr(a)]() > 0.5:
                                i = a[1]
                                path.append(i)
                                probability *= G.edge[a[0]][a[1]]['probability'] \
                                    if reachability_fn is None \
                                    else reachability_fn(G.edge[a[0]][a[1]]['length'])

                    sol_paths.append((p.K[k], path, probability, d, int(d*probability)))
                else:
                    sol_paths.append((p.K[k], None, 0, d, 0))

        else:
            cpx = self.cpx
            for k in self.p.K:

                if cpx.solution.get_values(self.z_k(k)) > 0.5:
                    G = self.p.ExpandedNetworks[k]

                    path = []
                    i = 's'
                    path.append(i)
                    probability = 1
                    while i != 't':
                        for a in G.out_edges(i):
                            if cpx.solution.get_values(self.x_a_k(a,k)) > 0.5:
                                i = a[1]
                                path.append(i)
                                probability *= G.edge[a[0]][a[1]]['probability'] \
                                    if reachability_fn is None \
                                    else reachability_fn(G.edge[a[0]][a[1]]['length'])

                    sol_paths.append((k, path, probability, self.p.Demands[k], int(self.p.Demands[k]*probability)))
                else:
                    sol_paths.append((k, None, 0, self.p.Demands[k], 0))

        return sol_paths


    def InternalSolve(self):

        if not self.p.range_uncertainty:
            starttime = self.cpx.get_time()
            self.cpx.solve()
            endtime = self.cpx.get_time()
            return {
                'Solver' : 'MIP',
                'time' : endtime - starttime,
                'status' : self.cpx.solution.get_status(),
                'status_msg' : self.cpx.solution.get_status_string(),
                'objval' : self.cpx.solution.get_objective_value(),
                'sol_design' : [i for i in self.p.N_c if self.cpx.solution.get_values(self.y_i(i)) > 0.5]
            }
        else:
            results = self.opt.solve(self.MINLP_model)
            return {
                'Solver' : 'MINLP',
                'time' : results.Solver().Time,
                'status' : str(results.Solver().Status),
                'status_msg' : results.Solver().Message,
                'objval' : self.MINLP_model.Obj.expr(),
                'sol_design' : [y for y,val in self.MINLP_model.y.items() if val.value > 0.5]
            }











class CGSubProb:

    def __init__(self, p, k, use_jit=False, early_chk=False, maximum_num_chargings=1000, minimum_dist=0):

        self.p = p
        self.Demand = p.Demands[k]
        self.k = k
        self.G = p.ExpandedNetworks[k]
        self.Paths = []

        self.N_k_c = list(set(self.G.nodes()) & set(p.N_c))

        self.use_jit = use_jit
        self.early_chk = early_chk

        n = len(self.G.nodes())
        adj_mat = nx.to_numpy_matrix(self.G)
        self.out_list = [[j for j in range(n) if adj_mat[i,j] > 0] for i in range(n)]
        self.in_list = [[i for i in range(n) if adj_mat[i,j] > 0] for j in range(n)]


        # a hack(!!!!!!!!!) for numba jit-compiling
        # empty list([]) in the first element causes type checking error from numba...
        if len(self.out_list[0]) == 0:
            self.out_list.insert(-1, [-1])


        self.length_mat = nx.to_numpy_matrix(self.G, weight='length', nonedge=0)
        self.probability_mat = nx.to_numpy_matrix(self.G, weight='probability', nonedge=1)
        self.log_probability_mat = np.around(-np.log(self.probability_mat) * 10.0 ** p.probability_precision, decimals=0)

        self.n = n

        self.s = self.G.nodes().index('s')
        self.t = self.G.nodes().index('t')

        self.name_to_idx = lambda name: self.G.nodes().index(name)

        self.veh_range = p.veh_range

        self.maximum_num_chargings = maximum_num_chargings
        self.minimum_dist = minimum_dist

        self.cpx, \
        self.con_conv, \
        self.con_capa_i, \
        self.con_capa_names, \
        self.f_g, \
        self.N_k_c,\
        self.con_capa_begin,\
        self.con_capa_end = self.FormulateMP()

        self.AddInitialColumns()


    def FormulateMP(self):

        cpx = cplex.Cplex()
        cpx.set_log_stream(None)
        cpx.set_results_stream(None)


        Demand = self.Demand
        N_k_c = self.N_k_c

        f_g = lambda g: 'f_%d' % (g)

        con_conv = 'con_conv'
        con_capa_i = lambda i: 'con_capa_%s' % (i)

        con_capa_begin = 0
        con_capa_end = 0

        cpx.objective.set_sense(cpx.objective.sense.maximize)

        # convexity constraints
        cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[],
                    val=[]
                )
            ],
            senses=['L'],
            rhs=[1],
            names=[con_conv]
        )

        con_capa_begin = cpx.linear_constraints.get_num()
        for i in N_k_c:
            # capa constraints
            cpx.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(
                        ind=[],
                        val=[]
                    )
                ],
                senses=['L'],
                rhs=[0],
                names=[con_capa_i(i)]
            )

        con_capa_end = cpx.linear_constraints.get_num()-1

        cpx.set_problem_type(0)

        return \
            cpx,\
            con_conv,\
            con_capa_i,\
            [con_capa_i(i) for i in N_k_c],\
            f_g,\
            N_k_c,\
            con_capa_begin,\
            con_capa_end


    def AddInitialColumns(self):

        G = self.G
        path = nx.dijkstra_path(G, 's', 't', weight='length')
        self.AddColumn(path)


    def AddColumn(self, path, probability=None):
        Demand = self.Demand

        cpx = self.cpx
        f_g = self.f_g
        Paths = self.Paths

        con_conv = self.con_conv
        con_capa_i = self.con_capa_i

        if probability is None:
            probability = self.PathProbability(path)

        Paths.append(path)

        g = len(Paths)

        cpx.variables.add(
            # obj=[int(Demand * probability)],
            obj=[(Demand * probability)],
            lb=[0],
            ub=[cplex.infinity],
            names=[f_g(g)],
            columns=[
                cplex.SparsePair(
                    ind=[con_conv] + [con_capa_i(i) for i in path[1:-1]],
                    val=[1] + [1] * len(path[1:-1])
                )
            ]
        )

    def PathProbability(self, path, reachablity_fn=None):

        path_idx = [self.name_to_idx(i) for i in path]

        if reachablity_fn is None:
            log_prob = 0
            i = path_idx[0]
            for j in path_idx[1:]:
                log_prob += self.log_probability_mat[i,j]
                i = j
            return np.exp(-log_prob / (10.0**self.p.probability_precision))
        else:
            prob = 1
            i = path_idx[0]
            for j in path_idx[1:]:
                prob *= reachablity_fn(self.length_mat[i,j])
                i = j
            return prob




    def FixCapacity(self, y):
        N_k_c = self.N_k_c
        cpx = self.cpx
        con_capa_i = self.con_capa_i

        rhs = [(con_capa_i(i), 1 if i in y else 0) for i in N_k_c]

        cpx.linear_constraints.set_rhs(rhs)



    def ColumnGeneration(self):

        cpx = self.cpx
        G = self.G
        con_conv = self.con_conv
        con_capa_i = self.con_capa_i
        con_capa_names = self.con_capa_names
        N_k_c = self.N_k_c
        Demand = self.Demand


        it = 0
        while True:
            it += 1
            # solve the CG master
            t1 = self.cpx.get_time()
#            cpx.write('cg%d.lp' % it)
            cpx.solve()
            self.p.benders_metric['time:cg:mp'] += self.cpx.get_time() - t1

            pi = cpx.solution.get_dual_values(con_conv)
            phi_i = cpx.solution.get_dual_values(self.con_capa_begin, self.con_capa_end)


            # set dual costs on edges
            nx.set_edge_attributes(G, 'dual', 0)

            nx.set_node_attributes(G, 'phi', 0)

            for i, phi in zip(N_k_c, phi_i):
                if abs(phi) > 0.0001:
                    G.node[i]['phi'] = phi
                    for a in G.out_edges_iter(i, data=True):
                        a[2]['dual'] = phi

            t1 = self.cpx.get_time()

#            print phi, phi_i

            col_added = self.Pricing(G, Demand, pi)

            self.p.benders_metric['time:cg:sp'] += self.cpx.get_time() - t1

            if not col_added:
                break


        return (pi, {i:val for i,val in zip(N_k_c, phi_i)})

    def Pricing(self, G, Demand, pi):

        col_added = False

        if not self.p.range_uncertainty:
            # deterministic case
            length, path = nx.single_source_dijkstra(G, 's', weight='dual')
            reduced_cost = Demand - pi - length['t']
            sht_path = path['t']

            if reduced_cost > 0.001:
                self.AddColumn(sht_path)
                col_added = True

        else:
            s = self.s
            t = self.t
            nodes = G.nodes()
            n = len(nodes)
            cost1_mat = nx.to_numpy_matrix(G, weight='dual')
            cost2_mat = self.log_probability_mat
            out_list = self.out_list
            in_list = self.in_list


            try:

                # if the first list of parameter is empty, numba fails in inferring the type information...
                # we simply fail over to non-jit
                if self.use_jit and len(out_list[0]) > 0:
                    D = CGSubProb.BiObjSP_jit(
                        s,
                        t,
                        n,
                        cost1_mat,
                        cost2_mat,
                        out_list,
                        early_chk=self.early_chk,
                        demand=Demand,
                        pi=pi,
                        probability_precision=self.p.probability_precision
                    )
                else:
                    D = self.BiObjSP(
                        s,
                        t,
                        n,
                        cost1_mat,
                        cost2_mat,
                        out_list,
                        early_chk=self.early_chk,
                        demand=Demand,
                        pi=pi,
                        probability_precision=self.p.probability_precision
                    )

                col_num = 0
                for l in D[t]:
                    probability = np.exp(-l[1] / (10.0**self.p.probability_precision))
                    # exp_demand = int(Demand * probability)
                    exp_demand = (Demand * probability)
                    reduced_cost = exp_demand - pi - l[0]
                    if reduced_cost > 0.001:
                        path = self.PathBacktracking(D, l, s, t, n, cost1_mat, cost2_mat, in_list)
                        sht_path = [nodes[i] for i in path]

                        if self.CheckPathFeasibility(self.k, path, self.length_mat, self.log_probability_mat):
                            self.AddColumn(sht_path, probability)
                            col_added = True
                            col_num += 1
                            # print reduced_cost,
                            if col_num > 10000:
                                break

                # print
            except:
                data = {
                    'k': self.k,
                    's': s,
                    't': t,
                    'n': n,
                    'mat1': cost1_mat,
                    'mat2': cost2_mat,
                    'in_list': in_list,
                    'out_list': out_list

                }
                pickle.dump(data, file('data.dmp', 'w'))
                raise Exception


        return col_added

    def CheckPathFeasibility(self, k, path, dist_mat, log_prob_mat):
        num_chargings = len(path) - 2

        if num_chargings > self.maximum_num_chargings > 0:
            return False

        if num_chargings <= 1:
            return True

        if self.minimum_dist > 0:
            for idx in range(1, len(path)-2):
                if dist_mat[path[idx], path[idx+1]] < self.minimum_dist * self.veh_range:
                    return False

        return True


    def GetSolutionPath(self, reachability_fn=None):
        cpx = self.cpx

        f_begin, f_end = cpx.variables.get_indices([self.f_g(1), self.f_g(len(self.Paths))])

        sol = cpx.solution.get_values(f_begin, f_end)

        for g, path in zip(sol, self.Paths):
            if g > 0.9:
                prob = self.PathProbability(path, reachability_fn)
                return self.k, path, prob, self.Demand, int(self.Demand * prob)

        return self.k, None, 0, self.Demand, 0



    # Below are used for parallel execution
    SubProblems = None

    @staticmethod
    def SolveSubProblem(sol_y, k):
        sp = CGSubProb.SubProblems[k]
        sp.FixCapacity(sol_y)
        num_col_before = sp.cpx.variables.get_num()
        pi_k, phi_k = sp.ColumnGeneration()

        return pi_k, phi_k, sp.cpx.variables.get_num() - num_col_before





    @staticmethod
    def Merge(X, Y):
        m = 0
        i = 0
        j = 0

        e = 0.0001

        p = X.shape[0]
        q = Y.shape[0]

        M = np.zeros((p+q, 3))

        while True:

            if i >= p:
                M[m:m+q-j] = Y[j:q]
                m = m+q-j
                return M[:m]
            if j >= q:
                M[m:m+p-i] = X[i:p]
                m = m+p-i
                return M[:m]

            if X[i,0] - Y[j,0] < -e:
                while j < q and X[i,1] <= Y[j,1]:
                    j += 1
                M[m] = X[i]
                m += 1
                i += 1
            elif Y[j,0] - X[i,0] < -e:
                while i < p and Y[j,1] <= X[i,1]:
                    i += 1
                M[m] = Y[j]
                m += 1
                j += 1
            elif X[i,1] - Y[j,1] < -e:
                while j < q and Y[j,1] >= X[i,1]:
                    j += 1
                M[m] = X[i]
                m += 1
                i += 1
            else:
                while i < p and X[i,1] >= Y[j,1]:
                    i += 1
                M[m] = Y[j]
                m += 1
                j += 1



    @staticmethod
    def BiObjSP(
            s,
            t,
            n,
            cost1_mat,
            cost2_mat,
            out_list,
            early_chk=False,
            demand=0,
            pi=0,
            probability_precision=0):


        D = [np.zeros((0,3))] * n

        D[s] = np.zeros((1,3))
        D[s][:,2] = s


        Labeled = collections.deque([s])
        Labeled_mask = [False] * n
        Labeled_mask[s] = True


        while Labeled:
            u = Labeled.popleft()
            Labeled_mask[u] = False

            for j in out_list[u]:
                len_prob = np.array((cost1_mat[u,j], cost2_mat[u,j], 0))
                Du = D[u] + len_prob
                Du[:,2] = u
                if D[j][:,0:2].shape != Du[:,0:2].shape or not np.allclose(D[j][:,0:2], Du[:,0:2], atol=0.001):
                    DM = CGSubProb.Merge(D[j], Du)
                    if D[j][:,0:2].shape != DM[:,0:2].shape or not np.allclose(DM[:,0:2], D[j][:,0:2], atol=0.001):
                        D[j] = DM

                        if early_chk and j == t:
                            for l in DM:
                                probability = np.exp(-l[1] / (10.0**probability_precision))
                                exp_demand = int(demand * probability)
                                reduced_cost = exp_demand - pi - l[0]
                                if reduced_cost > 0.001:
                                    return D

                        if not Labeled_mask[j]:
                            Labeled.append(j)
                            Labeled_mask[j] = True


        return D



    @staticmethod
    @numba.jit(nopython=True)
    def Merge_jit(X, Y):
        m = 0
        i = 0
        j = 0

        e = 0.0001

        p = X.shape[0]
        q = Y.shape[0]

        M = np.zeros((p+q, 3))

        while True:

            if i >= p:
                M[m:m+q-j] = Y[j:q]
                m = m+q-j
                return M[:m]
            if j >= q:
                M[m:m+p-i] = X[i:p]
                m = m+p-i
                return M[:m]

            if X[i,0] - Y[j,0] < -e:
                while j < q and X[i,1] <= Y[j,1]:
                    j += 1
                M[m] = X[i]
                m += 1
                i += 1
            elif Y[j,0] - X[i,0] < -e:
                while i < p and Y[j,1] <= X[i,1]:
                    i += 1
                M[m] = Y[j]
                m += 1
                j += 1
            elif X[i,1] - Y[j,1] < -e:
                while j < q and Y[j,1] >= X[i,1]:
                    j += 1
                M[m] = X[i]
                m += 1
                i += 1
            else:
                while i < p and X[i,1] >= Y[j,1]:
                    i += 1
                M[m] = Y[j]
                m += 1
                j += 1


    @staticmethod
    @numba.jit(nopython=True)
    def IsSame_jit(X, Y):
        same = X.shape == Y.shape
        if same:
            num_l = X.shape[0]
            for l in range(num_l):
                if np.abs(X[l,0]-Y[l,0]) > 0.001:
                    same = False
                    break
                if np.abs(X[l,1]-Y[l,1]) > 0.001:
                    same = False
                    break

        return same

    @staticmethod
    @numba.jit(nopython=False, cache=True)
    def BiObjSP_jit(
            s,
            t,
            n,
            cost1_mat,
            cost2_mat,
            out_list,
            early_chk=False,
            demand=0,
            pi=0,
            probability_precision=0):

        if n == 1:
            return [np.zeros((0,3))] * n

        D = [np.zeros((0,3))] * n

        D[s] = np.zeros((1,3))
        D[s][:,2] = s

        # if the first element of the list has negative number...
        # we skip it
        # a hack for ensuring the numba type-checking...
        if len(out_list[0]) > 0 and out_list[0][0] < 0:
            out_list2 = out_list[1:]
        else:
            out_list2 = out_list


        QUEUE_SIZE = n + 1
        Labeled = np.zeros(QUEUE_SIZE, np.int32)
        f = -1
        r = -1

        if f == -1:
            f = 0
        r = (r+1) % QUEUE_SIZE
        Labeled[r] = s


        Labeled_mask = [False] * n
        Labeled_mask[s] = True


        while f != -1:
            u = Labeled[f]
            if f == r:
                f = -1
                r = -1
            else:
                f = (f+1) % QUEUE_SIZE

            Labeled_mask[u] = False

            for j in out_list2[u]:
                Du = D[u].copy()
                Du[:,0] += cost1_mat[u,j]
                Du[:,1] += cost2_mat[u,j]
                Du[:,2] = u

                same = CGSubProb.IsSame_jit(D[j], Du)

                if not same:
                    DM = CGSubProb.Merge_jit(D[j], Du)

                    same = CGSubProb.IsSame_jit(D[j], DM)
                    if not same:
                        D[j] = DM

                        if early_chk and j == t:
                            for l in DM:
                                probability = np.exp(-l[1] / (10.0**probability_precision))
                                exp_demand = int(demand * probability)
                                reduced_cost = exp_demand - pi - l[0]
                                if reduced_cost > 0.001:
                                    return D

                        if not Labeled_mask[j]:
                            if f == -1:
                                f = 0
                            r = (r+1) % QUEUE_SIZE
                            Labeled[r] = j

                            Labeled_mask[j] = True


        return D



    @staticmethod
    def PathBacktracking(D, l, s, t, n, cost1_mat, cost2_mat, in_list):
        j = t
        path = collections.deque()
        while True:
            path.appendleft(j)
            if j == s:
                break
            find = False
            i = int(l[2])
            c = np.array((cost1_mat[i,j], cost2_mat[i,j]))

            for l2 in D[i]:
                if (l2[0] + c[0] - l[0]) < 0.001 and (l2[1] + c[1] - l[1]) < 0.001:
                    j = i
                    l = l2
                    find = True
                    break
                if find:
                    break

            if not find:
                raise Exception('PathBacktracking error!!!!')

        return path



class BendersSolver(Solver):

    def __init__(
            self,
            p,
            dview=None,
            use_jit=False,
            early_chk=False,
            maximum_num_chargings=-1, # maximum allowed recharges
            minimum_dist=0 # minimum distance between two recharges, ratio of mean travel range
    ):

        p.benders_metric = {}
        p.benders_metric['time:bd:mp'] = 0
        p.benders_metric['time:cg:sp'] = 0
        p.benders_metric['time:cg:mp'] = 0
        p.benders_metric['num:bdcut'] = 0
        p.benders_metric['num:cg:col'] = 0

        if dview is None:
            p.benders_metric['num:cpu'] = 0
        else:
            p.benders_metric['num:cpu'] = len(dview)

        self.use_jit = use_jit

        self.early_chk = early_chk

        self.dview = dview
        self.p = p

        self.maximum_num_chargings = maximum_num_chargings
        self.minimum_dist = minimum_dist

        self.cpx,\
        self.z,\
        self.y_i,\
        self.y_begin,\
        self.y_end = self.FormulateMaster()

        self.SubProblems = self.MakeSubproblems()




        if use_jit:
            # compile numba function beforehand
            CGSubProb.BiObjSP_jit(0, 0, 1, np.zeros((2,2)), np.zeros((2,2)), [[0]])

            # if dview is not None:
            #     dview.execute('CGSubProb.BiObjSP_jit(0, 0, 1, np.zeros((2,2)), np.zeros((2,2)), [[0]])')
            #     dview.wait()


    def FormulateMaster(self):
        p = self.p

        Demand = p.Demands
        K = p.K
        N_c = p.N_c
        B = p.B
        c_i = p.c_i

        cpx = cplex.Cplex()

        z = 'z'
        y_i = lambda i: 'y_%s' % str(i)


        cpx.objective.set_sense(cpx.objective.sense.maximize)
        cpx.set_log_stream(None)
        cpx.set_results_stream(None)

        # z
        cpx.variables.add(
            obj = [1],
            lb = [0],
            ub = [sum([Demand[k] for k in K])],
            types = ['C'],
            names = [z]
        )

        y_begin = cpx.variables.get_num()

        # design variables
        cpx.variables.add(
            obj = [0] * len(N_c),
            lb = [0] * len(N_c),
            ub = [1] * len(N_c),
            types = ['B'] * len(N_c),
            names = [y_i(i) for i in N_c]
        )

        y_end = cpx.variables.get_num() - 1

        # budget constraint
        cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind = [y_i(i) for i in N_c],
                    val = [c_i[i] for i in N_c]
                )
            ],
            senses=['L'],
            rhs=[B],
            names=['budget']
        )


        return cpx, z, y_i, y_begin, y_end



    def SolveMaster(self):

        cpx = self.cpx

        cpx.solve()


        sol_y = [i for i in self.p.N_c if cpx.solution.get_values(self.y_i(i)) > 0.5]

        return cpx.solution.get_objective_value(), sol_y


    def MakeSubproblems(self):

        p = self.p

        SubProblems = None

        if self.dview is None:
            K = p.K
            SubProblems = [CGSubProb(p, k, self.use_jit, self.early_chk, self.maximum_num_chargings, self.minimum_dist) for k in K]
        else:
            # parallel run
            # self.dview['p'] = p
            # a dirty heck... otherwise it won't run on Jupyter notebook...
            self.dview[u'p'] = pickle.dumps(p)
            self.dview.execute('p = pickle.loads(p)', block=True)

            self.dview.execute(
                'CGSubProb.SubProblems = {k: CGSubProb(p, k, %s, %s, %d, %f) for k in p.K}' % (self.use_jit, self.early_chk, self.maximum_num_chargings, self.minimum_dist),
                block=True)

        return SubProblems




    def AddBendersCut(self, pi, phi):
        cpx = self.cpx
        z = self.z
        y_i = self.y_i
        N_c = self.p.N_c

        cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind = [z] + [y_i(i) for i in N_c],
                    val = [1] + [-phi[i] for i in N_c]
                )
            ],
            senses=['L'],
            rhs=[pi],
            names=['benders_%d' % cpx.linear_constraints.get_num()]
        )

    def GetSolutionPaths(self, reachability_fn=None):
        if self.dview is None:
            return [sp.GetSolutionPath(reachability_fn) for sp in self.SubProblems]
        else:
            return self.dview.map_sync(lambda k: CGSubProb.SubProblems[k].GetSolutionPath(reachability_fn), self.p.K)

    def GetExpectedObj(self, reachability_fn=None):
        return sum([ed for _,_,d,ed in self.GetSolutionPaths(reachability_fn=reachability_fn)])


    def InternalSolve(self):

        p = self.p

        starttime = self.cpx.get_time()

        num_col = 0
        iter = 1
        while True:
            t1 = self.cpx.get_time()
            obj, sol_y = self.SolveMaster()
            self.p.benders_metric['time:bd:mp'] += self.cpx.get_time() - t1

            print ('Iteration %d, obj=%f, elapsed time=%f, total cols=%d :' 
                % (iter, obj, self.cpx.get_time()-starttime, num_col))
            pi = 0
            phi = {i:0 for i in p.N_c}

            # num_col = 0

            if self.dview is None:

                for sp in self.SubProblems:
                    # print '{}'.format(sp.k)
                    sp.FixCapacity(sol_y)
                    num_col_before = sp.cpx.variables.get_num()
                    pi_k, phi_k = sp.ColumnGeneration()
                    pi += pi_k
                    for i, val in phi_k.iteritems():
                        phi[i] += val

                    num_col += (sp.cpx.variables.get_num() - num_col_before)

            else:
                # parallel computing
                all_results = self.dview.map_sync(lambda k:CGSubProb.SolveSubProblem(sol_y,k), p.K)
                for pi_k, phi_k, num_col_k in all_results:
                    pi += pi_k
                    for i, val in phi_k.iteritems():
                        phi[i] += val

                    num_col += num_col_k


            self.p.benders_metric['num:cg:col'] = num_col


            sub_obj = pi + sum([phi[i] for i in sol_y])

            if abs(sub_obj - obj) > 0.001:
                self.p.benders_metric['num:bdcut'] += 1
                self.AddBendersCut(pi, phi)
            else:
                break

            iter += 1

        endtime = self.cpx.get_time()

        return {
                    'Solver' : 'Benders' + ('' if self.dview is None else '-Parallel'),
                    'time' : endtime - starttime,
                    'status' : self.cpx.solution.get_status(),
                    'status_msg' : self.cpx.solution.get_status_string(),
                    'objval' : self.cpx.solution.get_objective_value(),
                    'sol_design' : [i for i in self.p.N_c if self.cpx.solution.get_values(self.y_i(i)) > 0.5],
                    'benders_metric' : self.p.benders_metric,
                    'maximum_num_chargings': self.maximum_num_chargings,
                    'minimum_dist': self.minimum_dist
                }




class BendersCB(cplex.callbacks.LazyConstraintCallback):

    def __call__(self):

        obj = self.get_objective_value()
        inc_obj = self.get_incumbent_objective_value()

        if obj > inc_obj:
            cur_sol = self.get_values(self.solver.y_begin, self.solver.y_end)
            sol_y = [i for i,sol in zip(self.solver.p.N_c, cur_sol) if sol > 0.5]

            pi = 0
            phi = {i:0 for i in self.solver.p.N_c}

            num_col = 0

            if self.solver.dview is None:

                for sp in self.solver.SubProblems:
                    sp.FixCapacity(sol_y)
                    pi_k, phi_k = sp.ColumnGeneration()
                    pi += pi_k
                    for i, val in phi_k.iteritems():
                        phi[i] += val

                    num_col += sp.cpx.variables.get_num()

            else:
                # parallel computing
                all_results = self.solver.dview.map_sync(lambda k:CGSubProb.SolveSubProblem(sol_y,k), self.solver.p.K)
                for pi_k, phi_k, num_col_k in all_results:
                    pi += pi_k
                    for i, val in phi_k.iteritems():
                        phi[i] += val

                    num_col += num_col_k


            self.solver.p.benders_metric['num:cg:col'] = num_col

            sub_obj = pi + sum([phi[i] for i in sol_y])

            if abs(sub_obj - obj) > 0.001:
                print ('Adding a Benders cut with sub_z=%f < z=%f' % (sub_obj, obj))
                self.solver.p.benders_metric['num:bdcut'] += 1
                self.AddBendersCut(pi, phi)




    def AddBendersCut(self, pi, phi):
        z = self.solver.z
        y_i = self.solver.y_i
        N_c = self.solver.p.N_c

        self.add(
            constraint=cplex.SparsePair(
                ind = [z] + [y_i(i) for i in N_c],
                val = [1] + [-phi[i] for i in N_c]
            ),
            sense='L',
            rhs=pi
        )




class BnCSolver(BendersSolver):

    def InternalSolve(self):

        cpx = self.cpx

        starttime = self.cpx.get_time()


        try:
            callback = cpx.register_callback(BendersCB)
            callback.solver = self

            cpx.solve()

            endtime = self.cpx.get_time()

            print ('obj=%f, elapsed time=%f' % \
                (cpx.solution.get_objective_value(), endtime-starttime))

            self.p.benders_metric['time:bd:mp'] = endtime-starttime

        #except cplex.exceptions.CplexError, exc:
            print ('Solving Error:' + exc.message)




        # return {
        #             'Solver' : 'Benders' + '' if self.dview is None else '-Parallel',
        #             'time' : endtime - starttime,
        #             'status' : self.cpx.solution.get_status(),
        #             'status_msg' : self.cpx.solution.get_status_string(),
        #             'objval' : self.cpx.solution.get_objective_value(),
        #             'sol_design' : [i for i in self.p.N_c if self.cpx.solution.get_values(self.y_i(i)) > 0.5],
        #             'benders_metric' : self.p.benders_metric
        #         }






    # if __name__ == '__main__':


    # from ipyparallel import Client
    # rc = Client()
    # dview = rc[:]
    # # # dview.use_dill()
    # dview.execute('from evuu import *', block=True)

    # p = Prob(
    #     dataDir='25NODE',
    #     veh_range=6,
    #     mid_node_range=1,
    #     OD_num_limit=-1,
    #     B=6,
    #     reachability_fn=Reachability.GaussianCDF,
    #     dview=dview
    # )

    # #
    # # bc = BnCSolver(p)
    # # bc.InternalSolve()

    # # %%prun
    # bs = BendersSolver(
    #     p,
    #     None,
    #     use_jit=True,
    #     early_chk=False
    # )
    # bs.Solve()

    # print bs.Solve()

    # # pbs = BendersSolver(p, dview, use_jit=True, early_chk=False)
    # # print pbs.Solve()

    # # ms = MIPSolver(p)
    # # print ms.Solve()