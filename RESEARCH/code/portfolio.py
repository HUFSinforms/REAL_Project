import pandas as pd
import time,datetime
import numpy as np
from pandas import *
from informs import portfolio
import class_informs as inform

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import math
import random as ran
from scipy.stats import truncnorm
#import seaborn as sbn

import time,datetime
import cmath


import re
import matplotlib.pyplot as plt

def cal_obj1(sol, risk_mat, as_) :

    a = np.full((len(risk_mat[0]), len(risk_mat[0])), 10000)
    risk_mat = risk_mat * a
    s_rr = np.dot(sol, risk_mat)
    s_r = np.dot(s_rr, sol)
    s_a = np.dot(sol, as_*10000)
    obj = s_r-s_a
    return obj

def cal_obj(sol, risk_mat, as_) :

    a = np.full((len(risk_mat[0]), len(risk_mat[0])), 10000)
    risk_mat = risk_mat * a
    sol_d = list(np.array(sol) - np.array(using_dic['bw']))
    s_rr = np.dot(sol_d, risk_mat)
    s_r = np.dot(s_rr, sol_d)
    s_a = np.dot(sol_d, as_*10000)
    obj = s_r-s_a
    return obj



def select_candi(sol) :
    
    diff_sub = np.array([0]*len(as_), float)
    diff_add = np.array([0]*len(as_), float)
    candi = []

    for i in range(len(sol)) :
        if(sol[i] > 0) :
            w = sol[i]
            sol[i] = 0
            diff_sub[i] = cal_obj(sol, risk_mat, as_)
            if min(diff_sub[diff_sub>0]) == diff_sub[i] :
                for j in range(len(sol)) :
                    if sol[j] == 0 :
                        sol[j] = w
                        diff_add[j] = cal_obj(sol, risk_mat, as_)
                        sol[j] = 0
            sol[i] = w
        
    sol[np.where(diff_sub == min(diff_sub[diff_sub>0]))[0][0]] = 0
    sol[np.where(diff_add == min(diff_add[diff_add>0]))[0][0]] = 0.1

    for val in range(len(sol)) :
        if sol[val] > 0. :
            candi.append(val)
    return candi



def update(sol,lim_time,as_,w_pre,O_scale) :

    kkset=time.time()
    ex = inform.informs(10000, 10000, w_pre, O_scale,dic_sector,dic_bench,risk_sedol,dic_MCAP,dic_beta,alpha,risk_mat)
    ex.set_omega(risk_mat)

    new_list = []

    con = []
    nums = 0
    for k in risk_sedol :
        new_list.append(sol[k])
        if sol[k] == 1:
            con.append(nums)
        nums += 1
    result_list = []
    result_list2 = []
    start_time = time.time()
    
    while(True):

        if time.time() - start_time > lim_time:
            break

        ex.set_con(cons=con)
        print('kkset time',time.time()-kkset)
      
        new_list2 = []
        new_list3 = []

        try:
            kkcplex=time.time()
            aabbaa = ex.solve(0.005)

            print('clexsolve time : ',time.time()-kkcplex)
            cplex_t=time.time()-kkcplex
            
            if aabbaa ==1234:
                break

            if aabbaa != 0:
                aaaa = aabbaa[0]
                bbbb= aabbaa[1]
                for k in risk_sedol :
                    new_list2.append(aaaa[k])
                for k in risk_sedol :
                    new_list3.append(bbbb[k])
                new_list = new_list2

                result_list.append(aabbaa[2])
                result_list2.append(aabbaa[0])
                print("obj_val")
                print(cal_obj1(new_list3, risk_mat, as_))
                con = select_candi(new_list)
                print("cplex_val")
                print(aabbaa[2])

            else:
                print("TE infeasible")
                con = select_candi(new_list)


        except:
            print("cplex no solution")
            con = select_candi(new_list)
            cplex_t=0


    return result_list,result_list2,cplex_t

def update2(sol,lim_time,as_,w_pre,O_scale) :
    start_time = time.time()
    
    
    
    
    ex = inform.informs(10000, 10000, w_pre, O_scale,dic_sector,dic_bench,risk_sedol,dic_MCAP,dic_beta,alpha,risk_mat)
    ex.set_omega(risk_mat)

    new_list = []

    con = []
    nums = 0
    for k in risk_sedol :
        new_list.append(sol[k])
        if sol[k] == 1:
            con.append(nums)
        nums += 1
    result_list = []
    result_list2 = []

    while(True):

        if time.time() - start_time > lim_time:
            break
        
        ex.set_con(cons=con)
        new_list2 = []
        new_list3 = []

        aabbaa = ex.solve(0.01)


        if aabbaa != 0:
            aaaa = aabbaa[0]
            bbbb= aabbaa[1]
            for k in risk_sedol :
                new_list2.append(aaaa[k])
            for k in risk_sedol :
                new_list3.append(bbbb[k])
            new_list = new_list2

            result_list.append(aabbaa[2])
            result_list2.append(aabbaa[0])
            con = select_candi(new_list)


        else:
            print("TE infeasible")
            con = select_candi(new_list)





    return result_list,result_list2


def GAN_reulst(best_sols, sols,fea_sample_obj_values):    
    good_sols=np.argsort(fea_sample_obj_values)[:10]
    card_list=[list(np.where(sols[i]>0.001)[0]) for i in range(len(good_sols))]
    card_freq=np.zeros(nbasset)
    for i in range(len(card_list)):
        for j in card_list[i]:
            card_freq[j]+=1
    best=np.argsort(sum(sols))[::-1][:len(card_freq[card_freq>len(card_list)-5])]
    bestsol=np.where(best_sols[0]>0.001)[0]
    return best,bestsol

def init_cplex(risk_mat,risk_sedol):
    sedol_var_list =[]
    for i in risk_sedol:
        sedol_var_list.append("d"+str(i))

    for i in risk_sedol:
        sedol_var_list.append("q"+str(i))

    sedol_var_list.append("assum")
    #global alpha
    alpha = []
    for i in risk_sedol:
        alpha.append(-10000.*dic_sedol_as[i])

    qmat=[]
    for i in range(len(risk_mat)):
        qmat_1=[]
        qmat_1.append(sedol_var_list)
        new_risk_mat=[]
        for j in risk_mat[i]:
            new_risk_mat.append(20000*j)
        for j in risk_mat[i]:
            new_risk_mat.append(0)
        new_risk_mat.append(0)
        qmat_1.append(new_risk_mat)
        qmat.append(qmat_1)

    for i in range(len(risk_mat)):
        qmat_1=[]
        qmat_1.append(sedol_var_list)
        new_risk_mat=[]
        for j in risk_mat[i]:
            new_risk_mat.append(0)
        for j in risk_mat[i]:
            new_risk_mat.append(0)
        new_risk_mat.append(0)
        qmat_1.append(new_risk_mat)
        qmat.append(qmat_1)

    for i in range(1):
        qmat_1=[]
        qmat_1.append(sedol_var_list)
        new_risk_mat=[]
        for j in risk_mat[i]:
            new_risk_mat.append(0)
        for j in risk_mat[i]:
            new_risk_mat.append(0)
        new_risk_mat.append(0)
        qmat_1.append(new_risk_mat)
        qmat.append(qmat_1)

    q_con1 = []
    q_con2 = []
    q_val = []


    for i in range(len((risk_mat[0]))):
        for j in range(len(risk_mat[0])):
            if j >= i:
                q_con1.append(i)
                q_con2.append(j)
                if i == j:
                    ex_list = list(risk_mat[i])
                    q_val.append(0.5*ex_list[j]*10000)
                else:
                    ex_list = list(risk_mat[i])
                    q_val.append(ex_list[j]*10000)

    Q_con = []
    Q_con.append(q_con1)
    Q_con.append(q_con2)
    Q_con.append(q_val)
    
    return Q_con, qmat, alpha

print("complete")