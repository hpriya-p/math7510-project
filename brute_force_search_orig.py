import gurobipy as gp
import numpy as np
 
def best_dist_dual(n, warmstart_n = None, write_only=False):
    """
    given a number of voters n, 
    """
    C = list(range(n))
    F = ['f' + str(i) for i in range(n)]
    M = 3 # since gamma <= 3, d[i, j, o] <= 3 for all i, all j, all, o


    try:
        print("Reading...")
        model = gp.read(str(n) + ".mps")
        print("Reading complete")
    except:
        print("Building...")
        model = gp.Model()

        d = model.addVars(F + C, F + C, F, lb=0, name='d')
        gamma = model.addVar(lb=0, name='gamma')
        x = model.addVars(F, C, range(1, n+1), name="x", vtype=gp.GRB.BINARY) # x[i, j, r] = 1 iff i = alt(j, r)
        z = model.addVars(C, F, F, name="z", vtype=gp.GRB.BINARY) # z[j, i1, i2] = 1 if rank(i1) < rank(i2)
        # constraints to enforce definition of x
        model.addConstr(gamma <= 2.74)    
        for j in C:
            for i in F:
                model.addConstr(sum(x[i, j, r] for r in range(1, n+1)) == 1)
            for r in range(1, n+1):
                model.addConstr(sum(x[i, j, r] for i in F) == 1)

         

        for j in C: 
            for i1 in F: 
                for i2 in F:
                    if i1 == i2:
                        continue
                    model.addConstr(z[j, i1, i2] + z[j, i2, i1]  == 1)
                           
                    for r in range(1, n):
                        model.addConstr(z[j, i1, i2] >= sum(x[i1, j, q] for q in range(1, r + 1)) - sum(x[i2, j, q] for q in range(1, r + 1))) 
                        
        model.addConstr(sum(d[o, j, o] for o in F for j in C) <= 1)
        for i in F:
            model.addConstr(gamma - sum(d[i, j, o] for o in F for j in C) <= 0)
        
        for j in C:
            for r in range(1, n):
                for o in F:
                    for i in F:
                        for k in F:
                            model.addConstr( d[i, j, o] <= d[k, j, o] +  M * (1 - z[j, i, k]))
        for i in C + F:
            for j in C + F:
                for k in C + F:
                    for o in F:
                        model.addConstr(d[i, j, o] <= d[i, k, o] + d[j, k, o])

        for i in C + F:
            for j in C + F:
                for o in F:
                    model.addConstr(d[i, j, o] == d[j, i, o])
        
        for i in C + F:
            for o in F:
                model.addConstr(d[i, i, o] == 0)

      
        model.setObjective(gamma, gp.GRB.MAXIMIZE) 
    model.setParam('MIPFocus', 1)
    model.update()
    model.optimize()
    all_vars = model.getVars()
    values = model.getAttr("X", all_vars)
    names = model.getAttr("VarName", all_vars)
    return dict(zip(names, values)), model.ObjVal   



def x_str(i, j, r):
    return "x[f" + str(i) + "," + str(j) + "," + str(r) + "]"
  





result, obj = best_dist_dual(4, None)
for key, val in result.items():
    if key[0] == "x":
        if val > 0:
            print(key, val)
for key, val in result.items():
    if key[0] == "z":
        if val > 0:
            print(key, val)
print(obj)
