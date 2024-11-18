import gurobipy as gp
import numpy as np
 
def best_dist_dual(n, m, inter_cand_dist_vals, weight_vals):
    """
    given a number of distinct voter locations (n) candidates (m) possible inter-candidate distance values (inter_cand_dist_vals), and possibly a dictionary of inter-candidate distances (inter_cand_dist_vec), solves the QCIP written in the overleaf. 

    """
    V = list(range(n)) 
    C = ['c' + str(i) for i in range(m)]
    O = C[0] #fixed optimal candidate



    model = gp.Model()
    d = model.addVars(C + V, V, lb=0, name='d')
    D = model.addVars(C, C, lb=0, name='D')
    model.addConstr(D['c0', 'c1'] == 0)
    model.addConstr(D['c1', 'c2'] == 1)
    model.addConstr(d['c0',0] == .5)
    model.addConstr(d['c2',0] == 1)
    model.addConstr(d['c0',1] == .5)
    model.addConstr(d['c2',1] == .5)
    psi = model.addVar(lb = 0, name='psi')
    w = model.addVars(V, lb=0, name='w', vtype=gp.GRB.INTEGER)

    model.addConstr(sum(w[v] for v in V) <= 2*n)
    
    x = model.addVars(C, V, range(1, m+1), name="x", vtype=gp.GRB.BINARY) # x[i, j, r] = 1 iff i = alt(j, r)
    z = model.addVars(V, C, C, name="z", vtype=gp.GRB.BINARY) # z[j, i1, i2] = 1 if rank(i1) < rank(i2)

    model.addConstr(psi <= min(3 - 2/m, 2.74))

    #constraints to enforce definition of x and z

    for j in V:
        for i in C:
            model.addConstr(sum(x[i, j, r] for r in range(1, m+1)) == 1)
        for r in range(1, m+1):
            model.addConstr(sum(x[i, j, r] for i in C) == 1)

    for j in V:
        for i1 in C:
            for i2 in C:
                if i1 == i2:
                    continue
                model.addConstr(z[j, i1, i2] + z[j, i2, i1]  == 1)
                            
                for r in range(1, m):
                    model.addConstr(z[j, i1, i2] >= sum(x[i1, j, q] for q in range(1, r + 1)) - sum(x[i2, j, q] for q in range(1, r + 1))) 

    # Constraints in QCIP
    for c in C:
        model.addConstr(psi <= sum(d[c, v] * w[v] for v in V))

    model.addConstr(sum(d[O, v] * w[v] for v in V) <= 1)

    for v in V:
        for c1 in C:
            for c2 in C:
                model.addConstr( z[v, c1, c2] * d[c1, v] <= d[c2, v])
    
    for v1 in V:
        for v2 in V:
            for v3 in V:
                model.addConstr(d[v1, v2] <= d[v1, v3] + d[v2, v3])
    
    for v1 in V:
        for v2 in V:
            for c in C:
                model.addConstr(d[c,v1] <= d[c, v2] + d[v1, v2])
    
    for v in V:
        for c in C:
            for k in C:
                    model.addConstr( D[c,k] <= d[c, v] + d[k, v])
                    model.addConstr(d[c, v] <=  D[c, k] + d[k, v])

    
    # Encode that D[c1, c2] can only take certain values
    y = model.addVars(C, C, inter_cand_dist_vals, vtype=gp.GRB.BINARY)
    for c1 in C:
        for c2 in C:
            if c1 == c2:
                continue 
            model.addConstr(sum(y[c1, c2, r] for r in inter_cand_dist_vals) == 1)
            for r in inter_cand_dist_vals:
                model.addConstr(y[c1, c2, r] * D[c1, c2] == y[c1, c2, r] * r)
            for c3 in C:
                if c3 == c1 or c3 == c2:
                    continue 
                model.addConstr(D[c1, c2] <= D[c2, c3] + D[c1, c3])

    # Encode that weights can only take certain values
    wt = model.addVars(V, weight_vals, name="wt", vtype=gp.GRB.BINARY)
    for v in V:
        model.addConstr(sum(wt[v,r] for r in weight_vals) == 1)
        for r in weight_vals:
            model.addConstr(wt[v,r] * w[v] == wt[v,r] * r)



    # Enforce symmetry of distance metrics
    for v1 in V:
        for v2 in V:
            model.addConstr(d[v1, v2] == d[v2, v1])
    for c1 in C:
        for c2 in C:
            model.addConstr(D[c1, c2] == D[c2, c1])

    model.setObjective(psi, gp.GRB.MAXIMIZE)
    #model.setParam('MIPFocus', 1)
    model.update()
    model.optimize()
    # model.computeIIS()
    # print('\nThe following constraints and variables are in the IIS:')
    # for c in model.getConstrs():
    #     if c.IISConstr: print(f'\t{c.constrname}: {model.getRow(c)} {c.Sense} {c.RHS}')

    # for v in model.getVars():
    #     if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
    #     if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')
    all_vars = model.getVars()
    values = model.getAttr("X", all_vars)
    names = model.getAttr("VarName", all_vars)
    return dict(zip(names, values)), model.ObjVal   



def x_str(i, j, r):
    return "x[f" + str(i) + "," + str(j) + "," + str(r) + "]"
  





result, obj = best_dist_dual(10, 3, [0, 2, 4], [0, 1, 2])

for key, val in result.items():
    print(key, val)
    if key[0] == 'n':
        print(key, val)
for key, val in result.items():
    if key[0] == "w":
        if val > 0:
            print(key, val)
print(obj)