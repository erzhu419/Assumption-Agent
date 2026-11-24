import numpy as np
from scipy.optimize import linprog

def solve_and_certify_facilities(instance):
    m, n = int(instance['m']), int(instance['n'])
    f=np.asarray(instance['fixed_cost'],float); cap=np.asarray(instance['capacity'],float)
    d=np.asarray(instance['demand'],float); c=np.asarray(instance['cost_matrix'],float)
    nv=m+m*n; obj=np.r_[f,c.ravel()]
    Aeq=np.zeros((n,nv));
    for j in range(n):
        for i in range(m): Aeq[j,m+i*n+j]=1
    Aub=np.zeros((m,nv))
    for i in range(m): Aub[i,i]=-cap[i]; Aub[i,m+i*n:m+(i+1)*n]=1
    def lp(C,O):
        bounds=[(0,0) if i in C else ((1,1) if i in O else (0,1)) for i in range(m)]+[(0,None)]*(m*n)
        return linprog(obj,A_ub=Aub,b_ub=np.zeros(m),A_eq=Aeq,b_eq=d,bounds=bounds,method='highs')
    best=float('inf'); besty=bestx=None
    for mask in range(1<<m):
        O={i for i in range(m) if mask>>i&1}; C=set(range(m))-O
        if sum(cap[i] for i in O)+1e-10 < sum(d): continue
        r=lp(C,O)
        if r.success and r.fun < best:
            best=float(r.fun); besty=[int(i in O) for i in range(m)]; bestx=r.x[m:].reshape(m,n)
    if besty is None: raise ValueError('no feasible facility subset')
    leaves=[]
    def visit(C,O):
        if sum(cap[i] for i in set(range(m))-C)+1e-10 < sum(d):
            leaves.append((frozenset(C),frozenset(O),float('inf'))); return
        r=lp(C,O)
        if not r.success:
            leaves.append((frozenset(C),frozenset(O),float('inf'))); return
        lb=float(r.fun)
        if lb >= best-1e-8 or len(C)+len(O)==m:
            leaves.append((frozenset(C),frozenset(O),lb)); return
        i=next(i for i in range(m) if i not in C and i not in O)
        visit(C|{i},O); visit(C,O|{i})
    visit(set(),set())
    # Compact only exact sibling pairs; parent bound is the minimum of valid child bounds.
    changed=True
    while changed:
        changed=False
        for i in range(m):
            groups={}
            for k,(C,O,L) in enumerate(leaves):
                if i in C: key=(C-{i},O); side=0
                elif i in O: key=(C,O-{i}); side=1
                else: continue
                groups.setdefault(key,{})[side]=(k,L)
            pairs=[(key,g) for key,g in groups.items() if len(g)==2]
            if pairs:
                remove=set()
                for (C,O),g in pairs:
                    (a,la),(b,lb)=g[0],g[1]; remove.update((a,b))
                    leaves.append((frozenset(C),frozenset(O),min(la,lb)))
                leaves=[x for k,x in enumerate(leaves) if k not in remove]
                changed=True
    proof=[{'fixed_closed':sorted(C),'fixed_open':sorted(O),'lower_bound':float(L)} for C,O,L in leaves]
    return {'open_y':besty,'flows_X':bestx,'upper_bound':best,'proof_leaves':proof}
