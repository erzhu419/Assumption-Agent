exec(open('/tmp/probe_facility.py').read().split('start=time.perf_counter()')[0])
K=[scores(c.min(axis=0))]; B=K[0].copy(); B[~feasible]=np.inf
ub=np.inf; solved=np.zeros(len(B),bool)
while True:
    i=int(np.argmin(np.where(solved,np.inf,B)))
    if B[i]>=ub-1e-8: break
    s=lp(i); solved[i]=True; ub=min(ub,s.fun+fixed[i]); K.append(scores(s.eqlin.marginals)); B=np.maximum(B,K[-1])
K=np.asarray(K); K[:,~feasible]=np.inf
all_masks=np.arange(1<<m)
def compact(order, mode):
    def descend(j, ids):
        if j==m: return 1
        if mode=='cut' and np.max(np.min(K[:,ids],axis=1))>=ub-1e-7: return 1
        if mode=='portfolio' and np.min(B[ids])>=ub-1e-7: return 1
        bit=1<<order[j]; on=(ids&bit)!=0
        return descend(j+1,ids[~on])+descend(j+1,ids[on])
    return descend(0,all_masks)
rng=np.random.default_rng(114)
orders=[np.arange(m),np.argsort(f),np.argsort(-f),np.argsort(a),np.argsort(-a)] + [rng.permutation(m) for _ in range(20)]
print('single-cut leaf counts:', [compact(o,'cut') for o in orders]); print('portfolio leaf count:',compact(orders[0],'portfolio')); print('root single-cut gap:',ub-np.max(np.min(K,axis=1))); print('root portfolio gap:',ub-B.min())
