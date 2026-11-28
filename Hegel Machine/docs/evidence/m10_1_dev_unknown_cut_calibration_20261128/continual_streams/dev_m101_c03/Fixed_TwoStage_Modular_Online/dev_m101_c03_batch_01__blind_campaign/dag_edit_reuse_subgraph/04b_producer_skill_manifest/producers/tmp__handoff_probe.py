import itertools
import pickle
import runpy
import numpy as np
from scipy.optimize import least_squares

s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
sink = runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']
def aff(u): return np.column_stack((np.ones(len(u)), u))
def fit(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def rms(a): return np.sqrt(np.mean(a*a))
a,b,c = (s[k] for k in ('panel_a','panel_b','panel_c'))
A=aff(a['u']); av=a['v']; co=fit(A,av)
for _ in range(12):
    r=np.linalg.norm(av-A@co,axis=1)
    w=1/np.sqrt(1+(r/max(np.median(r),1e-5))**2)
    co=fit(A*w[:,None],av*w[:,None])
r=np.linalg.norm(av-A@co,axis=1)
keep=np.argsort(r)[:-3]; co=fit(A[keep],av[keep]); print('A sorted tail',np.sort(r)[-7:],'clean RMS',rms(av[keep]-A[keep]@co))
x=s['train']['x']; u=np.linalg.solve(co[1:].T,(x-co[0]).T).T
ux=c['u']; iu,ju=np.triu_indices(4)
def square(u): return np.column_stack((aff(u),u[:,iu]*u[:,ju]))
p=fit(square(ux),c['v1']); print('v1 coefficients',np.round(p,4))
f1=square(u)@p

def cosbasis(u,w): return np.column_stack((aff(u), np.cos(u*w)))
def optimize(us,vs,start,shared):
    def r(w):
        m=cosbasis(us,w)
        return (m@fit(m,vs)-vs).ravel()
    return least_squares(r,np.full(1 if shared else 4,start),bounds=(.1,7),gtol=1e-12).x
for shared in (True,False):
    for start in (1.,2.5,4.):
        w=optimize(ux,c['v2'],start,shared); m=cosbasis(ux,w)
        print('v2 cosine',w,'RMS',rms(m@fit(m,c['v2'])-c['v2']))
for shared in (True,False):
    loo=[]
    for i in range(len(ux)):
        k=np.arange(len(ux))!=i; w=optimize(ux[k],c['v2'][k],1.,shared)
        loo.append(cosbasis(ux[i:i+1],w)@fit(cosbasis(ux[k],w),c['v2'][k])-c['v2'][i])
    print('v2 nested LOO',shared,rms(np.array(loo)))
w=optimize(ux,c['v2'],1.,True); f2=cosbasis(u,w)@fit(cosbasis(ux,w),c['v2']); f2q=square(u)@fit(square(ux),c['v2']); f2a=aff(u)@fit(aff(ux),c['v2']); fb=aff(u)@fit(aff(b['u']),b['v'])
y=s['train']['y']; tr=s['train']['traj']; t=s['eval_times']; allidx=np.arange(len(y)); folds=[(allidx[:-20],allidx[-20:])]+[(np.setdiff1d(allidx,g),g) for g in np.array_split(allidx,4)]
results=[]
for typ, z in [('cos',f2),('quad',f2q),('aff',f2a)]:
    for left,right,label in [(f1,z,'v1v2'),(z,f1,'v2v1'),(fb,z,'bv2'),(f1,fb,'v1b')]:
        mixed=(left[:,:,None]*right[:,None,:]).reshape(len(u),16)
        choices={'u':u,'b':fb,'1':f1,'2':z}
        for n in range(1,5):
            for keys in itertools.combinations(choices,n):
                linear=np.column_stack((np.ones(len(u)),*(choices[k] for k in keys)))
                scores=[]
                for train,test in folds:
                    idx=np.r_[train,test]
                    out=sink({'node_00':dict(n=len(train),linear=linear[idx],mixed=mixed[idx],eval_times=t,y=y[train],traj=tr[train])})
                    scores.append((np.mean(abs(out['y_hat']-y[test])),np.mean(abs(out['state_pred']-tr[test]))))
                results.append((np.max(scores),typ,label,''.join(keys),scores[0]))
for rec in sorted(results)[:18]: print('worst, v2 basis, mixed order, linear blocks, official:',rec)
