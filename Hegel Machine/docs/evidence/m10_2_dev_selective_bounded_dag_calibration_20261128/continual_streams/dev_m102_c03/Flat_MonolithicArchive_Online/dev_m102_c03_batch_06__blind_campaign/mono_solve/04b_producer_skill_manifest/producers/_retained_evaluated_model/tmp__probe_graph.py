import pickle,runpy,numpy as np
from itertools import combinations,combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; Y=np.c_[tr['y'],tr['traj']]; n=52
z=runpy.run_path('/tmp/hegel_continual_archive/dev_m102_c03_batch_05_solver.py')['_calibrate'](s['panel_a'],tr['x'])
fit=lambda a,b: np.linalg.lstsq(a,b,rcond=1e-9)[0]; aff=lambda q:np.c_[np.ones(len(q)),q]
cyc=lambda q:q*np.roll(q,-1,axis=1)
ops={}
for k,p,out in [('b','panel_b','v'),('c','panel_c','v1'),('d','panel_c','v2')]:
 u=s[p]['u']; v=s[p][out]; pp=list(combinations(range(4),2)); pair=lambda q:np.column_stack([q[:,i]*q[:,j] for i,j in pp]); cand=[]
 for name,f in [('id',lambda q,w:q),('cycle',lambda q,w:cyc(q)),('pair',lambda q,w:pair(q)),('sq',lambda q,w:q*q),('lincycle',lambda q,w:np.c_[q,cyc(q)]),('cos',lambda q,w:np.cos(q*w)),('sin',lambda q,w:np.sin(q*w)),('soft',lambda q,w:q/(1+w*abs(q))),('tanh',lambda q,w:np.tanh(w*q)),('gauss',lambda q,w:np.exp(-w*q*q)),('rat',lambda q,w:q/(1+w*q*q))]:
  for lin in ([0,1] if name in ['cos','sin','soft','tanh','gauss','rat'] else [0]):
   def design(q,w,f=f,lin=lin): return aff(np.c_[q,f(q,w)] if lin else f(q,w))
   def err(w):
    A=design(u,w); Q=np.linalg.qr(A)[0]; return ((v-Q@(Q.T@v))/np.maximum(1-(Q*Q).sum(axis=1),.01)[:,None]).ravel()
   for st in [.6,2.4,3.7] if name in ['cos','sin','soft','tanh','gauss','rat'] else [1.]:
    w=least_squares(err,[st],bounds=(.05,5)).x; co=fit(design(u,w),v); cand.append((np.sqrt(np.mean(err(w)**2)),name,lin,w,design,co))
 cand.sort(key=lambda a:a[0]); print(k,[(r[0],r[1],r[2],r[3]) for r in cand[:8]],flush=True)
 a=next((r for r in cand if r[0] < 1.3*cand[0][0] and r[2]==0),cand[0]); ops[k]=lambda q,a=a:a[4](q,a[3])@a[5]
np.savez('/tmp/graph_data.npz',z=z,Y=Y,**{''.join(i):ops[i[-1]](z if len(i)==1 else ops[i[0]](z)) for i in [(k,) for k in ops]+[(k,l) for k in ops for l in ops]})
D=dict(np.load('/tmp/graph_data.npz')); D.pop('Y'); feats=[]
for a,b in combinations(D,2):
 A,B=D[a],D[b]; cross=(A[:,:,None]*B[:,None,:]).reshape(len(z),16)
 for typ,X in [('lin',np.c_[A,B]),('dot',np.c_[A,B,np.sum(A*B,1)]),('diag',np.c_[A,B,A*B]),('full',np.c_[A,B,cross])]:
  M=aff(X); pred=M[n:]@fit(M[:n],Y[:n]); loss=abs(pred-Y[n:]); feats.append((loss[:,0].mean(),loss[:,1:].mean(),a,b,typ))
for k in range(2): print('rank',k, sorted(feats,key=lambda a:a[k])[:16])
