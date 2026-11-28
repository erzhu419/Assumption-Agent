from probe import *
import runpy
from scipy.optimize import least_squares, minimize_scalar
m=runpy.run_path('/holdout_task/batch_solver.py'); x=m['_inverse'](s['train']['x'],m['_calibration'](s['panel_a'])); z=s['train']['traj']; y=s['train']['y']; t=s['eval_times']; p=s['panel_c']; U=p['u']; Y=p['v1'];
fun={'sin':np.sin,'tanh':np.tanh,'rat':lambda a:a/(1+a*a),'gauss':lambda a:a*np.exp(-a*a),'cos':np.cos,'abs':np.abs}
I=np.arange(52); J=np.arange(52,72)
B=m['_basis'](t,z[I]); T=solve(B,z.T).T
for label,X,V,ii,jj in [('traj',x,T,I,J),('C',U,Y,np.arange(18),np.arange(18,24))]:
 results=[]
 for name,g in fun.items():
  for base in [aff,pp]:
   def f(a):return np.c_[base(X),g(X*a)]
   def err(a):
    A=f(a);return (A[ii]@solve(A[ii],V[ii])-V[ii]).ravel()
   fs=np.linspace(.2,15,150); es=np.array([np.linalg.norm(err(a)) for a in fs]); idx=np.flatnonzero((es[1:-1]<es[:-2])&(es[1:-1]<es[2:]))+1; idx=np.unique(np.r_[idx,es.argmin()]); starts=sorted(idx,key=lambda j:es[j])[:5]; rrs=[]
   for k in starts:
    a=minimize_scalar(lambda a:np.linalg.norm(err(a)),bounds=(max(.1,fs[k]-.12),fs[k]+.12),method='bounded').x
    for a1 in [a,least_squares(err,np.full(4,a),bounds=(.08,25.),max_nfev=120).x]:
     F=f(a1); W=solve(F[ii],V[ii]); e=F[jj]@W-V[jj]; rrs.append((np.std(err(a1)),np.mean(abs(e@B.T)) if label=='traj' else np.mean(abs(e)),a1,W))
   rrs.sort(key=lambda a:a[0]); a,b,c,W=rrs[0]; results.append((a,b,name,base.__name__,c,W))
 for a,b,name,base,c,W in sorted(results,key=lambda l:l[0])[:7]: print(label,name,base,a,b,c,'W',W[-4:])
for g in [lambda x:x,lambda x:x*x]:
 F=np.c_[pp(x),g(np.sin(8*x))]; W=solve(F[I],y[I]); print('scalar coefficients',W,'fit',np.std(F[I]@W-y[I]),'test',np.mean(abs(F[J]@W-y[J])))
