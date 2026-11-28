import sys,pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import hyp2f1
sys.path.insert(0,'/holdout_task'); import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=m._coordinates(s,tr['x']); y,z,t=tr['y'],tr['traj'],s['eval_times']; pair=np.triu_indices(4,1)
u=s['panel_c']['u']; D=np.c_[np.ones(len(u)),u]; np.set_printoptions(precision=4,suppress=True,linewidth=140)
print('panel C support',u.min(0),u.max(0))
for k in ('v1','v2'):
 v=s['panel_c'][k]; W=np.linalg.lstsq(D,v,rcond=None)[0]; print(k,'linear coefficients',W.T,'residual RMS',np.std(v-D@W,axis=0)); print('sample',np.c_[u,v][:5]);
 for name,F in [('atan',np.arctan(2*u)),('asinh',np.arcsinh(3*u)),('tanh',np.tanh(2*u)),('square',u*u),('fullquad',np.c_[u*u,u[:,pair[0]]*u[:,pair[1]]])]:
  G=np.c_[D,F]; P=G@np.linalg.pinv(G); print(name,np.sqrt(np.mean(((P@v-v)/(1-np.diag(P))[:,None])**2,axis=0)))
splits=[np.arange(72),np.argsort(np.linalg.norm(x,axis=1))]+[np.argsort(sign*x[:,j]) for j in range(4) for sign in (1,-1)]
splits += [np.random.default_rng(692+k).permutation(72) for k in range(4)]
for family in ('asinh','atan','power'):
 for form in ('add','addlin','mix','mixlin'):
  for reg in (.003,.02,.08):
   errors=[]
   for idx in splits:
    i,j=idx[:52],idx[52:]; b=m._time_fit(t,z[i]); q=np.c_[y,z@np.linalg.pinv(b)]; sd=q[i].std(0); Q=q[i]/sd
    def predict(p,ret=False):
     a=x*np.exp(p[0])+p[1:5];h=np.arcsinh(a) if family=='asinh' else np.arctan(a) if family=='atan' else a*hyp2f1(.5,p[5],1.5,-a*a)
     G=np.c_[np.ones(len(x)),h,h*h]; D=np.c_[G,h[:,pair[0]]*h[:,pair[1]]];
     if 'lin' in form: G=np.c_[G,x]; D=np.c_[D,x]
     if 'mix' in form: G=np.c_[G,h[:,pair[0]]*h[:,pair[1]]]
     pp=[]; rr=[]
     for F,V in ((D,Q[:,:1]),(G,Q[:,1:])):
      L=np.diag(np.r_[0.,np.full(F.shape[1]-1,reg)])
      w=np.linalg.lstsq(np.r_[F[i],L],np.r_[V,np.zeros((len(L),V.shape[1]))],rcond=None)[0]; pp.append(F@w); rr.append((L@w).ravel())
     P=np.column_stack(pp)
     if ret:return P*sd
     return np.r_[(P[i]-Q).ravel(),*rr, .07*p[1:5]]
    p0=[np.log(3.),0,0,0,0]+([.5] if family=='power' else []); lo=[-1.5]+[-.45]*4+([.20] if family=='power' else []);hi=[1.9]+[.45]*4+([1.3] if family=='power' else [])
    opt=least_squares(predict,p0,bounds=(lo,hi),max_nfev=75); P=predict(opt.x,True); E=np.abs(P[j,1:]@b-z[j]).mean(0);errors.append([np.mean(abs(P[j,0]-y[j])),E.max()]);
   e=np.array(errors);print(family,form,reg,'ord/rad',e[:2].ravel(),'axis median/worst',np.median(e[2:10],0),e[2:10].max(0),'random worst',e[10:].max(0),flush=True)
