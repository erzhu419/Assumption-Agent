import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def affine(u):return np.c_[np.ones(len(u)),u]
def operators(s):
 p=s['panel_a']; u=p['u']; v=p['v']; f=affine(u); keep=np.linalg.norm(v-f@ls(f,v),axis=1)<2
 a=ls(f[keep],v[keep]); a[1:]=np.diag(np.diag(a[1:])); a[0]=(v[keep]-u[keep]@a[1:]).mean(0)
 p=s['panel_b']; b=ls(affine(p['u']),p['v']); return a,b
def coords(x,s):
 a,b=operators(s); A=lambda x:affine(x)@a; B=lambda x:affine(x)@b; I=lambda x:(x-a[0])/np.diag(a[1:]); J=lambda x:(x-b[0])@np.linalg.inv(b[1:]); return {'raw':x,'A':A(x),'Ai':I(x),'B':B(x),'BA':B(A(x)),'BAi':B(I(x)),'AB':A(B(x)),'AiB':I(B(x)),'Bi':J(x),'BiA':J(A(x)),'BiAi':J(I(x))}
def basis(t,q):return np.column_stack((-np.expm1(-q[0]*t),np.exp(-q[2]*t)*np.sin(q[1]*t)))
def modes(t,z):
 def res(q):
  f=basis(t,q);return (f@ls(f,z.T)-z.T).ravel()
 q=least_squares(res,[.55,1.13,.18],bounds=([.02,.5,-.3],[3,2,1])).x; return basis(t,q)
def powers(x,n):return np.column_stack([np.prod(x[:,p],axis=1) for p in combinations_with_replacement(range(x.shape[1]),n)])
if __name__=='__main__':
 np.set_printoptions(precision=4,suppress=True,linewidth=150)
 c=S['panel_c']; u=c['u']; x=S['train']['x']; y=S['train']['y']; z=S['train']['traj']; t=S['eval_times']; mo=modes(t,z); Y=np.c_[y,ls(mo,z.T).T]; print('mode coefficients std',Y.std(0)); print('raw linear coeff',ls(affine(x),Y)); print('quadratic coeff',ls(np.c_[affine(x),powers(x,2)],Y)); print('panel B svd',np.linalg.svd(operators(S)[1][1:],compute_uv=False))
 def cv(f,v):
  q=np.linalg.svd(f,full_matrices=False)[0]; h=(q*q).sum(1); return np.sqrt(np.mean(((v-f@ls(f,v))/(1-h[:,None]))**2,axis=0))
 for key in ['v1','v2']:
  v=c[key]
  for name,w in {'raw':u,'BA':coords(u,S)['BA'],'centered_affine':affine(u)@ls(affine(u),v),'centered_B':coords(u,S)['B']}.items():
   f=affine(u); options={};
   for family in ['sin','cos','tanh','quad','cube']:
    freq=np.linspace(.5,5,91) if family in ('sin','cos','tanh') else [1.]
    vals=np.array([cv(np.c_[f, {'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'quad':np.square,'cube':lambda x:x**3}[family](a*w)],v) for a in freq]); i=np.argmin(np.mean(vals,axis=1)); options[family]=(round(freq[i],3),np.round(vals[i],5))
   print(key,name,options)
  if key=='v2':print('v2 diag-quadratic coeff',ls(np.c_[affine(u),u*u],v))
 print('56/16 and six 56/16 splits; selected simple readout candidates')
 rng=np.random.default_rng(451); splits=[(np.arange(56),np.arange(56,72))]+[(p[:56],p[56:]) for p in [rng.permutation(72) for _ in range(6)]]
 results=[]
 for k,w in coords(x,S).items():
  for name,g in [('lin',affine(w)),('p2',np.c_[affine(w),powers(w,2)]),('d3',np.c_[affine(w),powers(w,2),w**3]),('d34',np.c_[affine(w),powers(w,2),w**3,w**4]),('cube',np.c_[affine(w),w**2,w**3]),('p3',np.c_[affine(w),powers(w,2),powers(w,3)]),*((f'{f}{a}',np.c_[affine(w),powers(w,2),h(a*w)]) for f,h in [('sin',np.sin),('th',np.tanh)] for a in [.8,1.3,1.8,2.3,3.])]:
   er=[]
   for tr,te in splits:
    m=modes(t,z[tr]); a=ls(m,z[tr].T).T; pr=g[te]@ls(g[tr],np.c_[y[tr],a]); er.append([np.mean(abs(pr[:,0]-y[te])),np.mean(abs(pr[:,1:]@m.T-z[te]))]);
   er=np.array(er); results.append((k,name,er))
 for col in range(2):
  print('target',col)
  for k,name,e in sorted(results,key=lambda q:q[2][:,col].mean())[:14]:print(k,name,'gate',e[0],'mean',e.mean(0),'worst',e.max(0))
