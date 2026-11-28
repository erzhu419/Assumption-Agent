import sys,pickle,itertools as it
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy.optimize import least_squares
import node_00 as n
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; x=n._invert(s['train']['x'],n._fit(a['u'],a['v'])); y=s['train']['y']; t=s['train']['traj']; b=s['panel_b']
A=lambda x:np.c_[np.ones(len(x)),x]
B=np.linalg.lstsq(A(b['u']),b['v'],rcond=None)[0]
terms=[p for k in (2,3) for p in it.combinations_with_replacement(range(4),k)]
def fit(D,y,l=0):
 p=np.linalg.lstsq(D,y,rcond=None)[0]; reg=np.r_[np.zeros(5),np.full(D.shape[1]-5,np.sqrt(l))]
 def pred(p,D):
  q=D@p[:-3]; o,a,f=p[-3:]; return q+a*np.tanh(f*q+o)
 def res(p):return np.r_[pred(p,D)-y,reg*p[:-3]]
 fs=[least_squares(res,np.r_[p,0,a,f],bounds=(np.r_[np.full(len(p)+2,-np.inf),.12],np.r_[np.full(len(p)+2,np.inf),4]),max_nfev=180) for a,f in [(0.15,1),(0.2,2),(0.35,3)]]
 return min(fs,key=lambda p:p.cost).x,pred
rng=np.random.default_rng(429); splits=[np.arange(len(x))]+[rng.permutation(len(x)) for _ in range(4)]
for frame,X in [('u',x),('z',A(x)@B)]:
 D=np.c_[A(X),*[np.prod(X[:,p],axis=1) for p in terms]]
 p,_=fit(D[:52],y[:52]); print(frame,'cubic',[(q,round(c,3)) for q,c in zip(terms,p[5:-3])]); print('link',np.round(p[-3:],3))
 ts=np.linalg.lstsq(D[:52,:15],t[:52],rcond=None)[0]; print('trajectory quadratic norms',np.round(np.linalg.norm(ts[5:],axis=1),3))
 for name,ids in [('all',range(35)),('pure',list(range(15))+[15+i for i,q in enumerate(terms[10:]) if len(set(q))==1]),('no-three',list(range(15))+[15+i for i,q in enumerate(terms[10:]) if len(set(q))<3])]:
  d=D[:,ids];
  for l in (0,.003,.03,.15):
   losses=[]
   for j in splits:
    tr,te=j[:52],j[52:]; p,pred=fit(d[tr],y[tr],l); losses.append(np.mean(abs(pred(p,d[te])-y[te])))
   print(frame,name,l,np.round(losses,5))
 for count in (5,8,11,15):
  losses=[]
  for j in splits:
   tr,te=j[:52],j[52:]; ids=list(range(35)); scale=np.std(D[tr],axis=0); scale[0]=1
   while len(ids)>5+count:
    p,_=fit(D[tr][:,ids],y[tr],.003); score=abs(p[5:-3])*scale[ids[5:]]; k=np.argsort(score)[:min(4,len(ids)-5-count)]; ids=[i for k0,i in enumerate(ids) if k0-5 not in k or k0<5]
   p,pred=fit(D[tr][:,ids],y[tr]); losses.append(np.mean(abs(pred(p,D[te][:,ids])-y[te])))
  print(frame,'sparse',count,np.round(losses,5))
