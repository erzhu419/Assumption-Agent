exec(open('/tmp/probe12.py').read().split('for key,tar')[0])
from scipy.optimize import minimize_scalar
k=s['telemetry_vec'][-1]
models={}
for key,tar in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[key]['u']; v=s[key][tar]; folds=np.array_split(np.arange(len(u)),6)
 def f(x,base,extra,k):
  d=np.cos(k*x)-1; h=np.sin(k*x); r=np.roll(x,-1,axis=1)
  a={'a':aff(x),'p':poly(x,2),'ac':np.c_[aff(x),x*r],'ah':np.c_[aff(x),x*x],'pc':poly(x,3)}[base]
  e={'cos':d,'cos+sin':np.c_[d,h],'cos+odd':np.c_[d,x*d],'cos+xc':np.c_[d,r*d],'cos+prod':np.c_[d,x*r*d],'cos+sat':np.c_[d,x/np.hypot(1,k*x)],'sat':x/np.hypot(1,k*x),'tanh':np.tanh(k*x)}[extra]
  return np.c_[a,e]
 results=[]
 for a in ['a','p','ac','ah','pc']:
  for e in ['cos','cos+sin','cos+odd','cos+xc','cos+prod','cos+sat','sat','tanh']:
   if f(u,a,e,k).shape[1]>len(u)*.7: continue
   results.append((cv(f(u,a,e,k),v,folds),a,e))
 for met,a,e in sorted(results)[:5]:
  F=f(u,a,e,k); w=ls(F,v); print(key,tar,a,e,met,'in',np.mean(abs(F@w-v)))
  if (met,a,e)==min(results):
   print('coef',np.round(w,3)); models[key+tar]=(a,e,w)
 if key=='panel_b':
  for a,e in [('p','cos'),('a','cos'),('ac','cos')]:
   opt=minimize_scalar(lambda t:np.linalg.norm((F:=f(u,a,e,t))@ls(F,v)-v),bounds=(1,4),method='bounded'); print('freq',a,e,opt.x,opt.fun)
bases0={nm:f(q,a,e,k)@w for nm,(a,e,w) in models.items()}
bases0.update({'cos':np.cos(k*q)-1, 'q':q})
bases0['bc']=np.c_[bases0['panel_bv'],bases0['panel_cv2']]
y=tr['y']; T=tr['traj']; folds=np.array_split(np.arange(len(q)),6)
R=[]
for name,b in bases0.items():
 for dn,d in bases0.items():
  if b.shape[1]!=4 or d.shape[1]!=4:continue
  for an,A in [('aff',aff(b)),('affq',np.c_[aff(b),q]),('poly',poly(b,2)),('cross',np.c_[aff(b), b*np.roll(d,-1,axis=1)]),('a+b',np.c_[aff(b),d]),('allcross', np.c_[aff(b),(b[:,:,None]*d[:,None,:]).reshape(len(b),-1)])]:
   R.append((cv(A,y,folds),cv(A,T,folds),name,dn,an,'none'))
   for i in range(4):
    g=1/(1+s['telemetry_vec'][2]*q[:,i]**2); F=np.c_[np.ones(len(q)),A[:,1:]*g[:,None]]; R.append((cv(F,y,folds),cv(F,T,folds),name,dn,an,'gate'+str(i)))
print('y best',*sorted(R)[:16],sep='\n'); print('t best',*sorted(R,key=lambda v:v[1])[:8],sep='\n')
