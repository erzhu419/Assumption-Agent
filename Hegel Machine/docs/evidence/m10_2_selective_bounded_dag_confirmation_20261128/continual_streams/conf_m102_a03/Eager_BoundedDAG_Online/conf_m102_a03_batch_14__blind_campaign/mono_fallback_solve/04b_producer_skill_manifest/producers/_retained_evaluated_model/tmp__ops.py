import numpy as np,pickle,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train'];x=t['x'];y=t['y'];T=t['traj']; times=s['eval_times']; tele=s['telemetry_vec']
def aff(a):return np.c_[np.ones(len(a)),a]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
fn={'sin':lambda x,w:np.sin(w*x),'cos':lambda x,w:np.cos(w*x)-1,'tanh':lambda x,w:np.tanh(w*x),'rat':lambda x,w:x/(1+w*x*x)}
ops={}
for name,p,k in [('A','panel_a','v'),('B','panel_b','v'),('C','panel_c','v1'),('D','panel_c','v2')]:
 u=s[p]['u'];v=s[p][k];i=np.linalg.norm(v-u,axis=1)<2 if name=='A' else np.ones(len(u),bool);u=u[i];v=v[i]
 scores=[]
 for f,g in fn.items():
  def obj(w,ret=False):
   a=aff(np.c_[u,g(u,w)]);c=ls(a,v);e=v-a@c; h=(np.linalg.qr(a)[0]**2).sum(1);return (c,e) if ret else np.sqrt(np.mean((e/(1-h[:,None]))**2))
  r=min((minimize_scalar(obj,bounds=(lo,lo+.5),method='bounded') for lo in np.arange(.05,4.55,.5)),key=lambda r:r.fun); c,e=obj(r.x,True);scores.append((r.fun,f,r.x,c));print(name,f,round(r.x,5),round(r.fun,7))
 score,f,w,c=min(scores,key=lambda z:z[0]);print('BEST',name, f,w,score,np.round(c,4));ops[name]=(lambda u,g=fn[f],w=w,c=c:aff(np.c_[u,g(u,w)])@c)
 if name=='A': cal=c

def features(x):
 z=np.linalg.solve(cal[1:5].T,(x-cal[0]).T).T
 for _ in range(35):
  step=np.linalg.solve((ops['A'](z[:,None,:].reshape(-1,4))[:,None,:]*0+np.array([(ops['A'](z+np.eye(4)[k]*1e-5)-ops['A'](z))/1e-5 for k in range(4)]).transpose(1,2,0)),(x-ops['A'](z))[:,:,None])[:,:,0];z+=np.clip(step,-.35,.35)
 F={'x':x,'z':z};
 for prefix in ['x','z']:
  for j in 'BCD':F[prefix+j]=ops[j](F[prefix]);
 for prefix in list(F)[2:]:
  for j in 'BCD':F[prefix+j]=ops[j](F[prefix]);
 for prefix in ['x','z']:
  for a,b in [('B','C'),('B','D'),('C','D')]:
   u,v=F[prefix+a],F[prefix+b];F[prefix+a+'+'+b]=u+v;F[prefix+a+'-'+b]=u-v;F[prefix+a+'*'+b]=u*v
 return F
if __name__=='__main__':
 F=features(x);np.savez('/tmp/truefeatures.npz',**F);print('range',F['z'].min(0),F['z'].max(0));
 for d in [1,2,3]:
  scores=[]
  for n,z in F.items():
   a=aff(np.c_[z,*[np.prod(z[:,i],1) for deg in range(2,d+1) for i in itertools.combinations_with_replacement(range(4),deg)]]);v=np.c_[y,T];p=a@ls(a[:52],v[:52]);e=np.mean(abs(p[52:]-v[52:]),0);scores.append((e[0],e[1:].mean(),n))
  print(d,'y',sorted(scores)[:8],'t',sorted(scores,key=lambda a:a[1])[:8]);
 for rate in [.22,.5,1,1.25,2.15]:
  a=np.array([1-np.exp(-rate*times),times]).T;c=ls(a,T.T).T; print('timebasis',rate,np.mean(abs(c@a.T-T)), np.std(c,0));
