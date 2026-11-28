exec(open('/tmp/diagnose.py').read().split("for k in ['panel_a'")[0])
sys.path.insert(0,'/holdout_task');import node_00 as m
z,b,c,p=m.features(s)
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
from itertools import product
from scipy.special import erf
from pathlib import Path
print('basis residual',abs(t['traj']-T[:,1:]@f.T).mean(), 'ranks',np.linalg.svd(t['traj'],compute_uv=False),flush=True)
# Coordinates before and after the panel mixing, and candidate paired sums/differences.
forms={'z':z,'b':b,'sym':np.c_[.3*z+b,.3*z-b],'sep':np.c_[.3*z,b],'zbc':np.c_[z,b,c],'zbcp':np.c_[z,b,c,p],'can':np.c_[z,np.sign(z)*np.maximum(abs(z)-.35,0)**1.5,z*z],'cz':np.c_[z,c],'zbp':np.c_[z,b,p]}
def grids(forms):
 for nm,F in forms.items():
  for tf in [0,1,3]:
   X=(np.arcsinh(tf*F)/tf if tf else F); x=X/F.std(); dot=x@x.T/x.shape[1]; d=cdist(x,x,'sqeuclidean')/x.shape[1]; u=abs(x[:,None,:]-x[None,:,:]);
   for decay in [.025,.15,.6,1.8,5.]:
    G=np.exp(-decay*d); g=np.exp(-decay*u*u); a=g.mean(2); pair=(a*a-(g*g).mean(2)/x.shape[1])*x.shape[1]/(x.shape[1]-1)
    for family,H in enumerate([G,(1+dot)**2+G,(1+dot)**3+G,(1+dot)**4+G,1+dot+a+.75*pair,(1+dot)**2+a+.75*pair]):
     yield (nm,tf,decay,family),H
Y=T; folds=sl; rankings=[]
for params,k in grids(forms):
 for ridge in [1e-7,1e-5,.0003,.003,.025]:
  scores=[]; preds=[]
  for i,j in folds:
   pred=k[np.ix_(j,i)]@cho_solve(cho_factor(k[np.ix_(i,i)]+ridge*np.eye(len(i))),Y[i]);e=pred-Y[j]; scores.append([abs(e[:,0]).mean(),abs(e[:,1:]@f.T).mean()]);preds.append(pred)
  rankings.append((params+(ridge,),np.array(scores),preds))
for h in (0,1):
 for weight in (0.,.25,.6):
  def score(e):return (1-weight)*(.65*e[:,h].mean()+.35*e[:,h].max())+weight*e[0,h]
  best=sorted(rankings,key=lambda r:score(r[1])); print('TOP',h,weight,[(r[0],r[1][:,h].round(5).tolist()) for r in best[:4]],flush=True)
  for number in (4,12,35,100):
   picks=[]; seen=set()
   for r in best:
    if r[0][:4] not in seen: picks.append(r);seen.add(r[0][:4]);
    if len(picks)==number:break
   E=[np.mean([r[2][k] for r in picks],axis=0)-Y[j] for k,(i,j) in enumerate(folds)]; scores=np.array([[abs(e[:,0]).mean(),abs(e[:,1:]@f.T).mean()] for e in E]);print('BLEND',h,weight,number,scores[:,h].round(6),picks[0][0],flush=True)
  if weight==.25:
   Path('/tmp/params'+str(h)).write_text(repr([r[0] for r in best[:24]]))
# Assess whether the head is an additive sum of known transformed coordinate functions.
for nm,A in forms.items():
 X=D(np.c_[A,np.arcsinh(3*A),np.tanh(2*A),A*A]); print('additive',nm,assess(X),flush=True)
