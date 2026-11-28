import itertools,pickle,sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from batch_solver import parts,times,fit,kernel,AF,D
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train'];n=len(tr['x']);z,b=parts(s)
splits=[(np.arange(52),np.arange(52,n))]+list(KFold(4,shuffle=True,random_state=921).split(z[:n]));
for x in [z,b]:
 j=np.argsort(np.linalg.norm(x[:n],axis=1))[-18:];splits.append((np.setdiff1d(np.arange(n),j),j))
X=np.c_[.3*z+b,.3*z-b]; W=np.c_[.3*z,b]; inputs=[np.arcsinh(X),np.arcsinh(W),X,W]
opts=[(0,3,.15,1e-6),(1,3,.7,1e-6)]+list(itertools.product(range(4),(2,3),(.15,.4,.7,1.4),(1e-6,1e-4)))
TF=[]
for i,j in splits:
 f=times(dict(s,train={'traj':tr['traj'][i]})); TF.append((f,np.c_[tr['y'],fit(f,tr['traj'].T).T]));
E=[]
for a,p,k,r in opts:
 K=kernel(inputs[a],p,k); pred=[]
 for (i,j),(f,Y) in zip(splits,TF):
  h=K[j][:,i]@np.linalg.solve(K[i][:,i]+r*np.eye(len(i)),Y[i]);pred.append((h[:,0]-tr['y'][j],h[:,1:]@f.T-tr['traj'][j]))
 E.append(pred)
chosen=[]
for h in range(2):
 base=E[h]; bmetrics=np.array([np.mean(abs(row[h])) for row in base]); best=(np.inf,h,h,1.,bmetrics)
 for a,c in itertools.combinations_with_replacement(range(len(opts)),2):
  for frac in ((1.,) if a==c else (0.25,0.5,0.75)):
   e=np.array([np.mean(abs(frac*u[h]+(1-frac)*v[h])) for u,v in zip(E[a],E[c])]);
   if e[0]>[.032,.037][h] or e[1:5].max()>bmetrics[1:5].max()+.001 or e[1:5].mean()>bmetrics[1:5].mean()+.0015 or e[5:].max()>bmetrics[5:].max():continue
   val=.15*e[0]+.35*e[1:5].mean()+.15*e[1:5].max()+.20*e[5:].mean()+.15*e[5:].max()
   if val<best[0]:best=(val,a,c,frac,e)
 _,a,c,frac,e=best; chosen.append((opts[a],opts[c],frac)); print('head',h,'baseline',bmetrics,'challenger',chosen[-1],e)
 if not (e[5:].max()<bmetrics[5:].max()*(.985)):chosen[-1]=(opts[h],opts[h],1.)
a=s['panel_a']; F=AF(a['u']); W0=fit(D(a['u']),a['v']); keep=np.linalg.norm(D(a['u'])@W0-a['v'],axis=1)<2
for label,F in [('affine',D(a['u'])),('cyclic cubic',F)]:
 rows=[]
 for j in np.flatnonzero(keep):
  i=keep.copy();i[j]=False;rows.append(F[j]@fit(F[i],a['v'][i])-a['v'][j]);
 print('Panel A inlier LOO',label,np.mean(np.abs(rows)))
p=s['panel_b']; u,v=p['u'],p['v'];
for name in ['hinge','erf reference']:
 from scipy.special import erf
 def H(x,q):
  return D(np.sign(x)*np.maximum(abs(x)-q[0],0)**q[1]) if name=='hinge' else D(np.c_[x,erf(q[0]*x)])
 errs=[]
 for j in range(len(u)):
  i=np.arange(len(u))!=j
  def err(q):
   F=H(u[i],q);return (F@fit(F,v[i])-v[i]).ravel()
  q=least_squares(err,[.35,1.5],bounds=([.15,.6],[.6,2.5])).x if name=='hinge' else least_squares(err,[1.49],bounds=([.4],[3.])).x
  errs.append(H(u[j:j+1],q)@fit(H(u[i],q),v[i])-v[j]);
 print('Panel B nested LOO',name,np.mean(np.abs(errs)))
P=Path('/holdout_task/batch_solver.py'); src=P.read_text(); start=src.index('def predict(study):'); end=src.index('def solve_batch',start)
if chosen!=[(opts[0],opts[0],1.),(opts[1],opts[1],1.)]:
 body='''def predict(study):
    z,b=parts(study)
    f=times(study)
    tr=study['train']; n=len(tr['x'])
    modes=fit(f,tr['traj'].T).T
    sym=np.c_[.3*z+b,.3*z-b]
    sep=np.c_[.3*z,b]
    designs=(np.arcsinh(sym),np.arcsinh(sep),sym,sep)
    def head(spec,target):
        form,degree,decay,ridge=spec
        k=kernel(designs[form],degree,decay)
        return k[n:,:n]@np.linalg.solve(k[:n,:n]+ridge*np.eye(n),target)
    def blend(a,b,w,target):
        pred=head(a,target)
        return pred if w==1 else w*pred+(1-w)*head(b,target)
    y=blend(*{y!r},tr['y'])
    state=blend(*{t!r},modes)@f.T
    return y,state

'''.format(y=chosen[0],t=chosen[1]);P.write_text(src[:start]+body+src[end:])
