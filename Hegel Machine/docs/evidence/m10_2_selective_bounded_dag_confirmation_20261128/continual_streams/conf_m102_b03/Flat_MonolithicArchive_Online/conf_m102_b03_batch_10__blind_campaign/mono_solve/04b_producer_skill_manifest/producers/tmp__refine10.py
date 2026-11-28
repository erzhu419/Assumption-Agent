from model10 import *
import pickle
from scipy.optimize import minimize_scalar
from scipy.special import expit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; u=s['panel_c']['u']; v=s['panel_c']['v1']; z=calibrate(s['panel_a'],tr['x']); Q=lambda x:D(np.c_[x,np.cos(2.4*x)]); c=Q(z)@ls(Q(u),s['panel_c']['v2']); t=timebasis(s['eval_times'],tr['traj'][:52]); Y=np.c_[tr['y'],ls(t,tr['traj'].T).T];
def cv(F):
 p=np.stack([f[52:]@ls(f[:52],Y[:52,j]) for j,f in enumerate(F)],1); e=abs(p[:,0]-Y[52:,0]).mean(); et=abs(p[:,1:]@t.T-tr['traj'][52:]).mean(); return np.round(np.r_[e,et,np.mean(abs(p-Y[52:]),0)],6)
funcs={'tanh':np.tanh,'rat':lambda x:x/(1+x*x),'softsign':lambda x:x/(1+abs(x)),'asinh':np.arcsinh,'sinh':np.sinh,'sin':np.sin,'atan':np.arctan,'softclip':lambda x:x/np.sqrt(1+x*x),'silu':lambda x:x*expit(x),'erf':FUN['erf'],'clip':lambda x:np.clip(x,-1,1),'swishodd':lambda x:x*expit(-abs(x)), 'logodd':lambda x:np.sign(x)*np.log1p(abs(x))}
extras={'none':lambda x:x[:,:0], 'edge':lambda x:x*np.roll(x,-1,1), 'square':lambda x:x*x,'triple':lambda x:x*np.roll(x,-1,1)*np.roll(x,-2,1),'prod':lambda x:x*x*np.roll(x,-1,1),'cubic':lambda x:x**3}
res=[]
for name,fun in funcs.items():
 for en,extra in extras.items():
  def f(x,k): return D(np.c_[x,fun(k*x),extra(x)])
  def score(k):
   a=f(u,k); return np.mean((a@ls(a,v)-v)**2)
  o=minimize_scalar(score,bounds=(.09,4.5),method='bounded'); res.append((o.fun**.5,name,en,o.x,f(z,o.x)@ls(f(u,o.x),v)))
for rr in sorted(res,key=lambda q:q[0])[:23]: print('panel',rr[:4]);
for r in [q for q in res if q[2]=='none']+sorted(res,key=lambda q:q[0])[:8]:
 a=r[-1]; dot=(a*c).sum(1); e=a[:,0]*c[:,1]; base=D(np.c_[a,c,dot]); ext=D(np.c_[a,c,dot,e]); slow=D(np.c_[c,e]); fastz=D(np.c_[z,a,c,dot,e]); cross=(a[:,:,None]*c[:,None,:]).reshape(len(z),-1); allx=D(np.c_[z,a,c,cross]);
 print(r[:4],cv([base,ext,slow]),cv([base,fastz,slow]),cv([base,allx,slow]));
