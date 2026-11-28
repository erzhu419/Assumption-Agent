from maps15 import *
from itertools import combinations
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
I,J=np.array(list(combinations(range(4),2))).T
def pairs(z):return z[:,I]*z[:,J]
def fitA(s,x):
 p=s['panel_a'];u=p['u'];v=p['v']; k=np.linalg.norm(v,axis=1)<4; u=u[k];v=v[k]; js=np.arange(4)
 def design(u,j,w):return one(u[:,j],np.sin(w*u[:,j]),np.sin(w*u[:,(j+2)%4]))
 def fit(w):return [ls(design(u,j,w),v[:,j]) for j in js]
 def err(w):return sum(np.sum((design(u,j,w)@b-v[:,j])**2) for j,b in enumerate(fit(w)))
 w=minimize_scalar(err,bounds=(1.8,3.0),method='bounded').x; b=np.array(fit(w)); print('A',w,np.sqrt(err(w)/v.size),b)
 bias,a,c,d=b.T;z=(x-bias)/a
 for _ in range(30):
  sin=np.sin(w*z);cos=w*np.cos(w*z); r=bias+a*z+c*sin+d*np.roll(sin,2,axis=1)-x
  jac=np.zeros((len(x),4,4));jac[:,js,js]=a+c*cos;jac[:,js,(js+2)%4]=d*np.roll(cos,2,axis=1)
  z-=np.clip(np.linalg.solve(jac,r[...,None])[...,0],-.35,.35)
 print('inv max',abs(r).max(), 'z range',z.min(0),z.max(0));return z

def branches(s,z):
 p=s['panel_c'];u=p['u'];out=[]
 for key in ('v1','v2'):
  v=p[key];scores=[]
  for name,f in func.items():
   for form in range(3):
    def des(u,w):return one(*([u]+([pairs(u)] if form else [])+([u*u] if form==2 else [])+[f(u,w)]))
    def err(w):
     h=des(u,w);r=v-h@ls(h,v);lev=np.sum(h*np.linalg.pinv(h).T,axis=1);return np.mean((r/(1-lev[:,None]))**2)
    grid=np.linspace(.25,5,40);k=grid[np.argmin([err(w) for w in grid])]; w=minimize_scalar(err,bounds=(max(.15,k-.13),k+.13),method='bounded').x
    scores.append((err(w),name,form,w,ls(des(u,w),v),des(z,w)))
  scores.sort(key=lambda a:a[0]); print(key,[(a[0]**.5,*a[1:4]) for a in scores[:6]]); e,name,form,w,b,h=scores[0]; print(b.round(4));out.append(h@b)
 return out

def run(s):
 tr=s['train'];z=fitA(s,tr['x']); B=s['panel_b'];b=one(z)@ls(one(B['u']),B['v']);c,d=branches(s,z)
 Y=np.column_stack((tr['y'],tr['traj']));n=len(z);cut=int(.75*n);print('n',n,'cut',cut,'t',s['eval_times']);np.savez('/tmp/coords15.npz',z=z,b=b,c=c,d=d,Y=Y)
 scores=[]
 for nam,X in [('z',z),('b',b),('c',c),('d',d),('cd',np.c_[c,d]),('zbcd',np.c_[z,b,c,d])]:
  for pre in ('none','tanh','atan'):
   xx={'none':X,'tanh':np.tanh(X),'atan':np.arctan(X)}[pre];xx=(xx-xx.mean(0))/xx.std(0);dd=cdist(xx,xx,'sqeuclidean')/xx.shape[1]
   for kind in ('RBF','M32','M52','poly'):
    for k in [.08,.2,.5,1,2,4]:
     a=np.sqrt(k*dd);K={'RBF':np.exp(-a*a),'M32':(1+a)*np.exp(-a),'M52':(1+a+a*a/3)*np.exp(-a),'poly':(1+xx@xx.T/xx.shape[1])**int(k+2)}[kind]
     for reg in [1e-6,1e-4,.001,.01,.05]:
      A=K[:cut,:cut]+np.eye(cut)*reg; P=np.linalg.inv(A); co=P@Y[:cut]; loo=np.mean(np.abs(co/np.diag(P)[:,None]),axis=0); e=np.mean(abs(K[cut:,:cut]@co-Y[cut:]),axis=0);scores.append((max(loo[0],loo[1:].mean()),e[0],e[1:].mean(),nam,pre,kind,k,reg))
 print('LOO',sorted(scores)[:12]);print('CV',sorted(scores,key=lambda a:max(a[1:3]))[:12]);
if __name__=='__main__':run(s)
