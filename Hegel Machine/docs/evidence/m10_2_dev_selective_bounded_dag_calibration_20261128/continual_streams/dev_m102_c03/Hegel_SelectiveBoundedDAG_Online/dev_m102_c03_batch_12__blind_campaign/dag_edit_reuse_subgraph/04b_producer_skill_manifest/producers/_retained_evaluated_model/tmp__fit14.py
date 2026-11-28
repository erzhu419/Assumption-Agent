import numpy as np,pickle,itertools as it
from scipy.optimize import least_squares
from scipy.linalg import qr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def one(x):return np.column_stack([np.ones(len(x)),x])
def ls(h,y):return np.linalg.lstsq(h,y,rcond=None)[0]
p=s['panel_a']; u,v=p['u'],p['v'];h=one(u);keep=np.argsort(np.linalg.norm(h@ls(h,v)-v,axis=1))[:-3]
a,b=u,np.roll(u,2,axis=1)
options={}
for form in ['sinboth','sinowncub','sincross','onesin','sinrat','sintanh','sincos','sinpoly']:
 def mat(a,b,k):
  basic=[np.ones_like(a),a,np.sin(k[0]*b)]
  rest={'sinboth':[np.sin(k[1]*a)],'sinowncub':[a**3],'sincross':[a*b,a*b*b,a*a*b],'onesin':[],'sinrat':[a/(1+k[1]*a*a)],'sintanh':[np.tanh(k[1]*a)],'sincos':[np.sin(k[1]*a),np.cos(k[1]*a)],'sinpoly':[b,a*a,b*b,a**3,b**3,a*b,a*a*b,a*b*b]}
  return np.stack(basic+rest[form],axis=2)
 def coef(m):return np.stack([ls(m[keep,j],v[keep,j]) for j in range(4)])
 def pred(m,c):return np.einsum('njp,jp->nj',m,c)
 def resid(k):
  m=mat(a,b,k);return (pred(m,coef(m))-v)[keep].ravel()
 opt=least_squares(resid,[2.4,3.1],bounds=([1.5,.05],[3,6]),max_nfev=80); k=opt.x; c=coef(mat(a,b,k));print(form,np.std(opt.fun),k,np.round(c[0],5));
 def f(z):return pred(mat(z,np.roll(z,2,axis=1),k),c)
 x=s['train']['x'];z=(x-c[:,0])/c[:,1];
 for j in range(len(x)):
  z[j]=least_squares(lambda q:f(q[None])[0]-x[j],z[j],max_nfev=50).x
 print('inverse',np.max(abs(f(z)-x)),np.max(abs(z)),flush=True);options[form]=z
x=s['train']['x'];p=s['panel_b']; c=ls(one(p['u']),p['v']);options['raw']=x;options['binv']=(x-c[0])@np.linalg.pinv(c[1:]);
p=s['panel_c'];edge=lambda z:z*np.roll(z,-1,axis=1);ca=ls(one(edge(p['u'])),p['v1']);cb=ls(one(np.cos(2.4*p['u'])),p['v2']);
Y=np.column_stack([s['train']['y'],s['train']['traj']]);n=len(Y);splits=[(np.arange(52),np.arange(52,n))]+[(np.where(np.arange(n)%4!=i)[0],np.where(np.arange(n)%4==i)[0]) for i in range(4)]
def score(h,Y=Y):
 out=[]
 for tr,te in splits:
  r=abs(h[te]@ls(h[tr],Y[tr])-Y[te]);out.append([r[:,0].mean(),r[:,1:].mean()])
 return np.round(np.array(out),4)
for nm,z in options.items():
 C=one(edge(z))@ca;D=one(np.cos(2.4*z))@cb; cd=np.einsum('ni,nj->nij',C,D).reshape(len(z),-1);dot=(C*D).sum(1); pair=np.array([z[:,i]*z[:,j] for i,j in it.combinations(range(4),2)]).T; cubic=np.array([z[:,i]*z[:,j]*z[:,k] for i,j,k in it.combinations_with_replacement(range(4),3)]).T;
 print(nm,[(name,score(one(np.column_stack(parts)))[0].tolist(),score(one(np.column_stack(parts)))[1:].max(0).tolist()) for name,parts in [('core',[C,D,dot,C[:,0]*D[:,1]]),('linear',[C,D,dot,C[:,0]*D[:,1],z]),('cd',[C,D,cd,z]),('quad',[C,D,dot,C[:,0]*D[:,1],z,z*z,pair]),('cube',[C,D,dot,C[:,0]*D[:,1],z,z*z,pair,cubic])]],flush=True)
np.savez('/tmp/latent14.npz',**options)
