import numpy as np,pickle
from scipy.optimize import root,minimize_scalar
from itertools import combinations_with_replacement as cr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=160)
def lin(z):return np.c_[np.ones(len(z)),z]
def ls(a,y):return np.linalg.lstsq(a,y,rcond=1e-9)[0]
def powers(z,d):return np.column_stack([np.prod(z[:,i],axis=1) for i in cr(range(z.shape[1]),d)])
def cyc(z):return z*np.roll(z,-1,axis=1)
def orth(a):return np.linalg.qr(a)[0]
def look(a,y):
 e=y-a@ls(a,y);return np.mean(abs(e)),np.mean(abs(e/(1-(orth(a)**2).sum(1)[:,None])))
def calib(p,X):
 u,v=p['u'],p['v'];D=lambda z:np.c_[lin(z),cyc(z),z**3];e=np.linalg.norm(v-D(u)@ls(D(u),v),axis=1);good=np.argsort(e)[:-3];w=np.roll(u,-1,axis=1); b=np.array([ls(np.c_[np.ones(len(u)),u[:,j],w[:,j],u[:,j]*w[:,j],u[:,j]**3][good],v[good,j]) for j in range(4)]);print('cal',b)
 def func(z):
  n=np.roll(z,-1,axis=-1);return b[:,0]+b[:,1]*z+b[:,2]*n+b[:,3]*z*n+b[:,4]*z**3
 print('cal err',np.max(abs(func(u)[good]-v[good])));z=np.array([root(lambda z:func(z)-x,x).x for x in X]);print('cal inverse err and range',np.max(abs(func(z)-X)),z.min(0),z.max(0));return z

def configs(p,y):
 z=p['u']; L=[]
 F={'id':lambda z,k:z,'cube':lambda z,k:z**3,'sinh':lambda z,k:np.sinh(k*z),'atan':lambda z,k:np.arctan(k*z),'tanh':lambda z,k:np.tanh(k*z),'asinh':lambda z,k:np.arcsinh(k*z),'rat':lambda z,k:z/(1+k*z*z)}
 E={'nil':lambda z:np.empty((len(z),0)),'lin':lambda z:z,'sq':lambda z:z*z,'quad':lambda z:powers(z,2),'linsq':lambda z:np.c_[z,z*z],'lincyc':lambda z:np.c_[z,cyc(z)],'cyc':cyc,'lincube':lambda z:np.c_[z,z**3]}
 for fn,f in F.items():
  for en,g in E.items():
   if fn=='id' and en.startswith('lin'):continue
   def d(z,k):return np.c_[lin(f(z,k)),g(z)]
   opt=minimize_scalar(lambda k:look(d(z,k),y)[1],bounds=(.15,4.),method='bounded') if fn not in ('id','cube') else None
   k=opt.x if opt else 1.; a=d(z,k);r,cv=look(a,y); beta=ls(a,y)
   def predict(x,f=f,g=g,k=k,beta=beta):return np.c_[lin(f(x,k)),g(x)]@beta
   L.append((cv,r,fn,en,k,predict,beta))
 L.sort(key=lambda r:r[0]); print('Panel fit',[(round(cv,5),round(r,5),f,g,round(k,3)) for cv,r,f,g,k,*_ in L[:12]]);print(L[0][-1]);return L

def readouts(z,w):
 yield 'linear',lin(w)
 yield 'quadratic',np.c_[lin(w),powers(w,2)]
 yield 'xw',np.c_[lin(w),z,w*z]
 yield 'cubicdiag',np.c_[lin(w),powers(w,2),w**3]
 yield 'wrat',np.c_[lin(w),w/(1+w*w),w*w/(1+w*w)]
 yield 'wth',np.c_[lin(w),np.tanh(w),w*np.tanh(w)]
 yield 'wcube',np.c_[lin(w),w**3]
 yield 'squares',np.c_[lin(w),w*w]
 yield 'nointercept',w

def run():
 X=np.r_[s['train']['x'],s['test']['x']];C=calib(s['panel_a'],X);Y=np.c_[s['train']['y'],s['train']['traj']];
 P=[]
 for nm in ['b','c']:
  p=s['panel_'+nm]
  for k,y in p.items():
   if k!='u':P.append((nm+k,configs(p,y)[:4]))
 trans={'id':lambda z:z,'cube':lambda z:z**3,'tanh':np.tanh,'atan':np.arctan,'rat':lambda z:z/(1+z*z),'sq':lambda z:z*z,'sinh':np.sinh,'x+x3':lambda z:z+z**3,'x+.22x3':lambda z:z+.22*z**3,'roll':lambda z:np.roll(z,-1,axis=1)}
 S=[]
 def score(n,F):
  if not np.all(np.isfinite(F)):return
  pred=F[52:72]@ls(F[:52],Y[:52]); a=np.mean(abs(pred-Y[52:]),axis=0);S.append((a[0],a[1:].mean(),n))
 for zn,z in [('raw',X),('cal',C)]:
  for tn,t in trans.items():
   W=t(z); pool={}
   for pnm,pm in P:
    for i,m in enumerate(pm):
     w=m[-2](W); k=f'{pnm}{i}';pool[k]=w
     for f,F in readouts(z,w):score((zn,tn,k,f),F)
     if np.max(abs(w))>20:continue
     for pp,qq in P:
      for j,op in enumerate(qq[:2]):
       v=op[-2](w)
       for f,F in readouts(w,v):score((zn,tn,k,pp+str(j),f),F)
   for i in range(3):
    for j in range(i+1,3):
     a,b=P[i][0]+'0',P[j][0]+'0';u,v=pool[a],pool[b];score((zn,tn,a,b,'concat'),lin(np.c_[u,v]));score((zn,tn,a,b,'prod'),lin(np.c_[u,v,u*v]));score((zn,tn,a,b,'full cross'),lin(np.c_[u,v,(u[:,:,None]*v[:,None,:]).reshape(len(u),-1)]))
 for key in [lambda r:sum(r[:2]),lambda r:r[0],lambda r:r[1]]:print('BEST',sorted(S,key=key)[:13]);
if __name__=='__main__':run()
