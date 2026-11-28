from graph import *
from scipy.optimize import minimize,least_squares
X=np.r_[s['train']['x'],s['test']['x']]; z=calib(s['panel_a'],X); Y=np.c_[s['train']['y'],s['train']['traj']];N=len(Y)
ops={'atan':np.arctan,'tanh':np.tanh,'asinh':np.arcsinh,'sin':np.sin,'rat':lambda x:x/(1+x*x),'soft':lambda x:x/np.sqrt(1+x*x),'erf':__import__('scipy.special',fromlist=['erf']).erf}
base={'0':lambda x:np.empty((len(x),0)),'lin':lambda x:x,'cube':lambda x:x**3,'lin_cube':lambda x:np.c_[x,x**3],'lin_sq':lambda x:np.c_[x,x*x],'sq':lambda x:x*x,'cyc':lambda x:cyc(x),'lin_cyc':lambda x:np.c_[x,cyc(x)],'cube_sq':lambda x:np.c_[x**3,x*x]}
P={}
for n,p in [('B',s['panel_b']),('C',s['panel_c'])]:
 u=p['u']; y=p['v' if n=='B' else 'v1']; L=[]
 for fn,f in ops.items():
  for en,g in base.items():
   D=lambda x,k,f=f,g=g:np.c_[lin(f(x*k)),g(x)];opt=minimize_scalar(lambda k:look(D(u,k),y)[1],bounds=(.25,5),method='bounded'); k=opt.x;a=D(u,k); b=ls(a,y);L.append((opt.fun,fn,en,k,lambda x,D=D,k=k,b=b:D(x,k)@b))
 for fn,f in ops.items():
  for en in ['0','lin','cube']:
   g=base[en]; D=lambda x,k,f=f,g=g:np.c_[lin(f(x*k)),g(x)]; opt=minimize(lambda k:look(D(u,k),y)[1],np.ones(4)*(1 if n=='B' else 2),method='Nelder-Mead',options={'maxiter':500}); k=np.abs(opt.x); a=D(u,k);b=ls(a,y);L.append((opt.fun,fn,en,k,lambda x,D=D,k=k,b=b:D(x,k)@b))
 # Sparse separable deformable shared transfer using per-axis spline-free feature recovery.
 for fn,f in ops.items():
  for extra in ['lin','cube','0']:
   g=base[extra]
   def D(x,k,f=f,g=g):return np.c_[lin(f(x*k[0])+k[1]*x+k[2]*x**3),g(x)]
   for k0 in [0.9,2.2,3.5]:
    opt=minimize(lambda k:look(D(u,k),y)[1],[k0,0,0],method='Nelder-Mead',options={'maxiter':350});k=opt.x;a=D(u,k);b=ls(a,y);L.append((opt.fun,fn,'shared'+extra,k,lambda x,D=D,k=k,b=b:D(x,k)@b))
 L.sort(key=lambda t:t[0]);print(n,[(round(e,6),f,g,np.round(k,4)) for e,f,g,k,_ in L[:12]]);P[n]=L[:15]
def design(b,c):return lin(np.c_[b,c,(b[:,:,None]*c[:,None,:]).reshape(len(b),-1)])
def evaluate(F,ridge=0):
 a=F[:52]; b=Y[:52]; scales=np.maximum(np.std(a,axis=0),1e-4);scales[0]=1;A=a/scales;pen=np.eye(A.shape[1]);pen[0,0]=0;coef=ls(np.r_[A,ridge*pen],np.r_[b,np.zeros((A.shape[1],Y.shape[1]))])/scales[:,None];E=np.abs(F[52:N]@coef-Y[52:]);return E[:,0].mean(),E[:,1:].mean(),coef
L=[]
for i,a in enumerate(P['B']):
 for j,b in enumerate(P['C']):
  for k in [-.15,0,.06,.13,.22]:
   for r in [0,.03,.12,.35]:
    F=design(a[-1](z+k*z**3),b[-1](z+k*z**3));ey,et,co=evaluate(F,r);L.append((ey,et,i,j,k,r))
print('leader',sorted(L)[:14]);print('joint',sorted(L,key=lambda a:a[0]+a[1])[:10]);
for i,j in [(0,0),tuple(sorted(L)[0][2:4])]:
 F=design(P['B'][i][-1](z),P['C'][j][-1](z));ey,et,co=evaluate(F);print('coefs',i,j,ey,et,co[:,[0,-1]].T);print('sv interaction',*[np.linalg.svd(co[9:,j].reshape(4,4),compute_uv=False) for j in [0,-1]]);
print('times',s['eval_times'],'telemetry',s['telemetry_vec'])
