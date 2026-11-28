import contextlib,io,glob,runpy
with contextlib.redirect_stdout(io.StringIO()): from probe import *
a=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*01_solver.py')[0]); pa=s['panel_a'];c=a['_calibration'](pa);u=pa['u'];v=pa['v'];q=np.roll(u,-1,axis=1);print('cal',c,np.sort(np.linalg.norm(c[0]+c[1]*u+c[2]*q+c[3]*u*q+c[4]*u**3-v,axis=1)));z=a['_inverse'](x,c)
Y=np.column_stack([y,t]);
def score(F):
 es=[]
 for tr,te in [(np.arange(52),np.arange(52,72)),(np.where(np.arange(72)%3!=0)[0],np.arange(0,72,3)),(np.arange(20,72),np.arange(20))]:
  f=F[tr]; g=F[te]; coef=ls(f,Y[tr]); err=abs(g@coef-Y[te]);es.append(np.r_[err[:,0].mean(),err[:,1:].mean()]);
 return np.round(np.array(es),4)
def chk(name,F):print(name,F.shape[1],score(F).ravel(), 'train',np.mean(abs(F@ls(F,Y)-Y),axis=0)[[0,-1]])
for deg in range(1,5):chk('latent '+str(deg),poly(z,deg))
for p,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 U=s[p]['u'];V=s[p][k];
 for deg in (1,2,3):
  P=poly(U,deg);C=ls(P,V);F=poly(z,deg)@C; chk(p+k+str(deg),poly(F,2));print('panel err',np.mean(abs(P@C-V),axis=0));
  if deg==2: print(C)
