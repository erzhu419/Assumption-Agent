import pickle,itertools as it
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; n=52
fit=lambda f,y: np.linalg.lstsq(f,y,rcond=1e-9)[0]
rms=lambda z:np.sqrt(np.mean(z*z)); mae=lambda z:np.mean(abs(z))
bias=lambda z:np.c_[np.ones(len(z)),z]
p=s['panel_a']; u,v=p['u'],p['v']; a=bias(u); mat=fit(a,v); err=np.linalg.norm(a@mat-v,axis=1); mask=err<4*np.median(err)
def design(q): return np.c_[bias(q),q*np.roll(q,-1,axis=1),q**3]
w=fit(design(u[mask]),v[mask]); print('calibration coefficients',w,'sparse model fitted RMS',rms(design(u[mask])@w-v[mask]));
raw=np.r_[tr['x'],s['test']['x']]; guess=np.linalg.solve(mat[1:].T,(raw-mat[0]).T).T
x=np.array([least_squares(lambda z: design(z[None,:])[0]@w-r,g,gtol=1e-12).x for r,g in zip(raw,guess)]); print('inverse RMS',rms(design(x)@w-raw),'max move',np.max(np.abs(x-guess),axis=0),'latent bounds',x.min(0),x.max(0)); x=x[:len(tr['x'])]
bp=s['panel_b']; wb=fit(bias(bp['u']),bp['v']); b=bias(x)@wb; bi=(x-wb[0])@np.linalg.inv(wb[1:]); print('B orthogonality',wb[1:]@wb[1:].T)
cp=s['panel_c']; u=cp['u'];
def bil(z):
 a,b,c,d=z.T; return bias(np.c_[a*b,a*d,b*c,c*d])
wc=fit(bil(u),cp['v1']); print('C coefficients',wc)
def cb(z): return bil(z)@wc
def cbi(z): return np.linalg.lstsq(wc[1:].T,(z-wc[0]).T,rcond=None)[0].T
v=cp['v2']
def cos(z,w):return bias(np.c_[z,np.cos(w*z)])
def v2_err(w):
 f=cos(u,w);return rms(f@fit(f,v)-v)
freq=minimize_scalar(v2_err,bounds=(.7,1.8),method='bounded').x; wd=fit(cos(u,freq),v); print('D coefficients',freq,wd)
def cb2(z):return cos(z,freq)@wd
Y=tr['y']; T=tr['traj']; tb=np.c_[-np.expm1(-1.8312*t), t/(1+.40745*t)]; amp=fit(tb,T[:n].T).T

def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def matched(a,b):return (a*b).sum(1,keepdims=True)
def poly(a):return np.column_stack([a[:,i]*a[:,j] for i,j in it.combinations_with_replacement(range(a.shape[1]),2)])
base={'X':x,'B':b,'Binv':bi,'C(X)':cb(x),'D(X)':cb2(x),'C(B)':cb(b),'D(B)':cb2(b),'C(Binv)':cb(bi),'D(Binv)':cb2(bi)}
print('individual bases: name, scalar val, traj val, scalar fit, fast amp fit, slow amp fit')
rows=[]
def check(name,f):
 f=bias(f); y=Y-f@fit(f[:n],Y[:n]); yy=T-f@fit(f[:n],T[:n]); ae=amp-f[:n]@fit(f[:n],amp)
 rows.append((name,mae(y[n:]),mae(yy[n:]),mae(y[:n]),mae(ae[:,0]),mae(ae[:,1])))
for k,z in base.items():check(k,z);check(k+'+quad',np.c_[z,poly(z)])
for k,l in it.combinations(base,2):
 if k in ('X','B','Binv') and l in ('X','B','Binv'):continue
 a,d=base[k],base[l]; label=k+' '+l
 check(label,np.c_[a,d]);check(label+' dot',np.c_[a,d,matched(a,d)]);check(label+' dot+01',np.c_[a,d,matched(a,d),a[:,0]*d[:,1]]);check(label+' cross',np.c_[a,d,cross(a,d)])
for j in range(1,6):
 print('rank',j)
 for r in sorted(rows,key=lambda r:r[j])[:8]:print(r[0],np.round(r[1:],6))
for k in ('X','B','Binv'):
 z=base[k]; c,d=cb(z),cb2(z); dot=matched(c,d); cr=c[:,0]*d[:,1]; f=bias(np.c_[c,d,dot]); yhat=f@fit(f[:n],Y[:n]); g=bias(np.c_[c,dot]); h=bias(np.c_[d,cr]);
 def mat(r):return np.concatenate((g[:,None,:]*(-np.expm1(-r[0]*t))[None,:,None],h[:,None,:]*(t/(1+r[1]*t))[None,:,None]),axis=2).reshape(len(z)*len(t),-1)
 def resid(r):
  a=mat(r)[:n*len(t)]; q=T[:n].ravel(); return a@fit(a,q)-q
 opt=least_squares(resid,[1.8,.4],bounds=([.05,.001],[6,2])); a=mat(opt.x); state=(a@fit(a[:n*len(t)],T[:n].ravel())).reshape(T.shape)
 print('arch form',k,opt.x,'fit',mae(yhat[:n]-Y[:n]),mae(state[:n]-T[:n]),'val',mae(yhat[n:]-Y[n:]),mae(state[n:]-T[n:]),'y coef',fit(f[:n],Y[:n]))
