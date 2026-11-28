import pickle,itertools,numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; s=c['single_channel_sweeps']; axis=(abs(x)>1e-10).sum(1)==1
# Separate a row-wide spike from the normal instrument span.
good=np.max(abs(y-np.median(y,axis=0)),axis=1)<8

def sweep(j):
 m=(s['lab_id']==j)&(abs(s['cmd_stress']-np.median(s['cmd_stress']))<8); a=axis&good&(x[:,j]!=0)&(x[:,j]!=-1.6)
 t=np.r_[s['true_stress'][m],x[a,j]]; z=np.r_[s['cmd_stress'][m],y[a,j]]; i=np.argsort(t)
 return t[i],z[i]

p=np.column_stack([CubicSpline(*sweep(j))(x[:,j]) for j in range(4)])
for j in range(4):
 z=(y-p)[good,j]; u=x[good]; k=(j+2)%4
 names=[(k,),(k,k)]+[p for d in (1,2,3,4) for p in itertools.combinations_with_replacement(range(4),d) if p not in [(k,),(k,k)] and len(set(p))>1 or d in (1,2,3,4) and len(set(p))==1 and p[0]!=j and p not in [(k,),(k,k)]]
 A=np.column_stack([np.prod(u[:,t],axis=1) for t in names]); ix=[0,1]
 for i in range(8):
  B=A[:,ix]; b=np.linalg.lstsq(B,z,rcond=None)[0]; r=z-B@b
  print(j,round(np.sqrt(np.mean(r*r)),6),list(zip([names[h] for h in ix],np.round(b,5))))
  v=A.T@r/np.linalg.norm(A,axis=0); v[ix]=0; ix.append(int(np.argmax(abs(v))))

def phi(t,q,kind):
 a,b=q; a=max(a,1e-6); b=max(b,1e-6)
 terms={'tanh_sinlin':[np.tanh(a*t),np.sin(b*t),t], 'tanh_rational2':[np.tanh(a*t),t/(1+b*t*t),t**3/(1+b*t*t)**2], 'tanh_sinrat':[np.tanh(a*t),np.sin(b*t)/(1+t*t)], 'tanh_stresswarp':[np.tanh(a*t+b*t**3),t,t**3], 'odd_pade':[t/(1+a*t*t+b*t**4),t**3/(1+a*t*t+b*t**4),t**5/(1+a*t*t+b*t**4)], 'tanh_sincub':[np.tanh(a*t),np.sin(b*t),t**3], 'atan_sinlin':[np.arctan(a*t),np.sin(b*t),t], 'tanh_cube':[np.tanh(a*t),np.tanh(a*t)**3,t], 'tanh_sinclip':[np.tanh(a*t),np.sin(b*t),t*np.abs(t)]}[kind]
 return np.column_stack([np.ones(len(t)),*terms])
for kind in ('tanh_sinlin','tanh_rational2','tanh_sinrat','tanh_stresswarp','odd_pade','tanh_sincub','atan_sinlin','tanh_cube','tanh_sinclip'):
 summary=[]
 for j in range(4):
  t,z=sweep(j)
  def residual(q):
   A=phi(t,q,kind); return A@np.linalg.lstsq(A,z,rcond=None)[0]-z
  opt=min((least_squares(residual,g,bounds=(.04,6),max_nfev=150) for g in ([.5,.4],[1.3,2.5],[2,1],[1,4])),key=lambda o:o.cost)
  A=phi(t,opt.x,kind); summary.append((round(np.sqrt(np.mean(opt.fun**2)),6),np.round(np.r_[opt.x,np.linalg.lstsq(A,z,rcond=None)[0]],5).tolist()))
 print(kind,summary)
