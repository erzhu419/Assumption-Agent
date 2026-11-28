import numpy as np
from scipy.optimize import least_squares

def lin(f,y):return np.linalg.lstsq(f,y,rcond=1e-10)[0]
def one(x):return np.column_stack((np.ones(len(x)),x))
def prof(f,y,q,bounds):
 def r(q):
  a=f(q); return (a@lin(a,y)-y).ravel()
 q=least_squares(r,q,bounds=bounds,gtol=1e-11).x
 return q,lin(f(q),y)
def features(s,x):
 p=s['panel_a']; mask=np.linalg.norm(p['v'],axis=1)<4; u,v=p['u'][mask],p['v'][mask]
 def design(z,q):
  return np.stack([np.ones_like(z),z,z/(1+(q[0]*z)**2),np.sin(q[1]*z),np.sin(q[1]*np.roll(z,2,axis=1))],axis=-1)
 def fit(q):
  f=design(u,q);w=np.array([lin(f[:,j],v[:,j]) for j in range(4)]);return w,(f*w).sum(-1)-v
 q=least_squares(lambda q:fit(q)[1].ravel(),[1.2,2.4],bounds=([.5,1.7],[2,3.2]),gtol=1e-13).x
 w,r=fit(q)
 z=np.array([least_squares(lambda a:(design(a[None],q)[0]*w).sum(-1)-row,(row-w[:,0])/w[:,1],gtol=1e-12).x for row in x]);
 p=s['panel_c']; U=p['u']; cyc=lambda z:z*np.roll(z,-1,axis=1)
 g=one(cyc(z))@lin(one(cyc(U)),p['v1'])
 k,b=prof(lambda k:one(np.cos(k[0]*U)),p['v2'],[2.4],(1.7,3.2));h=one(np.cos(k*z))@b
 p=s['panel_b']; B=lin(one(p['u']),p['v']);b=one(z)@B
 return z,g,h,b,(q,np.std(r),k,B)
def temporal(s,idx):
 t=s['eval_times'];Y=s['train']['traj'][idx].T
 f=lambda q:-np.expm1(-t[:,None]*q)
 q,w=prof(f,Y,[.42,1.82],([.1,.9],[.85,3.5]));return f(q),w.T
def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.asarray(train['x'])
    n = len(x)
    z, g, h, _, _ = features(study, np.concatenate((x, study['test']['x'])))
    interaction = (g[:, :, None] * h[:, None, :]).reshape(len(z), 16)
    design = one(np.column_stack((z, g, h, interaction)))
    basis, amplitudes = temporal(study, slice(None))
    output = design[n:] @ lin(design[:n], np.column_stack((train['y'], amplitudes)))
    y = output[:, 0].copy()
    state = output[:, 1:] @ basis.T
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
