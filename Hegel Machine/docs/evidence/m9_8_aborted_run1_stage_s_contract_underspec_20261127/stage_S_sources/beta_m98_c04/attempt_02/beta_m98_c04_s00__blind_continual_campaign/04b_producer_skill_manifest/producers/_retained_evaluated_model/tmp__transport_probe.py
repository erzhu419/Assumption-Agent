import numpy as np,pickle,itertools
from scipy.sparse import diags,kron,eye
from scipy.sparse.linalg import spsolve
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
cal=s['sensor_calibration_sweeps']; x=cal['ref_rosette']; y=cal['raw_rosette']; cf=np.linalg.lstsq(np.c_[x,np.ones(len(x))],y,rcond=None)[0]; print('affine fit rmse',np.mean((np.c_[x,np.ones(len(x))]@cf-y)**2)**.5, 'map',cf.tolist())
def inv(y):return np.linalg.solve(cf[:4].T,(y.reshape(-1,4)-cf[4]).T).T.ravel()
nx=s['nx']; ny=s['ny']; obs=np.array([inv(a['y_pulse_raw']) for a in s['reference_tracer_pulses']]).T; w=np.array([a['w_pulse'] for a in s['reference_tracer_pulses']]); rhs=np.column_stack([a.ravel(order='F') for a in np.tensordot(w,s['source_modes'],axes=1)]); idx=s['sensor_ij']@np.array([1,nx]); Z=s['Z_nuisance']
def axis(ax,sp,avg,adv,robs):
 k=s['kappa_'+ax]; v=s['vel_'+ax]; n=len(k); h=s['L'+ax]/(n-sp); a,b=k[:-1],k[1:]; d={'arith':(a+b)/2,'harm':2*a*b/(a+b),'geom':np.sqrt(a*b),'left':a,'right':b}[avg]/h**2
 l=-d.copy(); u=-d.copy(); main=np.r_[d,0]+np.r_[0,d]; vf=(v[1:]+v[:-1])/2/h; vp=np.maximum(vf,0); vm=np.minimum(vf,0)
 if adv.startswith('flux'):
  l-=vp; u+=vm; main+=np.r_[vp,0]-np.r_[0,vm];
  if adv=='fluxout':main[0]+=max(-v[0],0)/h; main[-1]+=max(v[-1],0)/h
  if adv=='fluxfull':main[0]-=v[0]/h; main[-1]+=v[-1]/h
 elif adv.startswith('local'):
  v=(v if adv=='local' else np.r_[vf[0]*h,(vf[:-1]+vf[1:])*h/2,vf[-1]*h])/h; l-=np.maximum(v[1:],0); u+=np.minimum(v[:-1],0); main+=abs(v)
 elif adv=='noncon':
  l-=vp; u+=vm; main+=np.r_[0,vp]-np.r_[vm,0]
 main[[0,-1]]+=robs
 return diags([l,main,u],[-1,0,1],format='csc')
def pred(sp,avg,adv,r):
 a=kron(eye(ny),axis('x',sp,avg,adv,r[:2]))+kron(axis('y',sp,avg,adv,r[2:]),eye(nx))+diags(s['V_reaction_2d'].ravel(order='F')); I=eye(nx*ny,format='csc'); return np.concatenate([spsolve((a+mu*I).tocsc(),rhs)[idx] for mu in s['mu_shifts']]);
if __name__=='__main__':
 print('pulse RMS',np.mean(obs**2)**.5)
 results=[]
 for sp,avg,adv in itertools.product([0,1],['arith','harm','geom','left','right'],['flux','fluxout','fluxfull','local','localface','noncon','none']):
  f=lambda r:(pred(sp,avg,adv,r)-obs).ravel(); nom=np.repeat([s['robin_x'],s['robin_y']],2); fit=least_squares(f,nom,bounds=(0,100),gtol=2e-10); results.append((np.linalg.norm(fit.fun)/np.sqrt(obs.size),sp,avg,adv,fit.x.tolist(),np.linalg.norm(f(nom))/np.sqrt(obs.size)))
 for row in sorted(results)[:20]:print(row)
