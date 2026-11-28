import json,pickle,sys,itertools
import numpy as np
from scipy.sparse import diags,eye,kron,lil_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; pay=json.load(open('/holdout_task/_upstream_param_payload.json')); pulses=item['campaign_diagnostics']; W=np.array([p['w_pulse'] for p in pulses]).T
Y=np.column_stack([calibrate_upstream(dict(s,y_obs_raw=p['y_pulse_raw']),pay)['y_obs_raw'] for p in pulses]); Z=s['Z_nuisance']; ZQ=np.linalg.qr(Z)[0]
nx,ny=s['nx'],s['ny']; I=eye(nx*ny,format='csc'); R=diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); rhs=np.column_stack([a.ravel(order='F') for a in s['source_modes']]) @ W
ij=s['sensor_ij']; slots=ij[:,0]+nx*ij[:,1]
def axis(a,n,leak,conduct,wrap):
 k=s['kappa_'+a]; h=s['L'+a]/(n-1); v=s['vel_'+a][1:-1]/h; r=s['robin_'+a]/h; f=(k[:-1]+k[1:])/(2*h*h)
 l=np.r_[-f[:-1]-np.maximum(v,0),-conduct[1]*k[-1]/h**2]; u=np.r_[-conduct[0]*k[0]/h**2,-f[1:]+np.minimum(v,0)]; w=wrap*(k[0]+k[-1])/(2*h*h)
 mat=diags((l,np.r_[-u[0]+r*leak[0]+w[0],f[:-1]+f[1:]+abs(v),-l[-1]+r*leak[1]+w[1]],u),(-1,0,1),format='lil'); mat[0,-1]=-w[0]; mat[-1,0]=-w[1]; return mat.tocsc()
def pred(leak,conduct=np.ones(4),wrap=np.zeros(4)):
 x,y=[axis(a,n,leak[j:j+2],conduct[j:j+2],wrap[j:j+2]) for a,n,j in [('x',nx,0),('y',ny,2)]]
 op=R+kron(eye(ny),x,format='csc')+kron(y,eye(nx),format='csc'); return np.vstack([splu(op+mu*I).solve(rhs)[slots] for mu in s['mu_shifts']])
def resid(l,c=np.ones(4),w=np.zeros(4),project=False):
 r=pred(l,c,w)-Y
 return r-ZQ@(ZQ.T@r) if project else r
def rms(r): return np.sqrt(np.mean(r*r,axis=0))
if __name__=='__main__':
 for flags in itertools.product([0.,1.],repeat=2):
  for wrap in [np.zeros(4),np.ones(4),np.r_[1.,1.,0.,0.],np.r_[0.,0.,1.,1.]]:
   l=np.repeat(flags,2); print('discrete',l,wrap,rms(resid(l,w=wrap)),rms(resid(l,w=wrap,project=True)))
 for project in [False,True]:
  for d in [4,8,12]:
   def func(t): return resid(t[:4],t[4:8] if d>=8 else np.ones(4),t[8:] if d==12 else np.zeros(4),project)
   for held in [-1,0,1,2]:
    train=np.arange(3)!=held
    fit=least_squares(lambda t:func(t)[:,train].ravel(),np.r_[[.2]*4,[1.]*(min(d,8)-4),[.1]*max(0,d-8)],bounds=(0,10),gtol=1e-12,ftol=1e-12,xtol=1e-12)
    print('fit',project,d,held,np.round(fit.x,6),rms(func(fit.x)))
 sw=s['sensor_calibration_sweeps']; raw=sw['raw_rosette']; ref=sw['ref_rosette']; print('rosette shape',raw.shape,'residual quantiles',np.quantile(abs(raw-ref*np.array(pay['gains'])-np.array(pay['offsets'])),[0,.5,.9,.99,1]),'all-bank OLS'); ok=(abs(raw)<=8.5).all(axis=1)
 print(np.array([np.linalg.lstsq(np.c_[np.ones(ok.sum()),ref[ok,j]],raw[ok,j],rcond=None)[0] for j in range(4)])); print('single',[(k,np.shape(v)) for k,v in sw['single_bank_sweeps'].items()]); print('banks',s['row_bank_id'][:20])
