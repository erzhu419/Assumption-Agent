import pickle,runpy,numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
old=runpy.run_path('/tmp/hegel_continual_archive/beta_m100_c04_s00_solver.py')
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=i['study']; pulses=i['campaign_diagnostics']; nx,ny=s['nx'],s['ny']; sizes=(nx,ny)
rows=s['sensor_ij'][:,0]+nx*s['sensor_ij'][:,1]; E=sparse.eye(nx*ny,format='csc'); V=sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); axes=[]
for a,n in zip('xy',sizes):
 h=s['L'+a]/(n-1); kap=s['kappa_'+a]; vel=s['vel_'+a]; rob=s['robin_'+a]; axes.append(old['_axis'](kap,vel,h,rob).toarray()); print(a,'h,rob,endpoint kappa,velocity',h,rob,kap[[0,-1]],vel[[0,-1]])
W=np.column_stack([p['w_pulse'] for p in pulses]); Y=np.column_stack([old['_calibrate'](dict(s,y_obs_raw=p['y_pulse_raw'])) for p in pulses]); rhs=np.column_stack([mode.ravel(order='F') for mode in s['source_modes']])@W

def forward(ax,ay):
 L=sparse.kron(sparse.eye(ny),sparse.csc_matrix(ax),format='csc')+sparse.kron(sparse.csc_matrix(ay),sparse.eye(nx),format='csc')+V
 return np.concatenate([splu(L+mu*E).solve(rhs)[rows] for mu in s['mu_shifts']])

def stencils(x,kind):
 blocks=[ax.copy() for ax in axes]; q=0
 for j,(a,n,B) in enumerate(zip('xy',sizes,blocks)):
  h=s['L'+a]/(n-1); robin=s['robin_'+a]/h
  for wall in range(2):
   end,d=(0,1) if wall==0 else (n-1,-1)
   if kind in ('sealed','robin','robin_axis','robin_side'):
    coef=0. if kind=='sealed' else x[0 if kind=='robin' else j if kind=='robin_axis' else 2*j+wall]
    B[end,end]+=(coef-1)*robin
   else:
    cols=[end,end+d]
    if kind in ('wrap','both'): cols.append(n-1-end)
    if kind in ('second','both'): cols.append(end+2*d)
    B[end,cols]+=x[q:q+len(cols)]; q+=len(cols)
 return blocks

def calc(x,kind): return forward(*stencils(x,kind))-Y
err=forward(*axes)-Y
print('baseline prediction range', (Y+err).min(),(Y+err).max(),'signed mean observed-minus-predicted by shift',-err.reshape(3,-1).mean(1))
print('sealed pulse rmse',np.sqrt(np.mean(calc([], 'sealed')**2,axis=0)))
for kind,p in [('robin',1),('robin_axis',2),('robin_side',4),('local',8),('wrap',12),('second',12),('both',16)]:
 x0=np.ones(p) if kind.startswith('robin') else np.zeros(p)
 def fit(keep): return least_squares(lambda x: calc(x,kind)[:,keep].ravel(),x0,max_nfev=160,ftol=1e-11,gtol=1e-11,xtol=1e-11)
 try:
  z=fit(slice(None)); e=calc(z.x,kind); cv=[np.sqrt(np.mean(calc(fit([j for j in range(len(pulses)) if j!=k]).x,kind)[:,k]**2)) for k in range(len(pulses))]; print(kind,'rmse',np.sqrt(np.mean(e**2,axis=0)),'held out',cv,'delta',np.round(z.x,6),flush=True)
 except (RuntimeError,ValueError) as exc: print(kind,exc,flush=True)
