import importlib.util,pickle,numpy as np,itertools
from scipy.optimize import least_squares
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'beta_w03_s00_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
a=pickle.load(open(p+'public_beta_w03_s00.pkl','rb'))[0]; s,g=a['study'],a['gt']; U=m._responses(s); i,j=s['sensor_ij'].T; S=U[:,:,i,j].transpose(0,2,1).reshape(-1,U.shape[1]); Z=s['Z_nuisance']; y=s['y_obs_raw']; row=s['row_bank_id']; sw=s['sensor_calibration_sweeps']; ids=sw['bank_id']; x=sw['ref_signal']; cap=s['u_box']; budget=s['total_budget']; ridge=s['source_ridge']; qn=Z.shape[1]; md=S.shape[1]; zr=s['nuisance_ridge']
def profile(D,Z): return D-Z@np.linalg.solve(Z.T@Z+zr*np.eye(qn),Z.T@D)
def metrics(t):
 D=profile(np.column_stack((S,t)),Z); w,c=m._box_simplex(D[:,:-1],D[:,-1],ridge,cap,budget); F=np.einsum('m,mxy->xy',w,U[-1]); r=g['S_tilde']@w-g['r_y']; grad=g['S_tilde'].T@r+ridge*w; obj=.5*(r@r+ridge*(w@w)); nu=c['nu_sum']; lb=obj-grad@w+nu*budget+cap*np.minimum(0,grad-nu).sum(); gap=obj-min(lb,c['dual_lower_bound']); return np.array([np.max(abs(w-g['w_opt'])),np.linalg.norm(F-g['terminal_field_A'])/np.linalg.norm(g['terminal_field_A']),abs(c['primal_objective']-obj),gap,np.max(abs(grad-nu-c['lam_lower']+c['lam_upper']))])
# Exact source budget in the calibration fit, with centered source coefficients.
w0=np.full(md,budget/md); E=np.linalg.qr(np.column_stack((np.ones(md),np.eye(md)[:,:-1])))[0][:,1:]; A=np.column_stack((S@E,Z)); src=S@w0; penalty=np.zeros((md-1,A.shape[1])); penalty[:,:md-1]=np.sqrt(ridge)*np.eye(md-1); worst={}; passed={}
for sparse in [(0,1),(2,3),(0,),(2,),(0,1,2)]:
 for loc in ['low','middle','high']:
  for gains in [np.array([.91,1.12,1.17,.86]),np.array([.73,1.32,.82,1.19])]:
   sparse=np.array(sparse); train=~np.isin(row,sparse); offsets=np.array([.11,-.17,.21,-.08]); raw=gains[row]*y+offsets[row]; spots={b: {'low':min,'middle':lambda v:v[np.argmin(abs(v))],'high':max}[loc](x[ids==b]) for b in sparse}; B=np.column_stack([row==b for b in sparse]); x0=np.where(train,y, B@np.array([spots[b] for b in sparse])); q=np.where(train,0.,raw-B@np.array([gains[b]*spots[b]+offsets[b] for b in sparse])); Q=q[:,None]*B
   D=profile(np.column_stack((S[train],y[train])),Z[train]); w,_=m._box_simplex(D[:,:-1],D[:,-1],ridge,cap,budget); pred=S@w+Z@np.linalg.lstsq(Z[train],y[train]-S[train]@w,rcond=None)[0]; inv=(Q.T@(pred-x0))/np.sum(Q*Q,axis=0)
   zz=np.linalg.lstsq(np.vstack([np.column_stack((A,-Q)),np.column_stack((penalty,np.zeros((md-1,len(sparse)))))]),np.r_[x0-src,np.zeros(md-1)],rcond=None)[0]
   def forward(params):
    pred=src+A@params[:A.shape[1]]; gain=B@params[A.shape[1]:]; residual=np.where(train,pred-x0,gain*(pred-x0)-q); return np.r_[residual,penalty@params[:A.shape[1]]]
   fg=least_squares(forward,np.r_[zz[:A.shape[1]],1/zz[A.shape[1]:]],gtol=1e-11).x[A.shape[1]:]
   for name,v in [('identity',np.ones(len(sparse))),('held-out',inv),('joint-inverse',zz[A.shape[1]:]),('joint-forward',1/fg)]:
    val=metrics(x0+Q@v); worst[name]=np.maximum(worst.get(name,0),val); passed[name]=passed.get(name,0)+int(val[2]<2e-4 and val[3]<2e-4 and val[4]<1e-4)
   if len(sparse)==2 and loc=='middle' and gains[0]>.8: print('middle anchor wing',sparse,[(name,metrics(x0+Q@v)) for name,v in [('held-out',inv),('joint-inverse',zz[A.shape[1]:]),('joint-forward',1/fg)]])
print('worst weight, field, objective, reference gap, stationarity; passing out of 30')
for name,val in worst.items(): print(name,val,passed[name])
