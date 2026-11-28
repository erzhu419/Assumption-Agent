exec(open('/tmp/sparse_readout.py').read().split('for base in')[0]); b={}; exec(Path('/holdout_task/_frozen_nodes/node_01.py').read_text(),b)
from scipy.optimize import least_squares
folds=folds + [np.argsort(np.linalg.norm(x,axis=1))[-14:]]
p=z[:,0]*z[:,2]; q=z[:,1]*z[:,3]; a=x[:,1]; d=x[:,3]
freq,amp,W=b['_fit_panel'](s['panel_b']['u'],s['panel_b']['v']); warp=x+amp*np.sin(freq*x)
print('parent keys',list(c),list(dct for dct in b['run_node']({'__input__':s,'node_00':c})))
print('panel C keys/shapes',[(k,np.shape(v)) for k,v in s['panel_c'].items()])
P=np.column_stack((np.ones(n),z)); Z=np.c_[P,p,q]; C=z[:,0]*a*a
M={
 'cubic':np.c_[Z,C],
 'cubic+x':np.c_[Z,C,x],
 'cubic+sine_x':np.c_[Z,C,x,np.sin(2*x)],
 'cubic+z0trig':np.c_[Z,C,z[:,0]*np.sin(2*a),z[:,0]*np.cos(2*a)],
 'cubic+contrast2':np.c_[Z,C,(a-d)**2],
 'cubic+products_x':np.c_[Z,C,x[:,0]*x[:,2],x[:,1]*x[:,3]],
 'cubic+warpedmods':np.c_[Z,C,z[:,0]*warp[:,1]**2,z[:,0]*a*warp[:,1]],
 'cubic+calquad':np.c_[Z,C,poly(x,2)],
 'cubic+opquad':np.c_[P,C,poly(z,2)],
 'cubic+harmonic':np.c_[Z,C,np.sin(3*(z[:,1]-z[:,3]))],
 'cubic+term_diagnostic':np.c_[Z,C,T[:,-1]],
}
for name,B in M.items():
 E=[]; coef=[]
 for v in folds:
  t=np.setdiff1d(idx,v)
  def fit(par,ret=False):
   f,g=par; A=np.c_[B,np.sin(f*(a-g*d))]; w=np.linalg.lstsq(A[t],y[t],rcond=None)[0]; r=A@w-y
   return (np.mean(abs(r[v])),w) if ret else r[t]
  opt=min((least_squares(fit,[f,1],bounds=([.55,.25],[4.5,2.5]),max_nfev=50) for f in [1.5,2.8]),key=lambda h:h.cost); e,w=fit(opt.x,True); E.append(e); coef.append(np.r_[opt.x,w])
 print(name,np.round(E,4),'coef median',np.round(np.median(coef,0),3))
for name,G in [('op',P),('cross',np.c_[P,x]),('op_quad',np.c_[P,poly(z,2)]),('cal_correction',np.c_[P,np.sin(2*x)]),('cross_cubic',np.c_[P,x,x**2,x**3])]:
 E=[]
 for v in folds:
  t=np.setdiff1d(idx,v); r=G[v]@np.linalg.lstsq(G[t],T[t],rcond=None)[0]-T[v]; E.append(np.mean(abs(r),0))
 print('trajectory',name,np.round(np.mean(E,1),5),'worst-time',np.round(np.max(E,0),5))
