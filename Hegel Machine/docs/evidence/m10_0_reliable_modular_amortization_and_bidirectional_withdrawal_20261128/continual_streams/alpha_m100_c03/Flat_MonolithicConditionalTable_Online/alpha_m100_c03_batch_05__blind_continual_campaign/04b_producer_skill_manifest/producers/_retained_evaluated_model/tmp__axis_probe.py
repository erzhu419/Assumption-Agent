import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; s=c['single_channel_sweeps']; y=c['true_channel_loads']; r=c['bridge_raw']; m=abs(s['cmd_stress'])<5; x=s['true_stress'][m]; j=s['lab_id'][m]; v=s['cmd_stress'][m];
F={'tanh':np.tanh,'atan':np.arctan,'erf':erf,'rat':lambda x:x/(1+x*x),'gaussian':lambda x:x*np.exp(-x*x),'sqrt':lambda x:x/np.sqrt(1+x*x),'asinh':np.arcsinh,'sin':np.sin}
def prof(p,f,powers):
 a=np.column_stack([np.ones(len(x)),f(p[0]*x)]+[x**k for k in powers]); mat=(a[:,:,None]*(j[:,None,None]==np.arange(4))).reshape(len(x),-1); b=np.linalg.lstsq(mat,v,rcond=None)[0].reshape(-1,4); return a, v-mat@b.ravel(),b
for n,f in F.items():
 for powers in [(3,),(1,3),(1,5),(1,3,5)]:
  o=min((least_squares(lambda p:prof(p,f,powers)[1],[start],bounds=(.05,5)) for start in [.4,1.2,2]),key=lambda o:np.linalg.norm(o.fun)); print(n,powers,o.x.round(5),np.linalg.norm(o.fun).round(6),prof(o.x,f,powers)[2][:,0].round(5))
for n in ['tanh','atan','erf','rat']:
 f=F[n]; o=least_squares(lambda p:prof(p,lambda z:f(z)+0*z,(1,3))[1],[1],bounds=(.05,5));
print('calibration abs raw maxima sorted',np.sort(abs(r).max(1))[-12:]); print('specimen abs raw max',[(k,abs(D[k]['past_channel_raw']).max(),abs(D[k]['hold_channel_raw']).max()) for k in ['train','test']])
