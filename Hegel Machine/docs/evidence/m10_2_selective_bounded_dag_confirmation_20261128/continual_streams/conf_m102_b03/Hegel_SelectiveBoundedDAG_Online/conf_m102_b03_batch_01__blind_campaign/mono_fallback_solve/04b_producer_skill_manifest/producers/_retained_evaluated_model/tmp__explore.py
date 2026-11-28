import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=150)
def ones(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
i,j=np.triu_indices(4)
def quad(x):return ones(np.c_[x,x[:,i]*x[:,j]])
a=s['panel_a']; u,v=a['u'],a['v']; f=ones(u); r=np.linalg.norm(f@ls(f,v)-v,axis=1); g=r<3; A=ls(f[g],v[g]);print('A',A,'inlier max',abs(f[g]@A-v[g]).max()); print('A residual rows', (v-f@A)[g][:6]);print('reverse',abs(ones(v[g])@ls(ones(v[g]),u[g])-u[g]).max())
b=s['panel_b'];B=ls(ones(b['u']),b['v']);c=s['panel_c'];C=ls(quad(c['u']),c['v1']); print('B',B,'C1',C,sep='\n'); print('C2 u v',np.c_[c['u'],c['v2']][:12]);print('times',s['eval_times'],'tele',s['telemetry_vec'],'summary',s['train']['summary']); x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; print('x y t',np.c_[x,y,t][:10]); print('test ranges',s['test']['x'].min(0),s['test']['x'].max(0));print('svd',np.linalg.svd(t,compute_uv=False));print('svd centered',np.linalg.svd(t-t.mean(0),compute_uv=False)); print('ratio',t[:8]/t[:8,-1:]);
Y=np.c_[y,t];z=(x-A[0])@np.linalg.inv(A[1:]);
for name,x0 in [('raw',x),('invA',z),('fwdA',ones(x)@A)]:
 q=ones(x0)@B; c1=quad(x0)@C; qc=quad(q)@C
 print(name,'range',x0.min(0),x0.max(0))
 for key,F in [('aff',ones(x0)),('poly2',quad(x0)),('tanhB',ones(np.tanh(q))),('tanhBquad',quad(np.tanh(q))),('B_C1',ones(qc)),('C1quad',quad(c1)),('B_C1quad',quad(qc)),('C1_C1quad',quad(quad(c1)@C)),('q_x_C1',ones(np.c_[q,c1,(q[:,:,None]*c1[:,None,:]).reshape(len(q),-1)])),('q_x_QC1',ones(np.c_[q,qc,(q[:,:,None]*qc[:,None,:]).reshape(len(q),-1)]))]:
  p=F@ls(F[:52],Y[:52]); print(key,F.shape[1],np.abs(p[52:]-Y[52:]).mean(0)[[0,1,8]],'train',np.abs(p[:52]-Y[:52]).mean(0)[[0,8]])
