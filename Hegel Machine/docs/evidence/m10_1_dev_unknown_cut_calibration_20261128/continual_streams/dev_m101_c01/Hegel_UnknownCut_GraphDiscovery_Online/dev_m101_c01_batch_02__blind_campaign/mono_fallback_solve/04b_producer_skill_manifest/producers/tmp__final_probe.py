import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import batch_solver as m
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; y=t['y']; z=t['traj']; ids=np.arange(len(x)); order=np.argsort(np.linalg.norm(x,axis=1)); rng=np.random.default_rng(511)
splits=[(ids[:56],ids[56:])]+[(p[:56],p[56:]) for p in [rng.permutation(ids) for _ in range(5)]]+[(order[:56],order[56:]),(np.r_[order[:49],order[65:]],order[49:65])]
a=m._calibrate(s['panel_a'],x)
coef=np.linalg.lstsq(np.c_[np.ones(24),s['panel_b']['u']],s['panel_b']['v'],rcond=None)[0]
b=a@coef[1:]+coef[0]; ai=(x-m._calibrate(s['panel_a'],np.zeros((1,4))))@np.linalg.inv(m._calibrate(s['panel_a'],np.eye(4))-m._calibrate(s['panel_a'],np.zeros((1,4))))
def gram(f):return f@f.T
def phi(x):return np.column_stack([m._poly(x,j) for j in range(3)])
def additional(a):
 p=gram(phi(a)); T=[]; Q=[]
 for c in (.6,1.3,2.):
  f=np.tanh(a*c); T.append(gram(phi(f))); Q.append(gram(phi(np.sin(a*c))))
 return np.stack([p,*T,*Q])
def anova(a):
 ds=(a[:,None,:]-a[None,:,:])**2; k=np.exp(-ds/2); terms=[k[:,:,i] for i in range(4)]; pairs=[terms[i]*terms[j] for i in range(4) for j in range(i)]; return np.stack([*terms,*pairs])
def test(name,K,mode):
 out=[]
 for i,j in splits:
  l=np.r_[i,j]; B=K[:,l[:,None],l]; q=m._temporal(s['eval_times'],z[i]); target=z[i]@q
  if mode=='scalar':
   pred=m._regress(B,y[i,None]).ravel(); out.append(np.mean(abs(pred-y[j]))); continue
  if mode=='separate':
   pred=np.c_[m._regress(B,target[:,:1]),m._regress(B,target[:,1:])]@q.T
  elif mode=='svdsep':
   _,_,vh=np.linalg.svd(target,full_matrices=False); c=target@vh.T; pred=np.c_[m._regress(B,c[:,:1]),m._regress(B,c[:,1:])]@vh@q.T
  else: pred=m._regress(B,target)@q.T
  out.append([np.mean(abs(pred-z[j])),np.mean(abs(pred[:,-1]-z[j,-1]))])
 print(name,np.round(np.array(out).T,5),flush=True)
base=m._blocks(a,'classic')
for n,K in [('raw',m._blocks(x,'classic')),('inv',m._blocks(ai,'classic')),('classic',base),('B-composed',m._blocks(b,'classic')),('quadratic+harmonic',np.concatenate([base,additional(a)[1:]])),('classic+anova',np.concatenate([base,anova(a)])),('classic+lowdegree',np.concatenate([base,[gram(m._poly(a,3)),gram(m._poly(a,4))]])),('wide',np.concatenate([base,[np.exp(-cdist(a,a,'sqeuclidean')/24),np.exp(-cdist(a,a,'sqeuclidean')/3.5)]]))]:
 test(n,K,'scalar')
 if n in ('classic','wide','quadratic+harmonic','classic+anova'):
  test(n+' joint',K,'joint'); test(n+' svdsep',K,'svdsep')
test('classic sep',base,'separate')
