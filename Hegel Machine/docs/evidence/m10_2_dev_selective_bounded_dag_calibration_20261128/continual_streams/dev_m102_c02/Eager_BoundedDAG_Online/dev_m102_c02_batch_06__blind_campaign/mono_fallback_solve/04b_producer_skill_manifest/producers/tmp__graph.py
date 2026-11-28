import pickle,numpy as np
from scipy.optimize import least_squares,minimize_scalar,root
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; L=lambda z,y:np.linalg.lstsq(z,y,rcond=None)[0]; F=lambda x:np.c_[np.ones(len(x)),x]; T=s['train']['traj']; y=s['train']['y']; xx=np.r_[s['train']['x'],s['test']['x']]; t=s['eval_times']; print('summary',s['train']['summary'],'telemetry',s['telemetry_vec'])
u,v=[s['panel_a'][k] for k in ('u','v')]; keep=np.max(abs(v-u),1)<2;u,v=u[keep],v[keep];
def cal(x,j,extra='none'):
 a,b=x[:,j],x[:,(j+1)%4]; c=x[:,(j+2)%4]; d=x[:,(j+3)%4]; z=np.c_[a,b,a*b]; extras={'none':[], 'triple':[a*b*c,a*b*d,a*c*d,b*c*d], 'cube':[a*a*b,a*b*b], 'rat':[a*b/(1+.22*a*a),a*b/(1+.22*b*b)], 'cos':[np.cos(1.25*a),np.cos(1.25*b)], 'sin':[np.sin(1.25*a),np.sin(1.25*b)],'other':[c,d,a*c,a*d,b*c,b*d,c*d]}; return F(np.column_stack([z]+extras[extra]))
for e in ['none','triple','cube','rat','cos','sin','other']:
 print('A',e,[(round(np.mean(abs((v[:,j]-z@L(z,v[:,j]))/(1-np.sum(np.linalg.qr(z)[0]**2,1)))),5),np.round(L(z,v[:,j]),3)) for j in range(4) for z in [cal(u,j,e)]])
def mkcal(e):
 bs=[L(cal(u,j,e),v[:,j]) for j in range(4)]; return lambda x:np.column_stack([cal(x,j,e)@bs[j] for j in range(4)])
X={}
for e in ['none','triple','cube','rat','cos','sin']:
 fun=mkcal(e); X[e]=np.array([root(lambda q:fun(q[None,:])[0]-p,p).x for p in xx]); print('inverse',e,np.max(abs(fun(X[e])-xx)))
ops={}
for k,l,h in [('panel_b','v',np.tanh),('panel_c','v1',np.tanh),('panel_c','v2',np.cos)]:
 p=s[k]; U,V=p['u'],p[l]; feat=lambda z,a,h=h:F(np.c_[z,h(a*z)]); a=minimize_scalar(lambda a:np.linalg.norm(V-feat(U,a)@L(feat(U,a),V)),bounds=(.4,2),method='bounded').x; B=L(feat(U,a),V); print('op',k,l,a,np.mean(abs(V-feat(U,a)@B))); ops[l]=lambda z,a=a,B=B,feat=feat:feat(z,a)@B
print('y summary',np.quantile(y,[0,.25,.5,.75,1])); print('traj loc',T.min(0),T.max(0)); Y=np.c_[y,T];
def score(z):
 r=Y-z[:72]@L(z[:52],Y[:52]); return np.mean(abs(r[52:,0])),np.mean(abs(r[52:,1:]),0)[-1],np.mean(abs(r[:52,0]))
for e,x in X.items():
 vals={'x':x}; vals.update({k:f(x) for k,f in ops.items()});
 for k,z in list(vals.items())[1:]:
  for k2,f in ops.items(): vals[k+'>'+k2]=f(z)
 A=[]
 for k,z in vals.items():
  for name,q in [('linear',z),('quad',np.c_[z,*[z[:,i]*z[:,j] for i in range(4) for j in range(i,4)]]),('tanh',np.c_[z,np.tanh(1.25*z)]),('mix',np.c_[x,z,z*z])]:
   if k=='x' and name=='mix':continue
   A.append((score(F(q)),k,name))
 print('CV',e,'y',sorted(A)[:5],'state',sorted(A,key=lambda z:z[0][1])[:5])
