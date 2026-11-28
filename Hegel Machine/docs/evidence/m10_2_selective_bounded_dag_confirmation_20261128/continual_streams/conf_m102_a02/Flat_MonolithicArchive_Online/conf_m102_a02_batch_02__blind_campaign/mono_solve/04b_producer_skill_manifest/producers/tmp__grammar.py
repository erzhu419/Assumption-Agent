import numpy as np,pickle,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; n=len(x)
def aff(x):return np.c_[np.ones(len(x)),x]
def op(p):
 F=aff(p['u']); v=p['v']; c=np.linalg.lstsq(F,v,rcond=None)[0]
 for i in range(4):
  r=np.linalg.norm(F@c-v,axis=1); m=r<max(.001,3*np.median(r)); c=np.linalg.lstsq(F[m],v[m],rcond=None)[0]
 return c
A=op(s['panel_a']); B=op(s['panel_b']); print('A',A,'B',B)
Z={};Z['raw']=x;Z['cal']=(x-A[0])@np.linalg.inv(A[1:]);Z['rev']=aff(x)@A;Z['Braw']=aff(x)@B;Z['Bcal']=aff(Z['cal'])@B;Z['Bical']=(Z['cal']-B[0])@np.linalg.inv(B[1:]);Z['Brev']=aff(Z['rev'])@B
I=np.arange(52);J=np.arange(52,n)
def score(F,target=y,reg=0):
 if reg:
  P=np.diag(np.broadcast_to(reg,(F.shape[1],)).copy());P[:5,:5]=0;c=np.linalg.solve(F[I].T@F[I]+P,F[I].T@target[I]);
 else:c=np.linalg.lstsq(F[I],target[I],rcond=None)[0]
 r=F@c-target;return np.mean(abs(r[I])),np.mean(abs(r[J])),c
p2=list(itertools.combinations_with_replacement(range(4),2));p3=list(itertools.combinations_with_replacement(range(4),3));pr=list(itertools.combinations(range(4),2))
def terms(z,p):return np.column_stack([np.prod(z[:,i],axis=1) for i in p])
def r(v):return tuple(round(a,6) for a in v[:2])
for k,z in Z.items():
 Q=terms(z,p2); C=terms(z,p3); f=np.c_[aff(z),Q,C]; out=score(f); print(k,'cubic',r(out),'Q',out[2][5:15].round(3),'C',out[2][15:].round(3)); print('  restricted',[(name,r(score(np.c_[aff(z),Q,h]))) for name,h in [('self',z**3),('triple',terms(z,list(itertools.combinations(range(4),3)))),('gauss',z*np.exp(-z*z)),('rational',z/(1+.75*z*z)),('tanh',np.tanh(z)),('sin',np.sin(z)),('no_self',C[:,[len(set(p))>1 for p in p3]])]])
 print('  cubic ridge',[r(score(f,reg=q)) for q in [.001,.01,.05,.2]])
print('trajectory coeff',np.linalg.lstsq(aff(Z['cal']),T,rcond=None)[0].round(5));print('train summary',s['train']['summary'])
