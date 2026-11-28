import pickle, numpy as np
from scipy.optimize import least_squares
from pathlib import Path
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; ns={}; exec(Path('/holdout_task/_frozen_nodes/node_01.py').read_text(),ns); sink=ns['run_node']
ix=np.arange(4); jx=(ix+1)%4
lin=lambda u:np.column_stack((np.ones(len(u)),u))
pa=s['panel_a']; u=pa['u']; v=pa['v']; w=np.ones(len(u))
for _ in range(25):
 co=np.array([np.linalg.lstsq(np.column_stack((np.ones(len(u)),u[:,i],u[:,j],u[:,i]*u[:,j],u[:,i]**3))*w[:,None],v[:,i]*w,rcond=None)[0] for i,j in zip(ix,jx)]).T
 r=np.linalg.norm(co[0]+u*co[1]+u[:,jx]*co[2]+u*u[:,jx]*co[3]+u**3*co[4]-v,axis=1); w=1/np.sqrt(1+(r/max(1e-6,1.5*np.median(r)))**2)
forward=lambda u:co[0]+u*co[1]+u[jx]*co[2]+u*u[jx]*co[3]+u**3*co[4]
x=s['train']['x']; U=np.array([least_squares(lambda u:forward(u)-z,z-co[0],gtol=1e-11).x for z in x]); pb=s['panel_b']; bc=np.linalg.lstsq(lin(pb['u']),pb['v'],rcond=None)[0]; B=lin(U)@bc
pc=s['panel_c']; c1=lambda u:np.column_stack((np.ones(len(u)),u*u[:,jx])); c2=lambda u:np.column_stack((np.ones(len(u)),np.cos(2.4*u)))
b1=np.linalg.lstsq(c1(pc['u']),pc['v1'],rcond=None)[0]; b2=np.linalg.lstsq(c2(pc['u']),pc['v2'],rcond=None)[0]
F={'u':U,'x':x,'b':B,'q':(U-bc[0])@np.linalg.inv(bc[1:]),'r':(x-bc[0])@np.linalg.inv(bc[1:])}; L={}; pairs=[]
for k,z in list(F.items()):
 F['c'+k]=c1(z)@b1; F['d'+k]=c2(z)@b2; pairs.append((k,'c'+k,'d'+k))
for k,c,d in pairs:
 L[k]=lin(F[k]); L[k+c+d]=np.column_stack((L[k],F[c],F[d])); L['b'+c+d]=np.column_stack((lin(B),F[c],F[d]))
 L[c+d]=np.column_stack((lin(F[c]),F[d])); F[c+'+'+d]=F[c]+F[d]; L[c+'+'+d]=lin(F[c]+F[d])
y=s['train']['y']; t=s['train']['traj']; times=s['eval_times']; tr=np.arange(52); te=np.arange(52,len(y));
def time_basis(rates):return -np.expm1(-times[:,None]*rates)
def residual(rates):
 a=time_basis(rates); return (a@np.linalg.lstsq(a,t[tr].T,rcond=None)[0]-t[tr].T).ravel()
a=time_basis(least_squares(residual,[1.8,.4],bounds=([1,.01],[10,1])).x); amp=np.linalg.lstsq(a,t[tr].T,rcond=None)[0].T
Y=np.column_stack((y[tr],amp));
def predict(l,extra,target,pen):
 M=np.column_stack((l,extra)); A=M[tr]; g=A.T@A; g.flat[::len(g)+1]+=pen; g[0,0]-=pen; return M[te]@np.linalg.solve(g,A.T@target)
from itertools import combinations_with_replacement
from scipy.linalg import qr
print('n',len(y),'times',times,'panel B',np.round(bc,4),'panel C1',np.round(b1,4),'panel C2',np.round(b2,4),'panel A',np.round(co,4),sep='\n')
Z=np.column_stack((y,np.linalg.lstsq(a,t.T,rcond=None)[0].T))
base=L['ucudu']; print('trajectory sv',np.linalg.svd(t,compute_uv=False)); print('v2 frequency basis',np.max(abs(c2(pc['u'])@b2-pc['v2'])))
blocks={'base':base,'cub':np.column_stack([np.prod(U[:,j],axis=1) for j in combinations_with_replacement(range(4),3)]),'quad':np.column_stack([np.prod(U[:,j],axis=1) for j in combinations_with_replacement(range(4),2)])}
for k in ('u','b','x','q','r'):
 f=F[k]; blocks['tanh_'+k]=np.tanh(f); blocks['sin_'+k]=np.sin(f); blocks['sin24_'+k]=np.sin(2.4*f); blocks['rat_'+k]=f/(1+f*f); blocks['cos24_'+k]=np.cos(2.4*f)
 for j in ('cu','du'):
  blocks[k+'*'+j]=(f[:,:,None]*F[j][:,None,:]).reshape(-1,16)
blocks['c*d']=(F['cu'][:,:,None]*F['du'][:,None,:]).reshape(-1,16)
blocks['c*c']=(F['cu'][:,:,None]*F['cu'][:,None,:]).reshape(-1,16)
blocks['d*d']=(F['du'][:,:,None]*F['du'][:,None,:]).reshape(-1,16)
blocks['sym'] = np.column_stack((U**3,U**2*np.roll(U,1,axis=1),U**2*np.roll(U,-1,axis=1),np.roll(U,1,axis=1)*U*np.roll(U,-1,axis=1)))
tr=np.arange(len(y)-20);te=np.arange(len(y)-20,len(y))
def fit(M,tr=tr,te=te,pen=.00015):
 G=M[tr].T@M[tr]+pen*np.eye(M.shape[1]); G[0,0]-=pen; return M[te]@np.linalg.solve(G,M[tr].T@Z[tr])
def err(p,te=te):return np.round(np.r_[np.mean(abs(p-Z[te]),axis=0),np.mean(abs(p[:,1:]@a.T-t[te]))],5)
for k,v in blocks.items():
 if k!='base': print(k,err(fit(np.c_[base,v])))
for names in [('sym','c*d'),('cub','c*d'),('sym','c*d','tanh_b'),('sym','c*d','sin_b'),('sym','c*d','quad'),('sym','c*d','x*cu')]:
 M=np.column_stack([base,*[blocks[k] for k in names]]);print(names,err(fit(M)))
N=np.column_stack([v for k,v in blocks.items() if k!='base']); tags=[k+':'+str(i) for k,v in blocks.items() if k!='base' for i in range(v.shape[1])]
q=np.linalg.qr(base[tr])[0]; nr=N[tr]-q@(q.T@N[tr]); scale=np.linalg.norm(nr,axis=0); D=nr/np.maximum(scale,1e-9); R=Z[tr]-q@(q.T@Z[tr]);
for j in range(3):
 chosen=[]; r=R[:,j]; report=[]
 for n0 in range(10):
  sc=abs(D.T@r);sc[chosen]=-1;i=np.argmax(sc);chosen.append(i); G=D[:,chosen];r=R[:,j]-G@np.linalg.lstsq(G,R[:,j],rcond=None)[0];pred=fit(np.c_[base,N[:,chosen]])[:,j];report.append((tags[i],round(float(np.std(r)),5),round(float(np.mean(abs(pred-Z[te,j]))),5)))
 print('sparse',j,report)
