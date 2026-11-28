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
results=[]
for lk,l in L.items():
 for mk,m in F.items():
  for nk,n in F.items():
   z=predict(l,(m*n).sum(1),Y[:,:2],.00015); tail=predict(l,m[:,0]*n[:,1],Y[:,2],.0003)
   ye=np.mean(abs(z[:,0]-y[te])); he=np.mean(abs(np.column_stack((z[:,1],tail))@a.T-t[te])); results.append((max(ye,he),ye,he,lk,mk,nk))
print('best prototype slices'); print(*sorted(results)[:15],sep='\n'); print('best y',min(results,key=lambda z:z[1])); print('best traj',min(results,key=lambda z:z[2]))
