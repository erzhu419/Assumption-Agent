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
print('rates',least_squares(residual,[1.8,.4],bounds=([1,.01],[10,1])).x,'time fit',np.mean(abs(amp@a.T-t[tr]))); print('keys',[(k,list(v) if isinstance(v,dict) else np.shape(v)) for k,v in s.items()]);
Z=np.column_stack((y, np.linalg.lstsq(a,t.T,rcond=None)[0].T)); base=L['ucudu'];
def error(M):
 g=M[tr].T@M[tr]+.00015*np.eye(M.shape[1]); g[0,0]-=.00015; p=M[te]@np.linalg.solve(g,M[tr].T@Z[tr]); return np.mean(abs(p[:,0]-Z[te,0])),np.mean(abs(p[:,1:]@a.T-t[te]))
print('base',error(base));
for k in ('u','b','x','q','r'):
 f=F[k]; A=np.column_stack((base,f*f)); print('square',k,error(A)); A=np.column_stack((base,*[f[:,i]*f[:,j] for i in range(4) for j in range(i,4)])); print('quadratic',k,error(A));
# Look for missing dot symmetries and compact repeated factors by measuring independent residual reductions.
from itertools import permutations,combinations_with_replacement
from scipy.linalg import qr
terms={};
for i,j in combinations_with_replacement(list(F),2):
 if '+' in i or '+' in j: continue
 for p in permutations(range(4)):
  if i==j and p>tuple(np.argsort(p)):continue
  terms[(i,j,p)]=(F[i]*F[j][:,p]).sum(1)
N=np.column_stack(list(terms.values())); A=base[tr]; Q=np.linalg.qr(A)[0]; R=Z[tr]-Q@(Q.T@Z[tr]); D=N[tr]-Q@(Q.T@N[tr]); score=(D.T@R)**2/(np.sum(D*D,axis=0)[:,None]+1e-16)
for j in range(3):
 print('target',j,'resid',np.std(R[:,j]));
 for ix in np.argsort(score[:,j])[-8:][::-1]: print(list(terms)[ix],round(float(score[ix,j]/np.sum(R[:,j]**2)),4),np.round(error(np.column_stack((base,N[:,ix]))),5))
for k in ('u','b','q','r','x'):
 u0=F[k]; pairs=[u0[:,i]*u0[:,j] for i in range(4) for j in range(i,4)]; cos=[np.cos(2.4*u0[:,i]) for i in range(4)]; lib=np.column_stack([base,u0,*pairs,*cos]); print('extended',k,error(lib));
