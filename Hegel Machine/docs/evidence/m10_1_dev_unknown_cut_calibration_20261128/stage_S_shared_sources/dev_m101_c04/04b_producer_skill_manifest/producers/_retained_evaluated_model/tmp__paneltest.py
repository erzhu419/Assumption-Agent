import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.linalg import subspace_angles
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,T=[s['train'][k] for k in ('x','y','traj')]; t=s['eval_times']; np.set_printoptions(precision=4,suppress=True,linewidth=115)
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def lin(x):return np.column_stack([np.ones(len(x)),x])
def quad(x):return np.column_stack([x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(4),2)])
u,v=(s['panel_a'][k] for k in ('u','v')); A=[]
for i in range(4):
 h=lin(u[:,i]); init=np.polyfit(u[:,i],v[:,i],1); good=np.argsort(abs(h@init[::-1]-v[:,i]))[:42]; w=ls(h[good],v[good,i]); err=abs(h@w-v[:,i]); good=err<max(.001,5*np.median(err)); A.append(ls(h[good],v[good,i]))
a,d=np.asarray(A).T; B=ls(lin(s['panel_b']['u']),s['panel_b']['v']); C=s['panel_c']; C2=ls(np.c_[lin(C['u']),C['u']**2],C['v2']); z=x*d+a; bases={'x':x,'A_forward':z,'A_inverse':(x-a)/d,'AB':lin(z)@B,'XB':lin(x)@B,'AiB':lin((x-a)/d)@B,'ABt':(z-B[0])@B[1:].T,'ABinv':(z-B[0])@np.linalg.inv(B[1:])}; print('A',a,d,'B',B,'C2',C2)
for name,h in bases.items():
 q=quad(h); L=lin(h); w=ls(np.c_[L,q],y); mix=ls(L,T); print(name,'q',w[5:],'y lin',w[1:5],'T mix svd',np.linalg.svd(mix[1:],full_matrices=False)[0][:,:2].T)
 for tag,qq in [('diag',h*h),('cross',q[:,[1,2,3,5,6,8]]),('full',q),('C2',(h*h)@C2[5:])]:
  f=np.c_[L,qq]; w=ls(f[:56],y[:56]); e=f@w-y; print(tag,round(np.std(e[:56]),5),round(np.mean(abs(e[56:])),5),end='; ')
 print()
