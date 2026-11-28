import pickle,runpy,numpy as np
from itertools import combinations_with_replacement,product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; g=runpy.run_path('/holdout_task/node_00.py'); L=g['_ls']; aff=g['_affine']; a,A=g['_cal'](s['panel_a']); b,B=g['_cal'](s['panel_b']); Ai=np.linalg.inv(A); Bi=np.linalg.inv(B); x=s['train']['x']; k,wp,wr=g['_ops'](s['panel_c']); T=np.c_[s['train']['y'],s['train']['traj']]; Q=g['_quad']; n=52
print('calibration',a,A,b,B, 'telemetry',s['telemetry_vec']); print('poly and trig')
def poly(z,d):return np.column_stack([np.prod(z[:,ii],axis=1) for j in range(1,d+1) for ii in combinations_with_replacement(range(z.shape[1]),j)])
def check(F):
 D=aff(F); e=np.abs(D[n:]@L(D[:n],T[:n])-T[n:]); et=abs(D[:n]@L(D[:n],T[:n])-T[:n]); return np.r_[e[:,0].mean(),e[:,1:].mean(),et[:,0].mean(),et[:,1:].mean()]
for d in range(1,5):print(d,check(poly(x,d)))
zz={'i':x,'a':(x-a)@Ai,'af':x@A+a,'b':(x-b)@Bi,'bf':x@B+b,'ab':((x-a)@Ai-b)@Bi,'abf':(x-a)@Ai@B+b,'ba':((x-b)@Bi-a)@Ai,'cb':(x-a)@Bi@Ai,'bc':x@Bi@Ai-a}
R=[]
for zn,z in zz.items():
 h=poly(z,2); p=aff(Q(z))@wp; r=aff(np.cos(k*z))@wr
 for j in [.22,.5,.8,1,1.25,1.5,2,2.4,3,4]:
  c=np.cos(j*z); sn=np.sin(j*z)
  for name,F in [('polycos',np.c_[h,c]),('polysin',np.c_[h,sn]),('polytrig',np.c_[h,c,sn]),('product',np.c_[h,c,(p[:,:,None]*c[:,None,:]).reshape(len(x),-1)]),('p2r',np.c_[h,c,p*r,p*p,r*r]),('ratio',np.c_[h,c,p/(1+r*r),r/(1+p*p)]),('nested',np.c_[h,c,np.cos(j*p),np.cos(j*r)]),('cosBmix',np.c_[h,c,np.cos(j*((1-.22)*z+.22*zz['b'])),np.sin(j*((1-.22)*z+.22*zz['b']))])]:
   R.append((*check(F),zn,j,name))
for j in [0,1,2,3]: print('rank',j,*sorted(R,key=lambda r:r[j])[:5],sep='\n')
