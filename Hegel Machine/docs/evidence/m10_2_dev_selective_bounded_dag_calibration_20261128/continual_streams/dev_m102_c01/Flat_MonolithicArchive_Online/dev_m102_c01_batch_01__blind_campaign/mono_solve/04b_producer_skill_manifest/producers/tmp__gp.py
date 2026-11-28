exec(open('/tmp/probe.py').read().split('rows=[]')[0])
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
rng=np.random.default_rng(91); splits=[np.arange(72)]+[rng.permutation(72) for _ in range(8)]
for key in ['x','A-','A-B','A-Bi','A-v1','A-v2','A-Bv1','A-Bv2']:
 z=Q[key]; f=p(z,3); W=np.linalg.lstsq(f,Y,rcond=None)[0]; print(key,'poly2',np.round(np.linalg.lstsq(f[:,:15],Y,rcond=None)[0].T,3)); print('cubic',np.round(W[15:].T,3))
I,J=np.triu_indices(4)
def covparts(z):
 dot=z@z.T; t2=z[:,I]*z[:,J]; return [dot,t2@t2.T]
D=(x[:,None,:]-x[None,:,:])**2
K0=np.ones((72,72)); K1,K2=covparts(Q['A-']); Kb1,Kb2=covparts(Q['A-B']); Kc1,Kc2=covparts(Q['A-Bv1']); Kd1,Kd2=covparts(Q['A+Bv2']);
for setid,parts in enumerate([[K0,K1,K2],[K0,K1,K2,Kb2,Kc1,Kc2,Kd1,Kd2],[K0,K1,K2,K1**3,K1**4,Kb2,Kc1,Kc2,Kd1,Kd2]]):
 parts=np.array(parts); lp=len(parts)
 for criterion in ['nll','loo']:
  errors=[]
  for ix in splits:
   a,b=ix[:52],ix[52:]; ee=[]
   Pa=parts[:,a][:,:,a]; Da=D[a][:,a]; Pb=parts[:,b][:,:,a]; Db=D[b][:,a]; eye=np.eye(len(a))
   for yy in [y,tt]:
    v=yy[a]
    def make(P,D,h):
     h=np.exp(h); g=np.exp(-np.einsum('ijk,k->ij',D,h[lp:lp+4])); return np.einsum('k,kij->ij',h[:lp],P)+h[-2]*g,g
    def fun(h):
     K,g=make(Pa,Da,h); h0=np.exp(h); K+=h0[-1]*eye
     C=cho_factor(K); al=cho_solve(C,v); inv=cho_solve(C,eye)
     dK=np.concatenate([Pa*h0[:lp,None,None],-g[None,:,:]*Da.transpose(2,0,1)*h0[lp:lp+4,None,None]*h0[-2],(g*h0[-2])[None],(eye*h0[-1])[None]])
     if criterion=='nll':
      loss=v@al/2+np.log(np.diag(C[0])).sum(); G=(inv-np.outer(al,al))/2
     else:
      d=inv.diagonal(); r=al/d; loss=np.mean(r*r); A=inv@(2*r/d/len(a)); G=(inv*(2*r*r/d/len(a)))@inv-np.outer(A,al)
     return loss,np.einsum('ij,kij->k',G,dK)
    init=np.r_[np.repeat(-3.,lp),np.repeat(-1.,4),-1.,-9.]; res=minimize(fun,init,jac=True,method='L-BFGS-B',bounds=[(-16,5)]*lp+[(-6,2)]*4+[(-12,5),(-14,-3)],options={'maxiter':220}); h=res.x
    K,_=make(Pa,Da,h); Kb,_=make(Pb,Db,h); pred=Kb@cho_solve(cho_factor(K+np.exp(h[-1])*eye),v); ee.append(np.mean(abs(yy[b]-pred)))
    if len(errors)==0: print(setid,criterion,'params',np.round(h,2),'MAE',ee[-1]);
   errors.append(ee)
  print(setid,criterion,'metrics',errors[0],np.mean(errors,0),np.max(errors,0))
