from panels import *
from scipy.linalg import solve
from sklearn.model_selection import KFold
from itertools import combinations
S=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72)),*KFold(4,shuffle=True,random_state=13).split(X)]

def library(x):
 def bx(a):return aug(a)@B
 def inv(a):return (a-B[0])@np.linalg.inv(B[1:])
 def p(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
 def q(a,d):return PolynomialFeatures(d,include_bias=False).fit_transform(a)
 for name,d in routes(x):
  if name not in ['X','Ai','XB','AiB','AiBi']:continue
  z,w=C1(d),C2(d)
  for label,a in [('x',d),('z',z),('w',w),('zw',z+w),('z-w',z-w),('x+z',d+z),('x+w',d+w),('b(z)+w',bx(z)+w),('z+b(w)',z+bx(w)),('b(z)+b(w)',bx(z+w)),('b(z)-w',bx(z)-w),('z-b(w)',z-bx(w)),('x+bi(z)',d+inv(z)),('x+bi(w)',d+inv(w)),('z(w)',C1(w)),('w(z)',C2(z)),('z(z)',C1(z)),('w(w)',C2(w))]:
   for tag,h in [('id',a),('sin',np.sin(1.7*a)),('tanh',np.tanh(1.7*a)),('rat',a/(1+.2*a*a)),('soft',a/(1+.2*abs(a)))]:
    F=q(h,2);yield name,label,tag,'q2',F
    if tag=='id':
     yield name,label,tag,'q3',q(h,3)
     yield name,label,tag,'q2-cross',np.c_[F,q(d,2),p(d,h)]
  for k in [-2,-1,-.5,.5,1,2]:
   for l in [-2,-1,-.5,0,.5,1,2]:
    yield name,str((k,l)),'id','mix',q(d+k*inv(z)+l*inv(w),2)
  yield name,'branches','id','products',np.c_[q(d,2),p(z,w),p(d,z),p(d,w),q(z,2),q(w,2)]

if __name__=='__main__':
 R=[]; O=np.c_[Y,T]; predictions=[]
 for n,k,t,form,F in library(X):
  errors={a:[] for a in [.000001,.002,.02,.15]}; yp={a:[] for a in errors}
  for i,j in S:
   G=(F-F[i].mean(0))/np.maximum(F[i].std(0),1e-7); a=G[i]; b=G[j]; U,z,V=np.linalg.svd(a,full_matrices=False); h=(U.T@(O[i]-O[i].mean(0)))*z[:,None]; c=b@V.T
   for reg in errors:
    pred=c@(h/(z*z+reg)[:,None])+O[i].mean(0);e=abs(pred-O[j]);errors[reg].append([e[:,0].mean(),e[:,1:].mean()]);yp[reg].append(pred)
  for reg,e in errors.items():R.append((n,k,t,form,reg,np.array(e)));predictions.append(yp[reg])
 for col in [0,1]:
  for label,metric in [('worst',lambda a:a.max()),('designated',lambda a:a[0])]:
   print('rank',col,label)
   for h in sorted(R,key=lambda r:metric(r[-1][:,col]))[:8]: print(h[:-1],h[-1][:,col].round(5))
 np.save('/tmp/comp_predictions.npy',np.array(predictions,dtype=object),allow_pickle=True);import pickle;pickle.dump(R,open('/tmp/comp_rank.pkl','wb'))
