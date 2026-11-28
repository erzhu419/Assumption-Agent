import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from itertools import product

def _chart(z, mode, h):
    if mode==1:return np.arcsinh(h*z)/h
    if mode==2:return z/(1+h*abs(z))
    if mode==3:return np.sign(z)*np.log1p(h*abs(z))/h
    if mode==4:return z/(1+h*z*z)
    return z

def _gram(z, spec):
    mode,h,kind,a,degree,b,c=spec
    q=_chart(z,mode,h)
    L=z@z.T; M=q@q.T; D=cdist(q,q,'sqeuclidean'); D0=cdist(z,z,'sqeuclidean')
    U=(1+.38*L)**degree
    if kind==0: K=np.exp(-a*D)
    elif kind==1: K=(1+a*D)**(-1)
    elif kind==2:
        R=np.sqrt(5*a*D); K=(1+R+R**2/3)*np.exp(-R)
    elif kind==3: K=np.exp(-a*D)*(1+b*L); b=0
    elif kind==4:
        A=np.exp(-a*(q[:,None,:]-q[None,:,:])**2); S=A.sum(2); K=(S*S-(A*A).sum(2))/12*(1+b*L); b=0
    elif kind==5: K=np.exp(-a*D0)+b*(1+M)*(1+L); b=0
    elif kind==6: K=np.exp(-a*D0)+b*((1+M)**2+.2*M*L); b=0
    elif kind==7: K=np.exp(-a*D0)+b*np.exp(-a*D); b=0
    elif kind==8:
        V=1/(1+c*np.sum(q*q,axis=1)); K=np.exp(-a*D0)+b*(1+.55*M)**3*V[:,None]*V[None,:]; b=0
    elif kind==9:
        V=np.array([v for v in product([-1,0,1],repeat=z.shape[1]) if any(v) and next(t for t in v if t)!=-1]); V=V/np.linalg.norm(V,axis=1)[:,None]; P=q@V.T
        S=np.sin(a*P); C=np.cos(a*P); K=np.exp(-c*D0)+b*(S@S.T+C@C.T)*(1+.4*L)/len(V); b=0
    elif kind==10:
        G=np.sqrt(np.maximum(D0,0)); K=np.exp(-a*D0)+b*np.exp(-c*G)*(1+.25*L); b=0
    else:
        P=np.tanh(h*z); J=P@P.T; K=np.exp(-a*D0)+b*(1+J)*(1+L); b=0
    return U+K+b*M

def _prediction(C,i,j,y,r):
    return C[np.ix_(j,i)]@cho_solve(cho_factor(C[np.ix_(i,i)]+r*np.eye(len(i))),y)
