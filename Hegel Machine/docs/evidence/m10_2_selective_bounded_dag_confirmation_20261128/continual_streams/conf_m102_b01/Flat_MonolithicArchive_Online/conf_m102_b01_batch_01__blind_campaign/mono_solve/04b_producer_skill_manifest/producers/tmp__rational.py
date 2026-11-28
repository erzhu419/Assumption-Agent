from refine import *

def candidates(x,y):
 out=[]
 for pwr in [1,2,3,4,(1,2),(1,3),(2,3),(2,4),(1,2,3),(1,2,3,4)]:
  pw=(pwr,) if isinstance(pwr,int) else pwr
  for num in [1,3,5]:
   for lin in [0,1,2]:
    if len(pw)==1 and len(pw)+lin+num//2>4:continue
    def D(z,p,pw=pw,num=num,lin=lin):
     n=z**num
     if lin==2:n=n+p[-1]*z
     d=1+sum(a*abs(z)**b for a,b in zip(p,pw));w=n/d
     if lin:w=w+p[len(pw)]*z
     return aff(w)
    def err(p):
     d=D(x,p);return (d@ls(d,y)-y).ravel()
    opts=[]
    for a in [.15,.6,2,5]:
     o=least_squares(err,np.r_[np.repeat(a,len(pw)),np.zeros(lin)],bounds=(np.r_[np.zeros(len(pw)),np.full(lin,-100.)],np.r_[np.full(len(pw),20.),np.full(lin,100.)]),max_nfev=120);opts.append(o)
    o=min(opts,key=lambda o:np.linalg.norm(o.fun));d=D(x,o.x);m=ls(d,y);J=np.c_[np.kron(d,np.eye(4)),o.jac];h=np.sum(J*np.linalg.pinv(J).T,axis=1);out.append((np.std(o.fun/np.maximum(.05,1-h)),np.std(o.fun),pw,num,lin,o.x,lambda z,D=D,p=o.x,m=m:D(z,p)@m))
 return sorted(out,key=lambda v:v[0])
if __name__=='__main__':
 tr=s['train'];X=calibrate(s['panel_a'],tr['x']);Y=np.c_[tr['y'],tr['traj']];ops=[]
 for k,j in [('panel_b','v'),('panel_c','v1')]:
  rr=candidates(s[k]['u'],s[k][j]);print(k,*[v[:-1] for v in rr[:12]],sep='\n',flush=True);ops.append(rr)
 for b,c in itertools.product(ops[0][:3],ops[1][:3]):
  q=c[-1](b[-1](X));A=aff(q); q2=poly(q,2);line=A@ls(A[:52],Y[:52]);quad=q2@ls(q2[:52],Y[:52]);print('endpoint',b[2:5],c[2:5], 'line',np.mean(abs(line[52:]-Y[52:]),axis=0),'quad',np.mean(abs(quad[52:]-Y[52:]),axis=0));print('quad coeff y',ls(q2[:52],Y[:52,0]))
