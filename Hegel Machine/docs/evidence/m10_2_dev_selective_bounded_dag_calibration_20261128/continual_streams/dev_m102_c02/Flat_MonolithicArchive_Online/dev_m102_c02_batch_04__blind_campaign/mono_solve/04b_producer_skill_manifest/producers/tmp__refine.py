from operators import *
from itertools import combinations
from scipy.linalg import qr

def fitop(u,v,kind):
 def transform(x,p):
  a,b=p
  return {'tanh':lambda:x+a*np.tanh(b*x),'atan':lambda:x+a*np.arctan(b*x),'rational':lambda:x+a*x/(1+b*x*x),'sin':lambda:x+a*np.sin(b*x),'asinh':lambda:x+a*np.arcsinh(b*x),'soft':lambda:x+a*x/np.sqrt(1+b*x*x),'sat':lambda:x+a*x/(1+b*abs(x)),'quint':lambda:x+a*x**3+b*x**5}[kind]()
 def res(p):
  H=aff(transform(u,p));return (H@L(H,v,rcond=None)[0]-v).ravel()
 sols=[least_squares(res,[a,b],bounds=([-12,.001],[12,12]),max_nfev=150) for a,b in product([-.95,-.65,-.3,.3],[.6,1.4,2.5])];p=min(sols,key=lambda q:q.cost).x;H=aff(transform(u,p));W=L(H,v,rcond=None)[0];return np.sqrt(np.mean(res(p)**2)),p,lambda x:aff(transform(x,p))@W,lambda x:transform(x,p)

def checks(z):
 V=np.c_[s['train']['y'],s['train']['traj']];t=s['eval_times'];print('times',t,'y range',np.min(V[:,0]),np.max(V[:,0]));
 def modes(p):return -np.expm1(-np.exp(p)[:,None]*t)
 def r(p):
  H=modes(p);return (V[:52,1:]-(V[:52,1:]@np.linalg.pinv(H))@H).ravel()
 p=least_squares(r,np.log([.22,1.25])).x;print('rates',np.exp(p),'rms',np.sqrt(np.mean(r(p)**2)),'telemetry rms',np.sqrt(np.mean(r(np.log([.22,1.25]))**2)))
 bank={}
 for label,pk,vk in [('B','panel_b','v'),('C','panel_c','v1'),('D','panel_c','v2')]:
  u,v=s[pk]['u'],s[pk][vk]; rows=[]
  for kind in ['tanh','atan','rational','sin','asinh','soft','sat','quint']:
   e,p,f,g=fitop(u,v,kind);rows.append((e,kind,p,f,g))
  for e,kind,p,_,_ in sorted(rows)[:5]:print(label,kind,e,p)
  for e,kind,p,f,g in sorted(rows)[:2]: bank[label+kind]=f(z);bank[label+kind+'pre']=g(z)
  for kind in ['blend','sq','atan']: bank[label+kind+'old']=op(u,v,kind)[3](z)
 bank['Z']=z;bank['tanhZ']=np.tanh(z)
 def score(H):
  W=L(H[:52],V[:52],rcond=None)[0];e=np.mean(abs(H[52:]@W-V[52:]),0);ein=np.sqrt(np.mean((H[:52]@W-V[:52])**2,0));return np.r_[e[0],e[1:].mean(),e[-1],ein[0],ein[1:].mean()]
 results=[]
 for name,a in bank.items():
  opp=(a[:,:2]*a[:,2:]).sum(1);cyc=(a*np.roll(a,-1,axis=1)).sum(1)
  kinds={'aff':aff(a),'opp':np.c_[aff(a),opp],'cyc':np.c_[aff(a),cyc],'quad':mon(a,2),'cubdiag':np.c_[aff(a),a**3], 'acub':np.c_[a,a**3], 'a':a, 'affZ':np.c_[aff(a),z],'tanh':aff(np.tanh(a))}
  for n,H in kinds.items():results.append((name,n,score(H)))
 for metric in [0,1]:
  for nam,n,e in sorted(results,key=lambda v:v[2][metric])[:15]:print(nam,n,e)
if __name__=='__main__':checks(calibrate()(s['train']['x']))
