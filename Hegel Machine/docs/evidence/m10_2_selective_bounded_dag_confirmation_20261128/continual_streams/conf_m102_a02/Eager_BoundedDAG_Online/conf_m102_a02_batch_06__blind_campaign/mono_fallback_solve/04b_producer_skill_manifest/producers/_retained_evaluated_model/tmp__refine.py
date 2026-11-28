exec(open('/tmp/graph.py').read().split('D={}')[0]); z=F['inv']; p=s['panel_b'];u=p['u'];v=p['v']
func.update({'sinh':np.sinh,'erf':__import__('scipy').special.erf,'sqrt':lambda a:a/np.sqrt(1+a*a),'cubic':lambda a:a**3})
sol=[]
for name in ['tanh','atan','sin','rat','erf','sqrt','cubic']:
 f=func[name]
 for base in (0,1):
  for shift in (0,1,2):
   def mat(q,w):
    a=np.exp(q[:4]); b=0 if shift==0 else q[4:]; h=f(a*w+b);return np.c_[aff(w) if base else np.ones(len(w)),h]
   def residual(q):
    X=mat(q,u);return (X@fit(X,v)-v).ravel()
   q=np.r_[np.full(4,np.log({'tanh':1.45,'atan':1.35,'sin':2.8,'rat':.9,'erf':1.,'sqrt':1.,'cubic':1}[name])),np.zeros({0:0,1:1,2:4}[shift])]; lo=np.r_[np.full(4,-3),np.full(len(q)-4,-2)];hi=np.r_[np.full(4,1.55),np.full(len(q)-4,2)]; o=least_squares(residual,q,bounds=(lo,hi),max_nfev=120); P=mat(o.x,u); B=fit(P,v); g=mat(o.x,z)@B; E=np.c_[aff(np.c_[z,g]),(z[:,:,None]*g[:,None,:]).reshape(len(z),-1)]; C=fit(E[:52],np.c_[y,T][:52]); err=abs(E@C-np.c_[y,T]); line=(np.sqrt(np.mean(o.fun**2)),name,base,shift,np.round(o.x,5),np.round([err[:52,0].mean(),err[52:,0].mean(),err[:52,1:].mean(),err[52:,1:].mean()],5)); print(line);sol.append((line[0],line,g,B,E,C))
for _,line,g,B,E,C in sorted(sol,key=lambda a:a[0])[:5]:
 print('best',line,'B',B,'coef for y & state endpoint',C[:,[0,-1]].T,'cross sv',np.linalg.svd(C[9:,0].reshape(4,4),compute_uv=False),np.linalg.svd(C[9:,-1].reshape(4,4),compute_uv=False));print('largest y errs',np.sort(abs(E@C[:,0]-y))[-8:]);
