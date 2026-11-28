exec(open('/tmp/read08.py').read().split("print('C range'")[0]); from scipy.optimize import minimize_scalar
h=q[:,2]; B=poly(b,1); D=np.c_[np.ones(len(q)),C]
print('times',s['eval_times'],'telemetry',s['telemetry_vec']); print('poly gate co',ls(np.c_[base,b*h[:,None]**2][:52],y[:52]));
opts={}
for tag,(fn,bd) in {'inverse':(lambda a,h:1/(1+a*h*h),(-.18,8)), 'exp':(lambda a,h:np.exp(-a*h*h),(-.6,4)), 'cos':(lambda a,h:np.cos(a*h),(.01,3)), 'root':(lambda a,h:1/np.sqrt(1+a*h*h),(-.18,8)), 'power':(lambda a,h:abs(h)**a,(.65,5))}.items():
 for mode in ['B','all','diff','cyc','one0','one1','one2','one3']:
  def F(p):
   g=fn(p,h)[:,None];
   if mode=='B':return np.c_[D,b*g]
   if mode=='all':return np.c_[np.ones(len(q)),np.c_[b,C]*g]
   if mode=='diff':return np.c_[base,b*g]
   if mode=='cyc':return np.c_[D,b*g,c-C[:,None]/4][:,:9]
   j=int(mode[-1]);return np.c_[base,b[:,j:j+1]*g]
  es=[]; ps=[]
  for i,j in folds:
   def obj(p):
    f=F(p)[i];return np.linalg.norm(f@ls(f,y[i])-y[i])**2
   sol=minimize_scalar(obj,bounds=bd,method='bounded'); p=sol.x;f=F(p); es.append(err(f[j]@ls(f[i],y[i])-y[j]));ps.append(p)
  opts[tag,mode]=(np.mean(es[1:]),fmt(es),fmt(ps),np.sqrt(obj(ps[-1])/len(i)))
for k,v in sorted(opts.items(),key=lambda kv:kv[1][0]):print(k,v)
for a in [.1,.22,.32,.55,1.,1.25]:
 g=1/(1+a*h*h);print('fixed inv',a,'B',fmt(cv(np.c_[D,b*g[:,None]])),'all',fmt(cv(np.c_[np.ones(len(q)),np.c_[b,C]*g[:,None]])))
