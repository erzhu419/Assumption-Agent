exec(open('/tmp/expand08.py').read().split('R=[]')[0]);
def emit(k,f,p0=[],bd=(-np.inf,np.inf),reg=0):
 v=run(f,p0,bd,reg); print(k,fmt(v[:4]),v[4],v[5]);
for mode in ['base','full']:
 a=base if mode=='base' else full
 for style in ['b0','b01','b012','all']:
  js={'b0':[0],'b01':[0,1],'b012':[0,1,2],'all':[0,1,2,3]}[style]
  for cubic in [False,True]:
   for single in [False,True]:
    def f(p):
     G=1/(1+h[:,None]**2*np.broadcast_to(p,(len(js),))); extra=b[:,js]*G
     d=[a,extra*h[:,None]**2,b[:,3]*h]
     if cubic:d.append(b[:,3]*h**3)
     return np.column_stack(d)
    emit((mode,style,cubic,single),f,[.22]*(1 if single else len(js)),(0,1.5),.001)
for form in ['shared','indep','shared_odd','indep_odd','shared_partial','shared_polyodd']:
 def f(p):
  g=h[:,None]; a=np.broadcast_to(p[:1] if form.startswith('shared') else p[:4],(4,));
  F=[N,C,b/(1+a*g*g)]
  if form.endswith('odd'): F.append(b[:,3]*(h/(1+p[-1]*h*h)))
  if form.endswith('polyodd'):F.append(b[:,3]*h**3)
  if form.endswith('partial'): F.append(b[:,3]*h)
  return np.column_stack(F)
 emit(form,f,[.22]*((1 if form.startswith('shared') else 4)+int(form.endswith('_odd'))),(0,2),.001)
for tag,v in [('plain',base),('each',full)]:
 for p2 in [[],[0,1,2],[0,1,2,3]]:
  for p3 in [[],[3],[0,1,2,3]]:
   emit((tag,p2,p3),lambda _: np.c_[v,b[:,p2]*h[:,None]**2,b[:,3]*h,b[:,p3]*h[:,None]**3]);
