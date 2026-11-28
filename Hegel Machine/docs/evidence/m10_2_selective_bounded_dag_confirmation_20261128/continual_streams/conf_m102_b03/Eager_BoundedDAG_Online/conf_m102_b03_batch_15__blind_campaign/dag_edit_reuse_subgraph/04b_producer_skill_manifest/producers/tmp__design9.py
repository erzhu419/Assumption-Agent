import numpy as np
fun={'soft':lambda z:z/(1+abs(z)),'tanh':np.tanh,'rat':lambda z:z/(1+z*z),'sin':np.sin,'rat_even':lambda z:1/(1+z*z),'abs':np.abs,'square':np.square,'signed_sq':lambda z:z*abs(z),'cube':lambda z:z**3}
def design(x,p,ori,kind,mode):
 a,b=p; y=np.roll(x,-ori,axis=1); f=fun[kind]; z=f(a*y+b)
 if mode==0:return np.c_[np.ones(len(x)),x,z,x*y]
 if mode==1:return np.c_[np.ones(len(x)),x,z,x*z]
 if mode==2:return np.c_[np.ones(len(x)),x,z,x*y,x*z]
 if mode==3:return np.c_[np.ones(len(x)),x,z,x*y/(1+a*y*y)]
 if mode==4:return np.c_[np.ones(len(x)),x,x*y+b*z]
 if mode==5:return np.c_[np.ones(len(x)),x,x*z+b*f(a*y)]
 if mode==6:return np.c_[np.ones(len(x)),x,z,x*y/(1+a*(x-y)**2)]
 if mode==7:return np.c_[np.ones(len(x)),x,z,x*y+b*x*z]
