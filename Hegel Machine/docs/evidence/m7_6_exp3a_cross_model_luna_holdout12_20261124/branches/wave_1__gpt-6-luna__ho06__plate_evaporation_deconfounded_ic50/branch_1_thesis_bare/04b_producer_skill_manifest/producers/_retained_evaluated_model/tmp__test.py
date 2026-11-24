import pickle,numpy as np
from scipy.optimize import least_squares
x=pickle.load(open('/holdout_task/public_screen.pkl','rb'))
print('loaded',len(x['plates']))
