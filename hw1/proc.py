import pickle
import numpy as np

f=open('/home/sumuk/Desktop/sumuk/homework-master/hw1/data.pkl','rb')
data=pickle.load(f)
f.close()

action=data['actions']
action=action.reshape((action.shape[0],action.shape[-1]))
act=action
a=[]
obs=data['observations']
o=[]
for i in range(obs.shape[0]/1000):
	ob=obs[i*1000:i*1000+1000,:]
	ac=act[i*1000:i*1000+1000,:]
	for j in range(1000-1):
		o.append(ob[j:j+2,:])
		a.append(ac[j+1,:])
o=np.array(o)                
a=np.array(a)

