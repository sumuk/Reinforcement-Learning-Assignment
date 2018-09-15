import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
f=open('/home/sumuk/Desktop/sumuk/homework-master/hw1/data.pkl','rb')
data=pickle.load(f)
f.close()
def map(x, in_min, in_max, out_min, out_max):
    return ((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)

action=data['actions']
action=action.reshape((action.shape[0],action.shape[-1]))
#action=np.interp(action,[np.amin(action),np.amax(action)],[-1,1])
observation=data['observations']
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session() as sess:
        tf_util.initialize()
	saver = tf.train.import_meta_graph('/home/sumuk/Desktop/sumuk/my_test_model-1.meta')
    	saver.restore(sess,'/home/sumuk/Desktop/sumuk/my_test_model-1')
	last_layer=tf.get_collection("last_layer")[0]
	print last_layer
        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit
        #print "max_steps=",max_steps
        returns = []
        observations = []
        actions = []
	labels=[]
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action1 =sess.run(last_layer,feed_dict={"x:0":np.array([obs]),"y:0":np.array([action[0]])})#policy_fn(obs[None,:])
		
                observations.append(obs)
		#action1=map(action1,-1,1,np.amin(action),np.amax(action)) 
		#action1=np.interp(action1,[-1,1],[np.amin(action),np.amax(action)])
		#print action1,policy_fn(obs[None,:])                
		actions.append(action1)
		labels.append(policy_fn(obs[None,:]))
		print action1,policy_fn(obs[None,:])
                obs, r, done, _ = env.step(action1[0])
                totalr += r
                steps += 1
		print "done",done,steps
                if args.render:
		    
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
		    print "here"
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions),'labels':np.array(labels)}
        
	try:
		file1=open("/home/sumuk/Desktop/sumuk/homework-master/hw1/data1.pkl","rb")
		print "here in try"
		data1=pickle.load(file1)
		file1.close()	
		expert_data['observations']=np.concatenate((expert_data['observations'],data1['observations']))
		expert_data['actions']=np.concatenate((expert_data['actions'],data1['actions']))
		expert_data['labels']=np.concatenate((expert_data['labels'],data1['labels']))
	except:
		pass
	
	file1=open("/home/sumuk/Desktop/sumuk/homework-master/hw1/data1.pkl","wb")
        pickle.dump(expert_data,file1)
        file1.close()

if __name__ == '__main__':
    main()
