import os
import glob
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
import gym
from ppoenv2 import SpectrumEnv
import matplotlib.pyplot as plt
import sys

# set device to cpu or cuda
device = torch.device('cpu')
'''
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') ep
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
'''
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.returns = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.returns[:]
'''
class Squeeze(nn.Module) :
    def __init__(self) :
        super(Squeeze, self).__init__()
        
        self.fc = nn.Sequential(
                        nn.Linear(2048, 512),
                        nn.ReLU(),
                        nn.Linear(512, 128),
                        nn.ReLU(),
                        nn.Linear(128,1)
                    )
    def forward(self, x) :
        x = torch.transpose(x,0,1)
        #print(x.shape)
        x = self.fc(x)
        x = torch.transpose(x,0,1)
        #print(x.shape)
        return x
'''   
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        
        self.actor = nn.Sequential(
                        nn.Linear(2048, 512),
                        nn.Tanh(),
                        nn.Linear(512, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_dim),
                        nn.Softmax(dim=-1) #행에 대해 softmax
                    )
        
        self.critic = nn.Sequential(
                        nn.Linear(2048, 512),
                        nn.Tanh(),
                        nn.Linear(512, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )

    def forward(self):
        raise NotImplementedError
        
    
    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lamda = 0.95 # gae에서 미래에 대한 V에 대한 할인율
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        


    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            #print(state)
            action, action_logprob, state_val = self.policy_old.act(state)
            #print(action)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        #print(action)
        #print(state_val)
        #n = int(input())

        return action.item()
    
    def save_advantages(self, max_ep_len : int, env_n : int) :
        # Monte Carlo estimate of returns(gae)
        #print(self.buffer.rewards)
        rewards = []
        gae = 0 #gae + state_value를 rewards에 저장
        start_point = max_ep_len * env_n # start_point 변수에 시작점을 정해줌.
        
        next_state_value = 0  # 다음 상태의 value를 저장한 변수
        for reward, is_terminal, state_value in zip(reversed(self.buffer.rewards[start_point:start_point+max_ep_len]),reversed(self.buffer.is_terminals[start_point:start_point+max_ep_len]),reversed(self.buffer.state_values[start_point:start_point+max_ep_len])):
            if is_terminal:
                delta = reward - state_value
                next_state_value = 0  # 종료 상태의 가치는 0으로 설정 (즉 delta = reward - state_value)
                gae = delta
            else:
                delta = reward + self.gamma * next_state_value - state_value
                next_state_value = state_value  # 다음 상태의 가치 저장
            gae = delta + self.gamma * self.lamda * gae  # 
            rewards.insert(0, gae + state_value)

        for rew in rewards :
            self.buffer.returns.append(rew)

        #rewards의 값들을 returns에 append해줌.

    def update(self, batch_size):
        # Normalizing the rewards
        rewards = torch.tensor(self.buffer.returns, dtype=torch.float32).to(device)
        #print(rewards.shape)
        # convert list to tensor
        
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0),1).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        
        #print(old_states.shape)
        #print(old_actions.shape)
        #print(old_logprobs.shape)
        #print(old_state_values.shape)
        
        # shuffle indices
        indices = torch.randperm(old_states.shape[0])
        #print(indices)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs) :
            for i in range(batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_old_states = old_states[batch_indices]
                batch_old_actions = old_actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_old_state_values = old_state_values[batch_indices]
                batch_advantages = advantages[batch_indices]

                #print(batch_old_states.shape)
                #print(batch_old_actions.shape)
                #print(batch_old_logprobs.shape)
                #print(batch_advantages.shape)
                logprobs, state_values, dist_entropy = self.policy.evaluate(batch_old_states, batch_old_actions)
                state_values = torch.squeeze(state_values)
                ratios = torch.exp(logprobs - batch_old_logprobs.detach())
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * batch_advantages
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards[batch_indices]) - 0.01 * dist_entropy

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path): 
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

        
print("============================================================================================")

if __name__ == "__main__":
    ################################### Training ###################################
    env_name = "spectrum_env"

    env_nums = 8
    update_num = env_nums // 2
    env = list()

    for i in range(env_nums) :
        env.append(SpectrumEnv(10150+i))

    has_continuous_action_space = False
    max_ep_len = 2000
    action_std = None
    max_training_timesteps = 5000000

    print_freq = max_ep_len * 4    # print avg reward in the interval (in num timesteps)
    #log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)
    save_model_freq = 100000      # save model frequency (in num timesteps)

    update_timestep = max_ep_len * env_nums / 2
    K_epochs = 10
    eps_clip = 0.2  
    gamma = 0.99 

    lr_actor = 0.0003  
    lr_critic = 0.001  

    random_seed = 0 

    ############################### env #####################################




    state_dim = env[0].observation_space.shape[0]

    if has_continuous_action_space:
        action_dim = env[0].action_space.shape[0]
    else:
        action_dim = env[0].action_space.n
    '''
    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten

    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)


    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)


    #### create new log file for each run 
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    #####################################################
    '''

    ################### checkpointing ###################

    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)

    #####################################################


    ############# print all hyperparameters #############

    print("--------------------------------------------------------------------------------------------")

    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("update num : ", update_num)

    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

    print("--------------------------------------------------------------------------------------------")

    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)

    print("--------------------------------------------------------------------------------------------")
    '''
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    '''

    print("PPO update frequency : " + str(update_timestep) + " timesteps") 
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)

    print("--------------------------------------------------------------------------------------------")

    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    '''
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    '''
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)


    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    '''
    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')


    # printing and logging variable

    log_running_reward = 0
    log_running_episodes = 0
    '''
    print_running_reward = 0
    print_running_episodes = 0

    time_step = 0
    i_episode = 0


    sensing_num = 2048
    for i in range(env_nums) :
        env[i].connect()
        
    #model = Squeeze().to(device)
    result_log = list()
    result_time = list()

    # training loop
    while time_step <= max_training_timesteps:

        for i in range(env_nums) :
            time.sleep(0.5)
            current_ep_reward = 0
            env[i].reset(1200000)
            input_state = []

            sensings = 0
            actions = 0

            for _ in range(sensing_num) : #처음에 sensing번 한걸 담음.
                state, reward, done, _ = env[i].step(2)
                input_state.append(state[0])

            for t in range(1, max_ep_len+1): #2000

                #print(input_state)
                temp_state = torch.FloatTensor(input_state).to(device)
                temp_state = torch.transpose(temp_state,0,1)
                #print(temp_state)
                #real_state = model.forward(temp_state).to(device)
                #print(real_state.shape)
                #print(real_state)

                action = ppo_agent.select_action(temp_state)
                state, reward, done, _score = env[i].step(action)
                #print(action)
                #print(len(state))

                if action == 2 :
                    sensings+=1
                else :
                    actions+=1

                if t == max_ep_len :
                    print(_score)
                    done = True


                for j in range(len(state)) :
                    input_state.pop(0)
                    input_state.append(state[j])
                    
                #if reward < 0 :
                #    reward = (reward / 4) + reward * (3/4) * (time_step / max_training_timesteps)
                #else :
                #    reward = (reward * 3) - reward * 2 * (time_step / max_training_timesteps)
                
                # saving reward and is_terminals
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                current_ep_reward += reward
                time_step += 1
            current_ep_reward = round(current_ep_reward,2)
            print(f"current_ep_reward : {current_ep_reward}, sensings = {sensings}, actions = {actions}")
            result_log.append(_score['total_score'])
            result_time.append(time_step)

            ii = i % update_num
            #print("update num : ",ii)
            print("")
            ppo_agent.save_advantages(max_ep_len, ii)  # reward를 2000만 끊어서 advantage를 계산

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            done = False
            while not done :
                state, reward, done, _ = env[i].step(1)

            i_episode += 1

            # update PPO agent
            if time_step % update_timestep == 0:
                #print("---update---")
                ppo_agent.update(56) #batch size가 64

            '''
            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0
            '''
            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")


                #plt.pause(0.1)

    plt.plot(result_time, result_log, label="PPO")
    plt.legend()
    plt.xlabel('time step')
    plt.ylabel('Mean Episode Rewards')
    plt.show()




    #    log_running_reward += current_ep_reward
    #    log_running_episodes += 1




    #log_f.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

    print("============================================================================================")
