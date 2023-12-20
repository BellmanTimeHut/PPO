import gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1")

# These coefficients are experimentally determined in practice.
gamma = 0.99     # discount
kl_coeff = 0.20  # weight coefficient for KL-divergence loss
vf_coeff = 0.50  # weight coefficient for value loss

class ActorNet(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()

        self.hidden = nn.Linear(4, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        logits = self.output(outs)
        return logits

class ValueNet(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()

        self.hidden = nn.Linear(4, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        value = self.output(outs)
        return value

actor_func = ActorNet().to(device)
value_func = ValueNet().to(device)


# Pick up action and following properties for state (s)
# Return :
#     action (int)       action
#     logits (list[int]) logits defining categorical distribution
#     log_pi (float)     log probability w.r.t policy
def pick_sample_and_logp(s):
    with torch.no_grad():
        #---> size : (1, 4)
        s_batch = np.expand_dims(s, axis=0)
        s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)
        # Get logits from state; size : (2)
        logits = actor_func(s_batch).squeeze(dim=0)
        # From logits to policy
        policy = F.softmax(logits, dim=-1)
        # Pick up action's sample; size : ()
        a = torch.multinomial(policy, num_samples=1).squeeze(dim=0)
        # Calculate log probability
        log_pi = -F.cross_entropy(logits, a, reduction="none")
        # Return
        return a.tolist(), logits.tolist(), log_pi.tolist()

def kl(logits_1,logits_2):
    """
    Calculate KL-divergence for Softmax(logits_1) and Softmax(logits_2)
    P1 = Softmax(logits_1), P2 = Softmax(logits_2)
    Return :
    _kl = sum_a {P1[a] * log (P1[a]/P2[a])}
    """
    logits_old = logits_1 - torch.amax(logits_1, dim=1, keepdim=True) 
    logits_new = logits_2 - torch.amax(logits_2, dim=1, keepdim=True) 
    exp_old = torch.exp(logits_old)
    exp_new = torch.exp(logits_new)
    exp_old_sum = torch.sum(exp_old, dim=1, keepdim=True)
    exp_new_sum = torch.sum(exp_new, dim=1, keepdim=True)
    pi_old = exp_old / exp_old_sum
    _kl = torch.sum(
        pi_old * (logits_old - torch.log(exp_old_sum) - logits_new + torch.log(exp_new_sum)),
        dim=1,
        keepdim=True)
    return _kl

def cumulative_rewards(rewards):
    """
    Calculate and store cumulative rewards
    Return:
    cum_rewards (array [float])
    cum_rewards[t] = sum_{i = 0:t} gamma^{i} * rewards[t]
    """
    cum_rewards = np.zeros_like(rewards)
    reward_len = len(rewards)
    for j in reversed(range(reward_len)):
        cum_rewards[j] = rewards[j] + (cum_rewards[j+1]*gamma if j+1 < reward_len else 0)
    return cum_rewards


reward_records = []
all_params = list(actor_func.parameters()) + list(value_func.parameters())
opt = torch.optim.AdamW(all_params, lr=0.0005)
for i in range(5000):
    #--- Run episode till done
    done = False
    states = []
    actions = []
    logits = []
    logpis = []
    rewards = []
    s, _ = env.reset()
    while not done:
        states.append(s.tolist())
        a, l, p = pick_sample_and_logp(s)
        s, r, term, trunc, _ = env.step(a)
        done = term or trunc
        actions.append(a)
        logits.append(l)
        logpis.append(p)
        rewards.append(r)

    #--- Get cumulative rewards
    cum_rewards = cumulative_rewards(rewards)
    
    #--- Train (optimize parameters)
    opt.zero_grad()
    states = torch.tensor(states, dtype=torch.float).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    logits_old = torch.tensor(logits, dtype=torch.float).to(device)
    logpis = torch.tensor(logpis, dtype=torch.float).to(device).unsqueeze(dim=1)
    cum_rewards = torch.tensor(cum_rewards, dtype=torch.float).to(device).unsqueeze(dim=1)
    # Get values and logits with new parameters
    values_new = value_func(states)
    logits_new = actor_func(states)
    # Get advantages
    advantages = cum_rewards - values_new
    # Calculate pi_new / pi_old
    logpis_new = -F.cross_entropy(logits_new, actions, reduction="none").unsqueeze(dim=1)
    ratio = torch.exp(logpis_new - logpis)
    # Calculate KL-div for loss
    kl_loss = kl(logits_1 = logits_old,logits_2 = logits_new)
    # Get value loss
    vf_loss = F.mse_loss(values_new,cum_rewards,reduction="none")
    # Get total loss
    loss = -advantages * ratio + kl_loss * kl_coeff + vf_loss * vf_coeff
    # Optimize
    loss.sum().backward()
    opt.step()

    # Output total rewards in episode (max 500)
    print("Run episode{} with rewards {}".format(i, np.sum(rewards)), end="\r")
    reward_records.append(np.sum(rewards))

    # # stop if reward mean > 475.0
    # if np.average(reward_records[-50:]) > 475.0:
    #     break

print("\nDone")
env.close()





import matplotlib.pyplot as plt
# Generate recent 50 interval average
average_reward = []
for idx in range(len(reward_records)):
    avg_list = np.empty(shape=(1,), dtype=int)
    if idx < 50:
        avg_list = reward_records[:idx+1]
    else:
        avg_list = reward_records[idx-49:idx+1]
    average_reward.append(np.average(avg_list))
plt.plot(reward_records)
plt.plot(average_reward)
plt.title('PPO-Penalty')
plt.xlabel('Episode')
plt.ylabel('Cumulative Rewards')
plt.show()







