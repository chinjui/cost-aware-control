import sys
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default="")
parser.add_argument("--macro-len", type=int, default=5)
parser.add_argument("--macro-hidden-size", type=int, default=32)
parser.add_argument("--sub-hidden-sizes", nargs="*", type=int, default=[8, 64])
args = parser.parse_args()

assert args.file != ""

sns.set(style="darkgrid")
exp_name = args.file.split('/')[-3]
env_id = args.file.split('/')[-2]
print("Expirience ID: %s, Env ID: %s" % (exp_name, env_id))

random_scores = {
    "HalfCheetah-v3": -292.10,
    "Swimmer-v3": -0.189,
    "Walker2d-v3": 1.625,
    "Ant-v3": -50.882,
    "Hopper-v3": 14.11,
    "InvertedDoublePendulum-v2": 59.09,
    "Reacher-v2": -42.32,
    "FetchPush-v1": 0.068,
    "FetchPickAndPlace-v1": 0.034,
    "FetchSlide-v1": 0
                }
scores = {
          # returns
          "HalfCheetah-v3":     {8: 1560, 64: 7490,   256: 8105},
          "Swimmer-v3":         {8: 32.3, 64: 46,     256: 81.7},
          "Walker2d-v3":        {8: 581,  64: 1660,   256: 4690},
          "Ant-v3":             {8: -23,  64: 1720,   256: 4350},
          "Hopper-v3":          {8: 390,  64: 3310,   256: 3060},
          "Humanoid-v3":        {8: 77,   64: 3050,   256: 5110},
          "HumanoidStandup-v2": {8: 1.2e5,64: 1.32e5, 256: 1.97e5},
          "Reacher-v2":         {8: -12.8,64: -5.3,   256: -4.7},
          "InvertedDoublePendulum-v2": {8: 6000, 64: 9190, 256: 9140},

          # success rates
          "FetchPush-v1":         {8: 0.07, 64: 0.98, 256: 1.00},
          "FetchPickAndPlace-v1": {8: 0.05, 64: 0.48, 256: 1.00},
          "FetchSlide-v1":        {8: 0.03, 64: 0.15, 256: 0.76},
          }

# policy costs (in flops) of policies with 2 hidden layers
costs = {8:  595,
         32: 3907,
         64: 11907,
         256: 145923}

# hidden8_256_file = "Ant-v1_hid8,256_ent1e-2_seed1179.txt"
#
# combined_returns = []
# macro_ratios = []
# policy0_returns = []
# policy1_returns = []
# need_return = False
#
# with open(hidden8_256_file) as f:
#   for line in f:
#     match = re.search('macro_acts: ([0-9\.]+)', line)
#     if match is not None:
#       macro_ratios.append(float(match.group(1)))
#       need_return = True
#
#     match = re.search('Episode .* return: ([0-9\.]+)', line)
#     if match is not None:
#       if need_return:
#         combined_returns.append(float(match.group(1)))
#         need_return = False
#
#     match = re.search('sub 0: ([0-9\.]+), sub 1: ([0-9\.]+),', line)
#     if match is not None:
#       policy0_returns.append(float(match.group(1)))
#       policy1_returns.append(float(match.group(2)))

# macro_ratios = macro_ratios[:len(combined_returns)]
# print("combined_returns;", len(combined_returns))
# print("macro_ratios:", len(macro_ratios))
# print("policy0_returns:", policy0_returns)
# print("policy1_reutnrs:", policy1_returns)
# policy0_returns = 3501.64
large_policy_score = scores[env_id][args.sub_hidden_sizes[1]]
random_score = random_scores[env_id]

# use the last 300 results
# combined_returns: returns of combined macro policy (i.e. macro, small, large)
with open(args.file, 'r') as f:
  data = []
  for i, line in enumerate(f):
    if i in [1, 4, 7]:
      # 1: macro_ratios, 4: returns, 7: success rates
      data.append(line.split(' '))
      for j in range(len(data[-1])):
        try:
          data[-1][j] = float(data[-1][j])
        except Exception as e:
          print(e, "data[%d][%d]: [%s]" % (len(data)-1, j, data[-1][j]))
          del data[-1][j]

if len(data) == 3:
  print("Length (macro, return, success): %d, %d, %d" % (len(data[0]), len(data[1]), len(data[2])))
elif len(data) == 2:
  print("Length (macro, return): %d, %d" % (len(data[0]), len(data[1])))
else:
  raise "Error"
macro_ratios = data[0]
combined_returns = data[1] if len(data) == 2 or (len(data) == 3 and len(data[2]) == 0) else data[2] # does this file contain success rates?
macro_ratios = macro_ratios[-500:]
combined_returns = combined_returns[-500:]


macro_cost = costs[args.macro_hidden_size]
policy_costs = [costs[args.sub_hidden_sizes[0]], costs[args.sub_hidden_sizes[1]]]
costs = [(ratio * policy_costs[1] + (1-ratio) * policy_costs[0] + (1 / args.macro_len) * macro_cost) / policy_costs[1] for ratio in macro_ratios]

# fig, axes = plt.subplots(ncols=2)

# plot 1: performance v.s. costs
if env_id == '':
  relative_perf = combined_returns
else:
  relative_perf = [r / large_policy_score for r in combined_returns]
  relative_perf = [(r-random_score) / (large_policy_score-random_score) for r in combined_returns]
d = pd.DataFrame(data={'Perf (0 ~ large policy score)': relative_perf, 'Costs (%)': costs})
g = sns.jointplot('Costs (%)', 'Perf (0 ~ large policy score)', data=d, color="m", kind='reg', ratio=3, marginal_kws=dict(bins=15))
g.ax_joint.set_ylabel('')
g.ax_joint.set_xlabel('')
g.ax_joint.set_xlim([0, 1.1])
g.ax_joint.set_ylim([-0.1, 1.5])
# axes[0].set_xlim([0, 1])
# axes[0].set_ylim([0, 1])
# plt.ylim(min(0, min(relative_perf)), max(1, max(relative_perf)))
# plt.xlim(0, 1)

# plot 2: both %
# relative_perf = [(r-policy0_returns) / (policy1_returns-policy0_returns) for r in combined_returns]
# d = pd.DataFrame(data={'Perf (small policy ~ large policy score)': relative_perf, 'Costs (%)': costs})
# g = sns.regplot('Perf (small policy ~ large policy score)', 'Costs (%)', data=d, color="m", ax=axes[1])
# plt.xlim(0, 1)
# plt.ylim(0, 1)
textstr = exp_name + '\n'
# textstr = textstr + "small policy: %.2f, large policy: %.2f\n" % (policy0_returns, policy1_returns)
# textstr += "costs: %.2f : %.2f" % (policy_costs[0], policy_costs[1])
# plt.suptitle(textstr, y=0.03)

plt.show()
