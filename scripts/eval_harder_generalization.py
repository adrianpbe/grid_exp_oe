
"""Evalautes policies trained on environments that have different difficulty levels. Those policies
that solved the easier environments are evaluated on the harder environments to test generalization.
The policies' performance is compared to a random policy's performance on each tested environment.
A CSV like table is printed to stdout with the following columns:
- experiment: the name of the experiment
- evaluated_on: the environment id on which the policy was evaluated
- random_num_episodes: the number of episodes the random policy completed
- random_final_reward: the mean final reward of the random policy
- num_episodes: the number of episodes the trained policy completed
- final_reward: the mean final reward of the trained policy
"""
# uv run python evaluate.py --total_steps 5000  --random --env_id MiniGrid-DoorKey-6x6-v0
# example_output_data = """distshift1-v0_20250321-001307 MiniGrid-DistShift2-v0
# uv run python evaluate.py --total_steps 5000  --random --env_id MiniGrid-DistShift2-v0
# Total steps:  5000
# Total episodes: 61
# Mean reward: 0.000
# Std reward: 0.009
# Mean final reward: 0.013
# Std final reward: 0.085

# uv run python evaluate.py --total_steps 5000  --experiment experiments/simple_ppo_envs_comparison/distshift1-v0_20250321-001307 --env_id MiniGrid-DistShift2-v0 --num_envs 8
# no ckpt is provided, the best policy ckpt found will be used
# restored ckpts: experiments/simple_ppo_envs_comparison/distshift1-v0_20250321-001307/best_checkpoints/best_ppo_policy-41
# Total steps:  5000
# Total episodes: 181
# Mean reward: 0.000
# Std reward: 0.000
# Mean final reward: 0.000
# Std final reward: 0.000
# """
import subprocess


def run_command(command):
    p = subprocess.run(command.split(), capture_output=True)
    return p.stdout.decode("utf-8").split("\n")
 

def extract_stats(lines):
    num = None
    final = None
    for line in lines:
        if "Total episodes" in line:
            num = int(line.split(":")[1].strip())
        elif "Mean final reward" in line:
            final = float(line.split(":")[1].strip())
    return num, final

TOTAL_STEPS  = 5_000

experiments = [
    "distshift1-v0_20250321-001307",
    "doorkey-5x5-v0_20250321-012108",
    "keycorridors3r1-v0_20250321-034633",
    "lavacrossings9n1-v0_20250320-192909",
    "lavagaps5-v0_20250321-062826",
    "multiroom-n2-s4-v0_20250321-101102",
    "simplecrossings9n1-v0_20250320-215510",
]

harder_envs = [
    ["MiniGrid-DistShift2-v0"],
    ["MiniGrid-DoorKey-6x6-v0"],
    ["MiniGrid-KeyCorridorS3R2-v0"],

    ["MiniGrid-LavaCrossingS9N2-v0"],
    ["MiniGrid-LavaGapS5-v0"],
    ["MiniGrid-MultiRoom-N4-S5-v0"],
    ["MiniGrid-SimpleCrossingS9N2-v0"],

]

if __name__ == "__main__":
    FIELDS = ["experiment", "evaluated_on", "random_num_episodes", "random_final_reward", "num_episodes", "final_reward"]
    table_entries = []

    for experiment, hard_env in zip(experiments, harder_envs):
        for env_id in hard_env:
            # print(f"uv run python evaluate.py --total_steps {TOTAL_STEPS}  --random --env_id {env_id}")
            random_num_episodes, random_final_reward = extract_stats(run_command(f"uv run python evaluate.py --total_steps {TOTAL_STEPS}  --random --env_id {env_id}"))
            # print(f"uv run python evaluate.py --total_steps {TOTAL_STEPS}  --experiment experiments/simple_ppo_envs_comparison/{experiment} --env_id {env_id} --num_envs 8")
            num_episodes, final_reward =  extract_stats(run_command(f"uv run python evaluate.py --total_steps {TOTAL_STEPS}  --experiment experiments/simple_ppo_envs_comparison/{experiment} --env_id {env_id} --num_envs 8"))

            table_entries.append((experiment, env_id, random_num_episodes, random_final_reward, num_episodes, final_reward))

    print("\n".join(
        [",".join(str(e) for e in line) for line in [FIELDS] + table_entries])
    )
