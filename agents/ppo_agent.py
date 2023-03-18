from stable_baselines3.ppo.ppo import PPO
import torch
import wandb
from wandb.integration.sb3 import WandbCallback
from agents.environment_wrapper import make_environment


class PPOAgent:
    def __init__(self, schema_path: str, learn_timesteps: int):
        env = make_environment(schema_path)

        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        self.model = PPO(
            "MlpPolicy", env, verbose=2, device=device, tensorboard_log=f"runs/ppo"
        )
        self.model.load("ppo_agent")
        self.learn_timesteps = learn_timesteps
        self.action_space = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def eval(self):
        self.model.policy.eval()

    def train(self):
        self.model.learn(
            total_timesteps=self.learn_timesteps,
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{wandb.run.id}", # type: ignore
                verbose=2,
                log="all",
            ),
        )
        self.model.policy.to(torch.device("cpu"))

    def save(self, output_path):
        self.model.save(output_path)

    def load(self, input_path):
        self.model.load(input_path)

    def register_reset(self, observation, action_space, agent_id):
        return self.compute_action(observation, agent_id)

    def compute_action(self, observation, agent_id):
        action, _states = self.model.predict(observation, deterministic=True)
        return action
