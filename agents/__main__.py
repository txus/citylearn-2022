import click
import wandb
import wandb.util
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import matplotlib
import moviepy
import imageio
import numpy.random
matplotlib.use('agg')
from moviepy.video.io.bindings import mplfig_to_npimage
from agents.marlisa import MarlisaAgent

from agents.orderenforcingwrapper import OrderEnforcingAgent
from agents.ppo_agent import PPOAgent
from agents.rbc_agent import BasicRBCAgent
from agents.random_agent import RandomAgent
from citylearn.citylearn import CityLearnEnv


class Constants:
    schema_path = "./data/citylearn_challenge_2022_phase_1/schema.json"
    entity_name = "control-freaks"
    project_name = "citylearn-2022"


def action_space_to_dict(aspace):
    """Only for box space"""
    return {
        "high": aspace.high,
        "low": aspace.low,
        "shape": aspace.shape,
        "dtype": str(aspace.dtype),
    }


def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    obs_dict = {"action_space": action_space_dicts, "observation": observations}
    return obs_dict


@click.group()
def cli():
    pass

class MarlisaPlot:
    def __init__(self, building_count):
        plt.ion()
        px = 1/plt.rcParams['figure.dpi']  # type: ignore

        self.fig, axes = plt.subplots(figsize=(1440*px, 720*px), nrows=1, ncols=2) # type: ignore
        self.building_count = building_count

        right_gridspec = axes[1].get_subplotspec() # type: ignore
        right_subfig = self.fig.add_subfigure(right_gridspec)
        right_subfig.set_label('Q-values')

        self.value_axes = right_subfig.subplots(building_count, 1, sharex=True)
        self.value_plots = []
        for nn, ax in enumerate(self.value_axes): # type: ignore
            self.value_plots.extend(ax.plot([], [], 'r-'))
            if nn == len(self.value_axes) - 1: # type: ignore
                ax.set_xlabel("Time")
            if nn == 2:
                ax.set_ylabel("Q-value")

        self.action_ax = axes[0] # type: ignore

        self._rects = self.action_ax.bar(range(building_count), [0]*building_count, tick_label=[f"#{i+1}" for i in range(building_count)], align='center')
        self.action_ax.set_title('Policy')
        self.action_ax.set_ylim(-1, 1)
        self.action_ax.set_ylabel('Action')
        self.action_ax.set_xlabel('Building')

    def plot(self, agents_action, agents_values):
        for rect, h in zip(self._rects, agents_action):
            rect.set_height(h)
        self.action_ax.autoscale_view(True, False, False)
        self.action_ax.relim()

        keep_latest = 20
        for values, ax, plot in zip(agents_values, self.value_axes, self.value_plots): # type: ignore
            plot.set_data(range(keep_latest), values[-keep_latest:])
            ax.autoscale_view(True, True, True)
            ax.relim()

        return mplfig_to_npimage(self.fig)

class ValuePlot:
    def __init__(self, building_count, trailing_frames):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_title("Value")
        self.ax.set_xlabel("iteration")
        self.ax.set_ylabel("value")
        self.trailing_frames = trailing_frames
        self.values = []

        # initialize blank plot
        self.val_plt, = plt.plot([], [], 'r-')

    def plot(self):
        frame_num = len(self.values)
        idxs = slice(max(0, frame_num - self.trailing_frames), frame_num)
        self.val_plt.set_data(range(frame_num)[idxs], self.values[idxs])
        self.ax.autoscale_view(True, True, True)
        self.ax.relim()
        return mplfig_to_npimage(self.fig)


@cli.command()
@click.option("--episodes", default=5, help="Number of episodes.")
@click.option("--learn-timesteps", default=2000, help="Timesteps to learn, if there is any learning.")
@click.option(
    "--agent",
    type=click.Choice(["random", "rbc", "ppo", "marlisa"], case_sensitive=False),
    default="rbc",
    help="Agent to use.",
)
@click.option("--save-video", default=False, help="Save a video of the simulation.")
@click.option("--information-sharing/--no-information-sharing", default=False, help="Use information sharing for MARLISA.")
@click.option("--input-model", default=None, help="Where to load a trained model from")
def evaluate(episodes, learn_timesteps, agent, save_video = False, information_sharing = False, input_model = None):
    tags = []
    if agent in ["random", "rbc"]:
        tags.append("baseline")

    tags.append(f"{learn_timesteps}steps")

    run_id = wandb.util.generate_id()

    name = f"{agent}_{episodes}eps_{learn_timesteps}steps_{'information_sharing' if information_sharing else 'no_information_sharing'}_{run_id}" # type: ignore

    if input_model:
        original_run_name = input_model.split("/")[-1]
        name = f"{original_run_name}_eval_{run_id}" # type: ignore
        group = original_run_name # type: ignore
    else:
        group = name # type: ignore

    wandb.init(
        id=run_id,
        name=name,
        project=Constants.project_name,
        entity=Constants.entity_name,
        config={"episodes": episodes, "agent": agent, "learn_timesteps": learn_timesteps},
        sync_tensorboard=True,
        monitor_gym=True,
        tags=tags,
        group=group,
        job_type="eval" if input_model else "train"
    )

    env = CityLearnEnv(schema=Constants.schema_path)

    policy = None

    marlisa = False

    if agent == "rbc":
        agent = OrderEnforcingAgent(BasicRBCAgent())
    if agent == "random":
        agent = OrderEnforcingAgent(RandomAgent())
    if agent == "ppo":
        policy = PPOAgent(Constants.schema_path, learn_timesteps)
        agent = OrderEnforcingAgent(policy)
    if agent == "marlisa":
        marlisa = True
        agent = MarlisaAgent(Constants.schema_path, learn_timesteps, information_sharing=information_sharing)
        policy = agent

    if policy is not None:
        if input_model is not None:
            policy.load(input_model)
        else:
            print("Training policy...")
            policy.train()

            filename = f"trained/{wandb.run.name}" # type: ignore
            policy.save(filename)
            wandb.save(filename)

        print("Evaluating policy...")
        policy.eval()

    obs_dict = env_reset(env)

    agent_time_elapsed = 0

    plot = MarlisaPlot(5)

    # disable gradient
    with torch.no_grad():
        step_start = time.perf_counter()
        if marlisa:
            actions = agent.register_reset(obs_dict, save_q_values=save_video) # type: ignore
        else:
            actions = agent.register_reset(obs_dict)

        agent_time_elapsed += time.perf_counter() - step_start

        episodes_completed = 0
        num_steps = 0
        interrupted = False
        episode_metrics = []
        episode_video = []
        try:
            while True:
                observations, reward, done, _ = env.step(actions)
                if save_video and num_steps % 24 == 0:
                    if marlisa:
                        values = agent.q_values() # type: ignore
                    else:
                        values = np.zeros((5,0))

                    if marlisa and len(agent.q_values()) > 0 and len(actions[0]) > 0: # type: ignore
                        custom_plot = plot.plot(np.stack(actions), values)
                        env_render = env.render()
                        combined_plot = np.concatenate([custom_plot, env_render]).transpose((2,0,1)) # we want channel, height, width

                        episode_video.append(combined_plot)
                if done:
                    episodes_completed += 1
                    metrics_t = env.evaluate()
                    with_video = {}
                    if save_video:
                        with_video = {"eval/video": wandb.Video(np.array(episode_video), fps=4, caption='Episode {}'.format(episodes_completed))}
                    metrics = {
                        "eval/reward": reward,
                        "eval/price_cost": metrics_t[0],
                        "eval/emmission_cost": metrics_t[1],
                        "eval/grid_cost": metrics_t[2],
                        **with_video
                    }
                    if np.any(np.isnan(metrics_t)):
                        raise ValueError(
                            "Episode metrics are nan, please contact organizers"
                        )
                    episode_metrics.append(metrics)
                    print(
                        f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}",
                    )

                    obs_dict = env_reset(env)

                    step_start = time.perf_counter()
                    if marlisa:
                        actions = agent.register_reset(obs_dict, save_q_values=save_video) # type: ignore
                    else:
                        actions = agent.register_reset(obs_dict)
                    agent_time_elapsed += time.perf_counter() - step_start

                    wandb.log(metrics, step=num_steps)
                else:
                    wandb.log({'eval/reward': np.mean(np.array(reward)), 'step': num_steps, 'episode': episodes_completed})
                    step_start = time.perf_counter()
                    if marlisa:
                        actions = agent.compute_action(observations, save_q_values=save_video) # type: ignore
                    else:
                        actions = agent.compute_action(observations)
                    agent_time_elapsed += time.perf_counter() - step_start

                num_steps += 1
                if num_steps % 1000 == 0:
                    print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")

                if episodes_completed >= wandb.config.episodes:
                    break
        except KeyboardInterrupt:
            print("========================= Stopping Evaluation =========================")
            interrupted = True

        if not interrupted:
            print("=========================Completed=========================")

        if len(episode_metrics) > 0:
            print(
                "Average Price Cost:", np.mean([e["eval/price_cost"] for e in episode_metrics])
            )
            print(
                "Average Emmision Cost:",
                np.mean([e["eval/emmission_cost"] for e in episode_metrics]),
            )
            wandb.run.summary["eval/average_price_cost"] = np.mean( # type: ignore
                [e["eval/price_cost"] for e in episode_metrics]
            )
            wandb.run.summary["eval/average_emmission_cost"] = np.mean( # type: ignore
                [e["eval/emmission_cost"] for e in episode_metrics]
            )
            wandb.run.summary["eval/agent_time_elapsed_s"] = agent_time_elapsed # type: ignore

        print(f"Total time taken by agent: {agent_time_elapsed}s")
        wandb.finish()


if __name__ == "__main__":
    cli()
