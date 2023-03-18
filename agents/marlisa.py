import json
from collections import defaultdict
import os
from typing import DefaultDict, Dict, List, Optional, Tuple
from joblib import dump, load
import pickle
import shutil

import gym
import torch.optim as optim
import wandb
from citylearn.citylearn import CityLearnEnv
from gym.spaces import Box
from nptyping import Float, NDArray, Shape
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from agents.common import *

Action = List[float]
Encoder = NDArray[Shape, Float]
State = NDArray[Shape, Float]
CoordinationVars = NDArray[Shape["1, 2"], Float]


def GenStateActionFromJson(JsonPath, BuildingCount=5):

    with open(JsonPath) as json_file:
        buildings_states_actions = json.load(json_file)

    States = buildings_states_actions["observations"]
    Actions = buildings_states_actions["actions"]

    StateINFo = {}
    ActionINFo = {}
    INFos = {}

    for var, ins in States.items():
        # print(var, " <><> ", ins)
        if ins["active"]:
            StateINFo[var] = ins["active"]
    for act, ins in Actions.items():
        if ins["active"]:
            ActionINFo[act] = ins["active"]

    INFos["states"] = StateINFo
    INFos["action"] = ActionINFo

    return {"Building_" + str(key): INFos for key in range(1, BuildingCount + 1)}


class MarlisaAgent:
    def __init__(self, schema_path: str = "./data/citylearn_challenge_2022_phase_1/schema.json", learn_timesteps: int = 0, information_sharing: bool = False):
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")

        self.learn_timesteps = learn_timesteps

        env = CityLearnEnv(schema_path)

        observations_spaces, actions_spaces = env.observation_space, env.action_space

        building_info = env.get_building_information()

        def RenameBuilding(Building_Info):
            newB = {}
            for i, building in enumerate(Building_Info):
                newB["Building_" + str(i + 1)] = building_info[building]
            return newB

        building_info = RenameBuilding(building_info)

        BuildingStatesActions = GenStateActionFromJson(schema_path, BuildingCount=5)

        self.env = env

        # Instantiating the control agent(s)
        self.agents = MARLISA(
            building_ids=["Building_" + str(i) for i in [1, 2, 3, 4, 5]],
            buildings_states_actions=BuildingStatesActions,
            building_info=building_info,
            observation_spaces=observations_spaces,
            action_spaces=actions_spaces,
            information_sharing=information_sharing
        )
        self.coordination_variables = [[None, None] for _ in range(5)]

    def save(self, output_model):
        shutil.rmtree(output_model, ignore_errors=True)
        self.agents.save(output_model)

    def load(self, input_model):
        self.agents.load(input_model)

    def q_values(self) -> np.ndarray:
        return np.array(self.agents.q_values).transpose(1, 0)

    def eval(self):
        self.agents.eval()

    def raise_aicrowd_error(self, msg):
        raise NameError(msg)

    def register_reset(self, obs_dict: dict, deterministic=True, save_q_values=False):
        state = obs_dict["observation"]
        actions, coordination_vars = self.agents.select_action(
            state, deterministic=deterministic, save_q_values=save_q_values
        )
        self.previous = {
            "state": state,
            "actions": actions,
            "coordination_vars": coordination_vars,
        }
        return actions

    def compute_action(
        self, observations, deterministic=True, last_reward=None, save_q_values=False
    ):
        action_next, coordination_vars_next = self.agents.select_action(
            observations, deterministic=deterministic, save_q_values=save_q_values
        )
        if (
            (not self.agents.__eval__)
            and last_reward is not None
            and len(self.previous["actions"][0]) > 0
        ):
            wandb.log(
                {
                    "train/reward": np.mean(np.array(last_reward)),
                    "train/step": self.agents.time_step - 1,
                }
            )
            self.agents.add_to_buffer(
                self.previous["state"],
                self.previous["actions"],
                last_reward,
                observations,
                False,
                self.previous["coordination_vars"],
                coordination_vars_next,
            )
        self.previous = {
            "state": observations,
            "actions": action_next,
            "coordination_vars": coordination_vars_next,
        }
        return action_next

    def train(self):
        self.agents.training()

        steps = 0

        pbar = tqdm(total=self.learn_timesteps, unit="timesteps")

        while True and steps < self.learn_timesteps:
            state = self.env.reset()
            done = False

            action = self.register_reset({"observation": state}, deterministic=False)

            while not done:
                next_state, reward, done, _ = self.env.step(action)
                action = self.compute_action(
                    next_state, deterministic=False, last_reward=reward
                )
                steps += 1
                pbar.update(1)

                if steps >= self.learn_timesteps:
                    done = True

        pbar.close()


class Flags:
    def __init__(self, *names: str):
        self.flags: DefaultDict[str, int] = defaultdict(lambda: 0)
        self.names = names

    def inc(self, flag):
        if flag not in self.names:
            raise NotImplementedError
        self.flags[flag] += 1

    def get(self, flag) -> int:
        if flag not in self.names:
            raise NotImplementedError
        return self.flags[flag]


class RegressionPredictor:
    def __init__(
        self,
        observation_space: gym.Space,
        building_states_actions: dict,
        buffer_capacity: int,
        target_label: str,
    ):
        self.observation_space = observation_space
        self.building_states_actions = building_states_actions
        self.encoder, self.encoder_labels = self.__setup_encoder__()

        self.buffer = RegressionBuffer(buffer_capacity)

        if target_label not in self.encoder_labels.keys():
            raise Exception(
                "cannot predict {target_label}, which is not in the state space"
            )

        self.target_label = target_label

        self.estimator = LinearRegression()

    def save(self, path):
        os.makedirs(path)
        pickle.dump(self.buffer, open(f"{path}/buffer.pkl", "wb"))
        dump(self.estimator, f"{path}/regression.joblib")

    def load(self, path):
        self.buffer = pickle.load(open(f"{path}/buffer.pkl", "rb"))
        self.estimator = load(f"{path}/regression.joblib")

    def encode(self, state: State, action: Action) -> NDArray[Shape["1"], Float]:
        # exclude target label from the encoded state
        target_index = self.encoder_labels[self.target_label]
        xs = [j for j in self.encoder * state]
        xs = np.concatenate(
            (np.hstack(xs[:target_index]), np.hstack(xs[target_index + 1 :]), action)
        )
        return np.array([j for j in xs if j is not None])

    def add_example(self, state: State, action: Action, next_state: State):
        # Normalize all the states using periodical normalization, one-hot encoding, or -1, 1 scaling.
        x_reg = self.encode(state, action)
        y_reg = (self.encoder * next_state)[self.encoder_labels[self.target_label]]

        # Push inputs and targets to the regression buffer
        self.buffer.push(x_reg, y_reg)

    def train(self):
        self.estimator.fit(self.buffer.x, self.buffer.y)

    def predict(self, state: State, action: Action) -> float:
        x_reg = self.encode(state, action)
        return self.estimator.predict(x_reg.reshape(1, -1))

    def __setup_encoder__(self) -> Tuple[Encoder, Dict[str, int]]:
        encoder_labels = {}
        encoder = []

        state_n = 0
        for s_name, s in self.building_states_actions["states"].items():
            if not s:
                encoder.append(0)
            elif s_name in ["month", "hour"]:
                encoder.append(periodic_normalization(self.observation_space.high[state_n]))  # type: ignore
                state_n += 1
            elif s_name in [
                "outdoor_dry_bulb_temperature_predicted_6h",
                "outdoor_dry_bulb_temperature_predicted_12h",
                "outdoor_dry_bulb_temperature_predicted_24h",
                "outdoor_relative_humidity_predicted_6h",
                "outdoor_relative_humidity_predicted_12h",
                "outdoor_relative_humidity_predicted_24h",
                "diffuse_solar_irradiance_predicted_6h",
                "diffuse_solar_irradiance_predicted_12h",
                "diffuse_solar_irradiance_predicted_24h",
                "direct_solar_irradiance_predicted_6h",
                "direct_solar_irradiance_predicted_12h",
                "direct_solar_irradiance_predicted_24h",
            ]:
                encoder.append(remove_feature())
                state_n += 1
            else:
                # outdoor_dry_bulb_temperature
                # outdoor_relative_humidity
                # diffuse_solar_irradiance
                # direct_solar_irradiance
                # carbon_intensity
                # non_shiftable_load
                # solar_generation
                # electrical_storage_soc
                # net_electricity_consumption
                # electricity_pricing
                # electricity_pricing_predicted_6h
                # electricity_pricing_predicted_12h
                # electricity_pricing_predicted_24h
                encoder.append(no_normalization())
                state_n += 1

            encoder_labels[s_name] = len(encoder) - 1

        return np.array(encoder), encoder_labels


class SingleAgent:
    def __init__(
        self,
        uid: str,
        idx: int,
        building_info: dict,
        action_space: gym.Space,
        observation_space: gym.Space,
        building_states_actions: dict,
        pca_compression: float,
        information_sharing: bool,
        replay_buffer_capacity: float,
        regression_buffer_capacity: float,
        action_scaling_coef: float,
        hidden_dim: Tuple[int, int],
        soft_q_criterion: nn.Module,
        learning_rate: float,
        tau: float,
        discount: float,
        reward_scaling: float,
        device: torch.device,
        optimize_alpha: bool = False,
    ):
        self.uid = uid
        self.idx = idx
        self.building_info = building_info
        self.flags = Flags("pca", "regression")
        self.action_space = action_space
        self.observation_space = observation_space
        self.building_states_actions = building_states_actions
        self.soft_q_criterion = soft_q_criterion
        self.device = device
        self.action_scaling_coef = action_scaling_coef

        # information sharing
        self.information_sharing = information_sharing
        self.coordination_vars: CoordinationVars = np.array([0.0, 0.0])
        self.last_expected_consumption_prediction = 0.0
        self.last_expected_emmissions_prediction = 0.0

        self.norm_mean = 0.0
        self.norm_std = 1.0

        self.r_norm_mean = 0.0
        self.r_norm_std = 1.0

        # Whether to optimize the temperature parameter alpha, used for exploration
        # through entropy maximization
        self.optimize_alpha = optimize_alpha
        self.alpha = 0.2

        self.tau = tau
        self.discount = discount
        self.reward_scaling = reward_scaling

        self.state_estimator = LinearRegression()

        self.encoder = self.__setup_encoder__()

        self.pca, state_dim = self.__setup_pca__(
            pca_compression=pca_compression,
            with_information_sharing=information_sharing,
        )

        self.replay_buffer = ReplayBuffer(int(replay_buffer_capacity))

        self.consumption_predictor = RegressionPredictor(
            self.observation_space,
            building_states_actions,
            buffer_capacity=int(regression_buffer_capacity),
            target_label="net_electricity_consumption",
        )
        self.emmissions_predictor = RegressionPredictor(
            self.observation_space,
            building_states_actions,
            buffer_capacity=int(regression_buffer_capacity),
            target_label="carbon_intensity",
        )

        (
            self.soft_q_net1,
            self.soft_q_net2,
            self.target_soft_q_net1,
            self.target_soft_q_net2,
        ) = self.__setup_critics__(
            idx=idx,
            state_dim=state_dim,
            action_space=action_space,
            hidden_dim=hidden_dim,
            criterion=self.soft_q_criterion,
            device=device,
        )

        self.policy_net = self.__setup_actor__(
            idx=idx,
            state_dim=state_dim,
            action_space=self.action_space,
            action_scaling_coef=action_scaling_coef,
            hidden_dim=hidden_dim,
            device=device,
        )

        # Optimizers

        self.soft_q_optimizer1 = optim.Adam(
            self.soft_q_net1.parameters(), lr=learning_rate
        )
        self.soft_q_optimizer2 = optim.Adam(
            self.soft_q_net2.parameters(), lr=learning_rate
        )
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )

        self.target_entropy = -np.prod(self.action_space.shape).item()  # type: ignore
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)

    def save(self, path: str):
        os.makedirs(path)
        torch.save(self.policy_net.state_dict(), f"{path}/policy_net.pt")
        torch.save(self.soft_q_net1.state_dict(), f"{path}/soft_q_net1.pt")
        torch.save(self.soft_q_net2.state_dict(), f"{path}/soft_q_net2.pt")
        torch.save(self.target_soft_q_net1.state_dict(), f"{path}/target_soft_q_net1.pt")
        torch.save(self.target_soft_q_net2.state_dict(), f"{path}/target_soft_q_net2.pt")
        torch.save(self.log_alpha, f"{path}/log_alpha.pt")
        # save optimizers as well
        torch.save(self.soft_q_optimizer1.state_dict(), f"{path}/soft_q_optimizer1.pt")
        torch.save(self.soft_q_optimizer2.state_dict(), f"{path}/soft_q_optimizer2.pt")
        torch.save(self.policy_optimizer.state_dict(), f"{path}/policy_optimizer.pt")
        torch.save(self.alpha_optimizer.state_dict(), f"{path}/alpha_optimizer.pt")
        # save predictors
        self.consumption_predictor.save(f"{path}/consumption_predictor")
        self.emmissions_predictor.save(f"{path}/emmissions_predictor")
        dump(self.pca, f"{path}/pca.joblib")

    def load(self, path: str):
        self.policy_net.load_state_dict(torch.load(f"{path}/policy_net.pt"))
        self.soft_q_net1.load_state_dict(torch.load(f"{path}/soft_q_net1.pt"))
        self.soft_q_net2.load_state_dict(torch.load(f"{path}/soft_q_net2.pt"))
        self.target_soft_q_net1.load_state_dict(
            torch.load(f"{path}/target_soft_q_net1.pt")
        )
        self.target_soft_q_net2.load_state_dict(
            torch.load(f"{path}/target_soft_q_net2.pt")
        )
        self.log_alpha = torch.load(f"{path}/log_alpha.pt")

        self.soft_q_optimizer1.load_state_dict(
            torch.load(f"{path}/soft_q_optimizer1.pt")
        )
        self.soft_q_optimizer2.load_state_dict(
            torch.load(f"{path}/soft_q_optimizer2.pt")
        )
        self.policy_optimizer.load_state_dict(torch.load(f"{path}/policy_optimizer.pt"))
        self.alpha_optimizer.load_state_dict(torch.load(f"{path}/alpha_optimizer.pt"))
        self.consumption_predictor.load(f"{path}/consumption_predictor")
        self.emmissions_predictor.load(f"{path}/emmissions_predictor")
        self.pca = load(f"{path}/pca.joblib")

    def add_to_regression_buffer(self, state: State, action: Action, next_state: State):
        self.consumption_predictor.add_example(state, action, next_state)
        self.emmissions_predictor.add_example(state, action, next_state)

    def fit_regression_model(self):
        if self.information_sharing:
            self.consumption_predictor.train()
            self.emmissions_predictor.train()

    def add_to_replay_buffer(
        self,
        state: State,
        action: Action,
        reward: int,
        next_state: State,
        coordination_vars: CoordinationVars,
        coordination_vars_next: CoordinationVars,
        done: bool,
    ):
        # Normalize all the states using periodical normalization, one-hot encoding, or -1, 1 scaling. It also removes states that are not necessary (solar radiation if there are no solar PV panels).
        o = np.array([j for j in np.hstack(self.encoder * state) if j != None])  # type: ignore
        o2 = np.array([j for j in np.hstack(self.encoder * next_state) if j != None])  # type: ignore

        # Only executed during the random exploration phase. Pushes unnormalized tuples into the replay buffer.

        if self.information_sharing:
            o = np.hstack(np.concatenate((o, coordination_vars)))
            o2 = np.hstack(np.concatenate((o2, coordination_vars_next)))

        # Executed during the training phase. States and rewards pushed into the replay buffer are normalized and processed using PCA.
        if self.flags.get("pca") == 1:
            o = (o - self.norm_mean) / self.norm_std
            o = self.pca.transform(o.reshape(1, -1))[0]
            o2 = (o2 - self.norm_mean) / self.norm_std
            o2 = self.pca.transform(o2.reshape(1, -1))[0]
            r = (reward - self.r_norm_mean) / self.r_norm_std
        else:
            r = reward

        self.replay_buffer.push(o, action, r, o2, done)

    def reduce_replay_dimensionality(self):
        X = np.array([j[0] for j in self.replay_buffer.buffer])
        self.norm_mean = np.mean(X, axis=0)
        self.norm_std = np.std(X, axis=0) + 1e-5
        X = (X - self.norm_mean) / self.norm_std

        R = np.array([j[2] for j in self.replay_buffer.buffer])
        self.r_norm_mean = np.mean(R)
        self.r_norm_std = np.std(R) / self.reward_scaling + 1e-5

        self.pca.fit(X)
        new_buffer = []
        for s, a, r, s2, dones in self.replay_buffer.buffer:
            s_buffer = np.hstack(
                self.pca.transform(
                    ((s - self.norm_mean) / self.norm_std).reshape(1, -1)
                )[0]
            )
            s2_buffer = np.hstack(
                self.pca.transform(
                    ((s2 - self.norm_mean) / self.norm_std).reshape(1, -1)
                )[0]
            )
            new_buffer.append(
                (
                    s_buffer,
                    a,
                    (r - self.r_norm_mean) / self.r_norm_std,
                    s2_buffer,
                    dones,
                )
            )

        self.replay_buffer.buffer = new_buffer

    def q_value(self, state: State, action: Action):
        padded_state = np.concatenate([state, np.zeros(len(self.norm_mean) - len(state))])  # type: ignore

        encoded_state = torch.FloatTensor(
            self.pca.transform(
                ((padded_state - self.norm_mean) / self.norm_std).reshape(1, -1)
            )[0]
        ).unsqueeze(dim=0)

        encoded_action = torch.FloatTensor(action).unsqueeze(dim=0)

        return torch.min(
            self.target_soft_q_net1(encoded_state, encoded_action),
            self.target_soft_q_net2(encoded_state, encoded_action),
        )

    def perform_update(self, batch_size: int, time_step: int):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        if self.device.type == "cuda":
            state = torch.cuda.FloatTensor(state).to(self.device)  # type: ignore
            next_state = torch.cuda.FloatTensor(next_state).to(self.device)  # type: ignore
            action = torch.cuda.FloatTensor(action).to(self.device)  # type: ignore
            reward = torch.cuda.FloatTensor(reward).unsqueeze(1).to(self.device)  # type: ignore
            done = torch.cuda.FloatTensor(done).unsqueeze(1).to(self.device)  # type: ignore
        else:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        last_mean_q_target = 0.0

        with torch.no_grad():
            # Update Q-values. First, sample an action from the Gaussian policy/distribution for the current (next) state and its associated log probability of occurrence.
            new_next_actions, new_log_pi, _ = self.policy_net.sample(next_state)

            # The updated Q-value is found by subtracting the logprob of the sampled action (proportional to the entropy) to the Q-values estimated by the target networks.
            target_q_values = (
                torch.min(
                    self.target_soft_q_net1(next_state, new_next_actions),
                    self.target_soft_q_net2(next_state, new_next_actions),
                )
                - self.alpha * new_log_pi
            )

            q_target = reward + (1 - done) * self.discount * target_q_values
            last_mean_q_target = q_target.mean().item()

        # Update Soft Q-Networks
        q1_pred = self.soft_q_net1(state, action)
        q2_pred = self.soft_q_net2(state, action)

        q1_loss = self.soft_q_criterion(q1_pred, q_target)
        q2_loss = self.soft_q_criterion(q2_pred, q_target)

        self.soft_q_optimizer1.zero_grad()
        q1_loss.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q2_loss.backward()
        self.soft_q_optimizer2.step()

        # Update Policy
        new_actions, log_pi, _ = self.policy_net.sample(state)

        q_new_actions = torch.min(
            self.soft_q_net1(state, new_actions), self.soft_q_net2(state, new_actions)
        )

        policy_loss = (self.alpha * log_pi - q_new_actions).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        extra_log = {}

        if self.optimize_alpha:
            # Optimize the temperature parameter alpha, used for exploration through entropy maximization
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

            extra_log["train/alpha_loss"] = alpha_loss.item()
        else:
            self.alpha = 0.2

        wandb.log(
            {
                "train/q_target": last_mean_q_target,
                "train/actor_loss": policy_loss.item(),
                "train/critic1_loss": q1_loss.item(),
                "train/critic2_loss": q2_loss.item(),
                "train/log_pi": log_pi.mean(),
                "train/step": time_step,
                "train/building": self.uid,
                **extra_log,
            }
        )

        # Soft Updates
        for target_param, param in zip(
            self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(
            self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def act(self, state: State, deterministic: bool = False) -> Action:
        state_ = np.array([j for j in np.hstack(self.encoder * state) if j != None])  # type: ignore

        # Adding shared information to the state
        if self.information_sharing:
            state_ = np.hstack(np.concatenate((state_, self.coordination_vars)))

        state_ = (state_ - self.norm_mean) / self.norm_std
        state_ = self.pca.transform(state_.reshape(1, -1))[0]
        state_ = torch.FloatTensor(state_).unsqueeze(0).to(self.device)

        if deterministic is False:
            action, _, _ = self.policy_net.sample(state_)
        else:
            _, _, action = self.policy_net.sample(state_)

        return action.detach().cpu().numpy()[0]

    def explore(self, state: State, safely: bool = False) -> Action:
        if safely:
            # follow a rule-based policy to explore safely
            multiplier = 0.4
            hour_day = state[2]
            a_dim = len(self.action_space.sample())

            act = [0.0 for _ in range(a_dim)]
            if hour_day >= 7 and hour_day <= 11:
                act = [-0.05 * multiplier for _ in range(a_dim)]
            elif hour_day >= 12 and hour_day <= 15:
                act = [-0.05 * multiplier for _ in range(a_dim)]
            elif hour_day >= 16 and hour_day <= 18:
                act = [-0.11 * multiplier for _ in range(a_dim)]
            elif hour_day >= 19 and hour_day <= 22:
                act = [-0.06 * multiplier for _ in range(a_dim)]

            # Early nightime: store DHW and/or cooling energy
            if hour_day >= 23 and hour_day <= 24:
                act = [0.085 * multiplier for _ in range(a_dim)]
            elif hour_day >= 1 and hour_day <= 6:
                act = [0.1383 * multiplier for _ in range(a_dim)]
        else:
            act = self.action_scaling_coef * self.action_space.sample()

        return act

    def predict_expected_consumption_and_emmissions(
        self, state: NDArray[Shape, Float], action: Action
    ) -> Tuple[float, float]:
        consumption = self.consumption_predictor.predict(state, action)
        emmissions = self.emmissions_predictor.predict(state, action)
        self.last_expected_consumption_prediction = consumption
        self.last_expected_emmissions_prediction = emmissions
        return consumption, emmissions

    def __setup_critics__(
        self,
        idx: int,
        state_dim: int,
        action_space: gym.Space,
        hidden_dim: Tuple[int, int],
        criterion: nn.Module,
        device: torch.device,
    ) -> Tuple[SoftQNetwork, SoftQNetwork, SoftQNetwork, SoftQNetwork]:
        action_dim = action_space.shape[0]  # type: ignore

        soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        # wandb.watch(
        #     soft_q_net1,
        #     criterion=criterion,
        #     log="gradients",
        #     idx=(idx * 5) + 1,
        #     log_graph=False,
        # )
        # wandb.watch(
        #     soft_q_net2,
        #     criterion=criterion,
        #     log="gradients",
        #     idx=(idx * 5) + 2,
        #     log_graph=False,
        # )

        target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        # wandb.watch(target_soft_q_net1, log="gradients", idx=(idx * 5) + 3, log_graph=False)
        # wandb.watch(target_soft_q_net2, log="gradients", idx=(idx * 5) + 4, log_graph=False)

        for target_param, param in zip(
            target_soft_q_net1.parameters(), soft_q_net1.parameters()
        ):
            target_param.data.copy_(param.data)

        for target_param, param in zip(
            target_soft_q_net2.parameters(), soft_q_net2.parameters()
        ):
            target_param.data.copy_(param.data)

        return soft_q_net1, soft_q_net2, target_soft_q_net1, target_soft_q_net2

    def __setup_actor__(
        self,
        idx: int,
        state_dim: int,
        action_space: gym.Space,
        action_scaling_coef: float,
        hidden_dim: Tuple[int, int],
        device: torch.device,
    ) -> PolicyNetwork:
        action_dim = action_space.shape[0]  # type: ignore

        policy_net = PolicyNetwork(
            state_dim, action_dim, action_space, action_scaling_coef, hidden_dim
        ).to(device)

        # wandb.watch(policy_net, log="gradients", idx=idx * 5, log_graph=False)

        return policy_net

    def __setup_encoder__(self) -> Encoder:
        encoder = []
        state_n = 0
        for s_name, s in self.building_states_actions["states"].items():
            if not s:
                encoder.append(0)
            elif s_name in ["month", "hour"]:
                encoder.append(periodic_normalization(self.observation_space.high[state_n]))  # type: ignore
                state_n += 1
            elif s_name == "day":
                encoder.append(onehot_encoding([1, 2, 3, 4, 5, 6, 7, 8]))
                state_n += 1
            elif s_name == "daylight_savings_status":
                encoder.append(onehot_encoding([0, 1]))
                state_n += 1
            elif s_name == "net_electricity_consumption":
                encoder.append(remove_feature())
                state_n += 1
            else:
                encoder.append(normalize(self.observation_space.low[state_n], self.observation_space.high[state_n]))  # type: ignore
                state_n += 1

        return np.array(encoder)

    # Set up the encoder that will transform the states used by the regression model
    # to predict the net-electricity consumption
    def __setup_regression_encoder__(self) -> Tuple[Encoder, Dict[str, int]]:
        encoder_labels = {}
        encoder = []

        state_n = 0
        for s_name, s in self.building_states_actions["states"].items():
            if not s:
                encoder.append(0)
            elif s_name in ["month", "hour"]:
                encoder.append(periodic_normalization(self.observation_space.high[state_n]))  # type: ignore
                state_n += 1
            elif s_name in [
                "outdoor_dry_bulb_temperature_predicted_6h",
                "outdoor_dry_bulb_temperature_predicted_12h",
                "outdoor_dry_bulb_temperature_predicted_24h",
                "outdoor_relative_humidity_predicted_6h",
                "outdoor_relative_humidity_predicted_12h",
                "outdoor_relative_humidity_predicted_24h",
                "diffuse_solar_irradiance_predicted_6h",
                "diffuse_solar_irradiance_predicted_12h",
                "diffuse_solar_irradiance_predicted_24h",
                "direct_solar_irradiance_predicted_6h",
                "direct_solar_irradiance_predicted_12h",
                "direct_solar_irradiance_predicted_24h",
            ]:
                encoder.append(remove_feature())
                state_n += 1
            else:
                # outdoor_dry_bulb_temperature
                # outdoor_relative_humidity
                # diffuse_solar_irradiance
                # direct_solar_irradiance
                # carbon_intensity
                # non_shiftable_load
                # solar_generation
                # electrical_storage_soc
                # net_electricity_consumption
                # electricity_pricing
                # electricity_pricing_predicted_6h
                # electricity_pricing_predicted_12h
                # electricity_pricing_predicted_24h
                encoder.append(no_normalization())
                state_n += 1

            encoder_labels[s_name] = len(encoder) - 1

        return np.array(encoder), encoder_labels

    # PCA will reduce the number of dimensions of the state space to 2/3 of its the original size
    # Must be setup after the encoder.
    def __setup_pca__(
        self, pca_compression: float, with_information_sharing: bool
    ) -> Tuple[PCA, int]:
        if not hasattr(self, "encoder"):
            raise Exception("You must set up call __setup_encoder__ first")

        if with_information_sharing:
            state_dim = int((pca_compression) * (2 + len([j for j in np.hstack(self.encoder * np.ones(len(self.observation_space.low))) if j != None])))  # type: ignore
        else:
            state_dim = int((pca_compression) * (len([j for j in np.hstack(self.encoder * np.ones(len(self.observation_space.low))) if j != None])))  # type: ignore

        return PCA(n_components=state_dim), state_dim


class MARLISA:  # type: ignore
    def __init__(
        self,
        building_ids,
        buildings_states_actions,
        building_info,
        observation_spaces: List[Box],
        action_spaces: List[Box],
        hidden_dim=[256, 256],
        discount=0.99,
        tau=5e-3,
        lr=3e-4,
        batch_size=256,
        replay_buffer_capacity=1e5,
        regression_buffer_capacity=3e4,
        start_training=600,
        exploration_period=7500,
        start_regression=500,
        information_sharing=True,
        pca_compression=0.95,
        action_scaling_coef=0.5,
        reward_scaling=5.0,
        update_per_step=2,
        iterations_as=2,
        safe_exploration=True,
        seed=0,
    ):

        assert (
            start_training > start_regression
        ), "start_training must be greater than start_regression"

        self.__eval__ = False

        self.buildings_states_actions = buildings_states_actions

        self.building_ids = building_ids
        self.start_training = start_training
        self.start_regression = start_regression
        self.discount = discount
        self.batch_size = batch_size
        self.tau = tau
        self.action_scaling_coef = action_scaling_coef
        self.reward_scaling = reward_scaling
        self.regression_freq = 2500
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.deterministic = False
        self.information_sharing = information_sharing
        self.update_per_step = update_per_step
        self.iterations_as = iterations_as
        self.safe_exploration = safe_exploration
        self.exploration_period = exploration_period
        self.allow_exploration = True

        self.q_values = []

        self.time_step = 0

        # Optimizers/Loss using the Huber loss
        self.soft_q_criterion = nn.SmoothL1Loss()

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Device: " + ("cuda" if torch.cuda.is_available() else "cpu"))

        self.single_agents = {
            uid: SingleAgent(
                uid,
                idx,
                building_info[uid],
                action_space=action_space,
                observation_space=observation_space,
                building_states_actions=buildings_states_actions[uid],
                pca_compression=pca_compression,
                information_sharing=information_sharing,
                replay_buffer_capacity=replay_buffer_capacity,
                regression_buffer_capacity=regression_buffer_capacity,
                action_scaling_coef=action_scaling_coef,
                hidden_dim=hidden_dim,
                soft_q_criterion=self.soft_q_criterion,
                learning_rate=lr,
                tau=tau,
                discount=discount,
                reward_scaling=reward_scaling,
                device=self.device,
            )
            for idx, (uid, action_space, observation_space) in enumerate(
                zip(building_ids, action_spaces, observation_spaces)
            )
        }

    def eval(self):
        self.allow_exploration = False
        self.__eval__ = True

    def save(self, output_model):
        for uid, agent in self.single_agents.items():
            agent.save(output_model + f"/{uid}")

    def load(self, input_model):
        for uid, agent in self.single_agents.items():
            agent.load(input_model + f"/{uid}")

    def training(self):
        self.allow_exploration = True
        self.__eval__ = False

    def training_regression_model(self) -> bool:
        return self.time_step >= self.start_regression

    def regression_model_already_trained(self) -> bool:
        return self.time_step > self.start_regression

    def training_model(self) -> bool:
        return self.time_step >= self.start_training

    def explore(
        self, action_order: List[int], agents: List[SingleAgent], states: List[State]
    ) -> Tuple[List[Action], Optional[NDArray[Shape, Float]]]:
        actions: List[Optional[Action]] = [None for _ in range(len(self.building_ids))]

        total_consumption = 0
        total_emmissions = 0

        for agent, state, agent_index in zip(agents, states, action_order):
            action = agent.explore(state, safely=self.safe_exploration)

            actions[agent_index] = action

            if self.regression_model_already_trained() and self.information_sharing:
                (
                    consumption,
                    emmissions,
                ) = agent.predict_expected_consumption_and_emmissions(state, action)
                total_consumption += consumption
                total_emmissions += emmissions

        if self.regression_model_already_trained() and self.information_sharing:
            for agent in agents:
                agent.coordination_vars[0] = (
                    total_consumption - agent.last_expected_consumption_prediction
                )
                agent.coordination_vars[1] = (
                    total_emmissions - agent.last_expected_emmissions_prediction
                )

        return actions, np.array([agent.coordination_vars for agent in agents])  # type: ignore

    def iterative_action_selection(
        self,
        action_order: List[int],
        building_ids: List[str],
        agents: List[SingleAgent],
        agents_next: List[SingleAgent],
        states: List[State],
        deterministic: bool = False,
    ) -> Tuple[List[Action], NDArray[Shape, Float]]:
        actions: List[Action] = [[] for _ in range(len(self.building_ids))]
        k = 0
        total_consumption = 0
        total_emmissions = 0

        for iteration in range(self.iterations_as):
            for agent, agent_next, state in zip(agents, agents_next, states):
                is_last_iteration = (
                    iteration == self.iterations_as - 1
                    and agent.uid == building_ids[-1]
                )

                action = agent.act(state, deterministic=deterministic)

                # Get the actions in the last iterations if sharing information
                if iteration == self.iterations_as - 1:
                    actions[action_order[k]] = action
                    k += 1

                (
                    _expected_consumption_prediction,
                    _expected_emmissions_prediction,
                ) = agent.predict_expected_consumption_and_emmissions(state, action)

                if is_last_iteration:
                    pass
                    # why is this commented out?
                    # _total_demand += expected_demand[uid]
                else:
                    total_consumption += (
                        _expected_consumption_prediction
                        - agent_next.last_expected_consumption_prediction
                    )
                    total_emmissions += (
                        _expected_emmissions_prediction
                        - agent_next.last_expected_emmissions_prediction
                    )

                if is_last_iteration:
                    pass
                else:
                    agent_next.coordination_vars[0] = total_consumption
                    agent_next.coordination_vars[1] = total_emmissions

        return actions, np.array([agent.coordination_vars for agent in agents])

    def select_action(
        self,
        states: List[State],
        deterministic: bool = False,
        save_q_values: bool = False,
    ) -> Tuple[List[Action], NDArray[Shape, Float]]:
        self.time_step += 1

        exploring = self.allow_exploration and self.time_step <= self.exploration_period

        # Randomize the order in which buildings will decide their actions
        action_order = np.array(range(len(self.building_ids)))
        np.random.shuffle(action_order)

        _building_ids = [self.building_ids[i] for i in action_order]
        _building_ids_next = [
            self.building_ids[action_order[(i + 1) % len(action_order)]]
            for i in range(len(action_order))
        ]

        agents = [self.single_agents[uid] for uid in _building_ids]
        agents_next = [self.single_agents[uid] for uid in _building_ids_next]

        _states = [states[i] for i in action_order]

        actions: List[Action] = [[] for _ in range(len(self.building_ids))]

        coordination_variables = [[None, None] for _ in range(len(self.building_ids))]

        if exploring:
            actions, coordination_variables = self.explore(
                list(action_order), agents, _states
            )
        else:
            if self.information_sharing:
                actions, coordination_variables = self.iterative_action_selection(
                    list(action_order), _building_ids, agents, agents_next, _states
                )
            else:
                for idx, (agent, state) in enumerate(zip(agents, _states)):
                    action = agent.act(state, deterministic=deterministic)
                    actions[action_order[idx]] = action

        if (
            save_q_values and len(actions[0]) > 0
        ):  # run the target network on state an actions
            with torch.no_grad():
                self.q_values.append(
                    torch.tensor(
                        [
                            agent.q_value(state, action)
                            for state, action, agent in zip(
                                states, actions, self.single_agents.values()
                            )
                        ]
                    ).numpy()
                )

        return actions, np.array(coordination_variables)

    def add_to_buffer(
        self,
        states,
        actions,
        rewards,
        next_states,
        done,
        coordination_vars,
        coordination_vars_next,
    ):
        if self.__eval__:
            return

        for (uid, o, a, r, o2, coord_vars, coord_vars_next) in zip(
            self.building_ids,
            states,
            actions,
            rewards,
            next_states,
            coordination_vars,
            coordination_vars_next,
        ):
            agent = self.single_agents[uid]

            agent.add_to_regression_buffer(o, a, o2)

            if self.training_regression_model() and (
                agent.flags.get("regression") < 2
                or self.time_step % self.regression_freq == 0
            ):
                agent.fit_regression_model()
                agent.flags.inc("regression")

            # Run once the regression model has been fitted
            if agent.flags.get("regression") > 1:
                agent.add_to_replay_buffer(
                    o, a, r, o2, coord_vars, coord_vars_next, done
                )

        if (
            (not self.__eval__)
            and self.training_model()
            and self.batch_size
            <= len(self.single_agents[self.building_ids[0]].replay_buffer)
        ):
            for agent in self.single_agents.values():
                # This code only runs once. Once the random exploration phase is over, we normalize all the states and rewards to make them have mean=0 and std=1, and apply PCA. We push the normalized compressed values back into the buffer, replacing the old buffer.
                if agent.flags.get("pca") == 0:
                    agent.reduce_replay_dimensionality()
                    agent.flags.inc("pca")

            for _ in range(self.update_per_step):
                for agent in self.single_agents.values():
                    agent.perform_update(self.batch_size, self.time_step)
