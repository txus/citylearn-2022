from typing import List
import numpy as np
from nptyping import NDArray, Shape, Float

###########################################################################
#####                Specify your reward function here                #####
###########################################################################


def marl_reward(individual_costs: NDArray[Shape['1'], Float]) -> NDArray[Shape['1'], Float]:
    collective_costs = individual_costs.mean()
    weighted_costs = individual_costs * 3 + collective_costs
    return -weighted_costs


def get_reward(electricity_consumption: List[float], carbon_emission: List[float], electricity_price: List[float], agent_ids: List[int]) -> List[float]:
        """CityLearn Challenge user reward calculation.

        Parameters
        ----------
        electricity_consumption: List[float]
            List of each building's/total district electricity consumption in [kWh].
        carbon_emission: List[float]
            List of each building's/total district carbon emissions in [kg_co2].
        electricity_price: List[float]
            List of each building's/total district electricity price in [$].
        agent_ids: List[int]
            List of agent IDs matching the ordering in `electricity_consumption`, `carbon_emission` and `electricity_price`.

        Returns
        -------
        rewards: List[float]
            Agent(s) reward(s) where the length of returned list is either = 1 (central agent controlling all buildings) 
            or = number of buildings (independent agent for each building).
        """

        carbon      = np.array(carbon_emission).clip(min=0)
        electricity = np.array(electricity_price).clip(min=0)
        
        return marl_reward(carbon) + marl_reward(electricity) # type: ignore

        total_electricity_spent = sum(electricity)
            
        electricity_demand = np.array(electricity_demand)
        
        # Use this reward function when running the MARLISA example with information_sharing = True. The reward sent to each agent will have an individual and a collective component.
        return list(np.sign(electricity_demand)*0.01*(np.array(np.abs(electricity_demand))**2 * max(0, total_electricity_demand)))

        # *********** BEGIN EDIT ***********
        # Replace with custom reward calculation
        carbon_emission = np.array(carbon_emission).clip(min=0)
        electricity_price = np.array(electricity_price).clip(min=0)
        reward = (carbon_emission + electricity_price)*-1
        # ************** END ***************
        
        return reward