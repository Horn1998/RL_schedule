'''
本文件代码主要用于生成与DQNAgent进行对比的其它类型Agent
'''


from abc import ABC, abstractmethod
import numpy as np

from SimEnv import SimEnv

#简化版的DQNAgent
class Heuristik(ABC):
    """
    Defines the structure of a heuristik-agent
    """
    def __init__(self, env: SimEnv):
        self.env = env

    @classmethod
    @abstractmethod
    def _get_action(self):
        """
        Select the action according to the used policy
        """
        pass

    def schedule(self, epochs):
        
        ep_rewards = [0.0]

        # 对仿真过程进行追溯
        produced_parts = []
        ep_rewards_mean = []

        for epoch in range(epochs):
            next_sate = self.env.reset()
            while True:
                action = self._get_action()
                next_sate, reward, done, _ = self.env.step(action)

                ep_rewards[-1] += reward

                if done:
                    ep_rewards.append(0.0)
                    produced_parts.append(len(self.env.system.sink_store.items))
                    ep_rewards_mean.append(self._get_mean_reward(ep_rewards))

                    print('epoch:', epoch)
                    break
        
        return ep_rewards, produced_parts, ep_rewards_mean

    def _get_mean_reward(self, ep_rewards):
        """ mean reward over 100 episodes"""
        if len(ep_rewards) <= 100:
            return np.mean(ep_rewards[-len(ep_rewards):-1])
        else:
            return np.mean(ep_rewards[-100:-1])

#随机代理
class RandomAgent(Heuristik):
    """
    Selects actions according to a Random logic
    """
    
    def __init__(self, env:SimEnv):
        super().__init__(env)

    def _get_action(self):
        action = self.env.action_space.sample()
        return action

#
class FIFOAgent(Heuristik):
    """
    Selects actions according to a FIF0 logic
    """
    # 怎么代理没有减少维护资源？
    def __init__(self, env:SimEnv):
        super().__init__(env)
        self.actions = list(range(self.env.action_space.n))

    def _get_action(self):
        # if there is a maintenance resource available and at least one machine requested maintenance
        # 如果有可用的维护资源，并且至少有一台机器请求维护
        if self.env.system.available_maintenance > 0 and len(self.env.system.machines_to_repair) > 0:
             # assign as many maintenance resources as possibly
                for i in range(self.env.system.available_maintenance):
                    # check for machines still being available
                    if len(self.env.system.machines_to_repair) > 0:
                        # FIFO -> take first machine that requested repair
                        machine = self.env.system.machines_to_repair[0]

                        # remove the machine from the list of machines that requested maintenance
                        self.env.system.machines_to_repair.remove(machine)
                        self.env.logger.debug('Repairing machine {} with health {} due to FIFO maintenance logic. Machines waiting for repair: {}'.format(machine.id, machine.health,
                             self.env.system.machines_to_repair), extra = {'simtime': self.env.system.sim_env.now})
                    
                    # choose action
                    return self.env.system.machines.index(machine)

        else:
            # choose idle action 啥也不做
            return self.actions[-1]