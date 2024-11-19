import numpy as np

class TwoArmedBandit:
    '''
    Drifting two armed bandit task. Reward probabilities evolve
    according to Gaussian drift

    Parameters
    ----------
    sigma : float
        noise amount for the drift used to update the reward probs
    '''

    def __init__(
            self,
            sigma: float
        ) -> None:

        if sigma < 0:
            raise ValueError(f'sigma must be greater than 0')
        
        self._sigma = sigma
        self._init_reward_probs()

    def _init_reward_probs(self) -> None:
        self.reward_probs = np.random.rand(2)

    def _update_rewards(self) -> None:
        # update reward probabilities
        self.reward_probs += np.random.normal(0, self._sigma, 2)

        # bound between 0 and 1
        self.reward_probs = np.maximum(self.reward_probs, [0, 0])
        self.reward_probs = np.minimum(self.reward_probs, [1, 1])


    def step(self, choice: int) -> int:
        '''
        Samples reward associated with the choice according
        to the current reward probs

        Parameters
        ----------
        choice : int
            trial choice 0 or 1 corresponding to left or right
            port in the task
        
        Returns
        ----------
        int
            trial reward
        
        '''
        reward = int(np.random.rand() < self.reward_probs[choice])
        self._update_rewards()
        return reward
    

class QAgent:
    '''
    Q-Learning agent for solving the two armed bandit task

    Parameters
    ----------
    alpha : float
        learning rate
    
    beta : float
        sigmoid temperature
    '''
    def __init__(
            self,
            alpha: float,
            beta: float
        ) -> None:

        self.alpha = alpha
        self.beta = beta

        self.Q = np.full(2, 0.5)
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def sample_choice(self) -> int:
        '''
        Sample a choice according to the current Q-values

        Returns
        ----------
        int
            value can be 0 or 1 depending on the two Q-values
        '''
        logit = self.beta * np.diff(self.Q)
        p_left = self.sigmoid(logit)
        return int(np.random.rand() < p_left)
    
    def _update_q_values(self, choice: int, reward: int) -> None:
        '''
        update the q value given the reward according to the update
        equation given in Miller et al.
        '''
        self.Q[choice] = (1 - self.alpha) * self.Q[choice] + self.alpha * reward


class QAgentSession:
    '''
    Encapsulates both the TwoArmedBandit task and the QAgent

    Parameters
    ----------
    T : int
        number of trials in the session
    
    task_params : list
        parameters to give the TwoArmedBandit class
    
    agent_params : list
        parameters to give the QAgent class
    '''
    def __init__(
            self,
            T: int,
            task_params: list,
            agent_params: list
        ) -> None:

        self.task = TwoArmedBandit(*task_params)
        self.agent = QAgent(*agent_params)
        self.T = T

    
    def sample_session(self):
        '''
        Generate an episode of the two armed bandit task
        and the Q-learning agents performance.
        '''
        # init containers for the variables
        q_vals = np.zeros((self.T, 2))
        rewards = np.zeros(self.T)
        reward_probs = np.zeros((self.T, 2))
        choices = np.zeros(self.T)

        # iterate over trials
        for t in range(self.T):
            q_vals[t] = self.agent.Q
            reward_probs[t] = self.task.reward_probs

            choice = self.agent.sample_choice()
            reward = self.task.step(choice)
            self.agent._update_q_values(choice, reward)
            
            choices[t] = choice
            rewards[t] = reward
        
        session = {
            'Q-vals' : q_vals,
            'reward_probs' : reward_probs,
            'choices' : choices,
            'rewards' : rewards
        }
        return session