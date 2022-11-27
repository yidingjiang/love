# Adapted from https://github.com/ezliu/hrl
import collections
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import utils as torch_utils
import dqn_utils
import schedule
import replay
import embed

from torch.nn import functional as F
import tqdm
from grid_world import grid
from world3d import world3d


class DQNAgent(object):
    @classmethod
    def from_config(cls, config, env):
        dqn = DQNPolicy.from_config(config.get("policy"), env)
        replay_buffer = replay.ReplayBuffer.from_config(config.get("buffer"))
        optimizer = optim.Adam(dqn.parameters(), lr=config.get("learning_rate"))
        return cls(dqn, replay_buffer, optimizer, config.get("sync_target_freq"),
                   config.get("min_buffer_size"), config.get("batch_size"),
                   config.get("update_freq"), config.get("max_grad_norm"))

    def __init__(self, dqn, replay_buffer, optimizer, sync_freq,
                             min_buffer_size, batch_size, update_freq, max_grad_norm):
        """
        Args:
            dqn (DQNPolicy)
            replay_buffer (ReplayBuffer)
            optimizer (torch.Optimizer)
            sync_freq (int): number of updates between syncing the
                DQN target Q network
            min_buffer_size (int): replay buffer must be at least this large
                before taking grad updates
            batch_size (int): number of experience to sample per grad step
            update_freq (int): number of update calls per parameter update.
            max_grad_norm (float): gradient is clipped to this norm on each
                update
        """
        self._dqn = dqn
        self._replay_buffer = replay_buffer
        self._optimizer = optimizer
        self._sync_freq = sync_freq
        self._min_buffer_size = min_buffer_size
        self._batch_size = batch_size
        self._update_freq = update_freq
        self._max_grad_norm = max_grad_norm
        self._updates = 0

        self._losses = collections.deque(maxlen=100)
        self._grad_norms = collections.deque(maxlen=100)

    def behavioral_clone(self, dataset, seed, num_epochs=100, batch_size=64):
        """Assumes that dataset is list[episode = list[(s, a, s')]]."""
        def as_batches(indices):
            for i in range(0, len(indices), batch_size):
                yield indices[i: i + batch_size]

        for _ in tqdm.tqdm(range(num_epochs)):
            indices = list(range(len(dataset)))
            for batch in as_batches(indices):
                trajectories = [dataset[index] for index in batch]
                # list of batched state, action
                exp_batches = []
                losses = []
                for time_index in range(len(trajectories[0])):
                    states = [(torch.tensor(trajectory[time_index][0]),
                               torch.zeros(()).long())
                              for trajectory in trajectories]
                    actions = torch.tensor(np.stack(
                        [trajectory[time_index][1]
                         for trajectory in trajectories]))
                    q_values = self._dqn._Q(states, None)[0]
                    loss = F.cross_entropy(q_values, actions)
                    losses.append(loss)
                self._optimizer.zero_grad()
                total_loss = sum(losses)
                total_loss.backward()
                self._optimizer.step()


    def update(self, experience):
        """Updates agent on this experience.

        Args:
            experience (Experience): experience to update on.
        """
        self._replay_buffer.add(experience)

        if len(self._replay_buffer) >= self._min_buffer_size:
            if self._updates % self._update_freq == 0:
                experiences = self._replay_buffer.sample(self._batch_size)

                self._optimizer.zero_grad()
                loss = self._dqn.loss(experiences, np.ones(self._batch_size))
                loss.backward()
                self._losses.append(loss.item())

                # clip according to the max allowed grad norm
                grad_norm = torch_utils.clip_grad_norm_(
                        self._dqn.parameters(), self._max_grad_norm, norm_type=2)
                self._grad_norms.append(grad_norm.item())
                self._optimizer.step()

            if self._updates % self._sync_freq == 0:
                self._dqn.sync_target()

        self._updates += 1

    def act(self, state, prev_hidden_state=None, test=False):
        """Given the current state, returns an action.

        Args:
            state (State)

        Returns:
            action (int)
            hidden_state (object)
        """
        return self._dqn.act(state, prev_hidden_state=prev_hidden_state, test=test)

    @property
    def stats(self):
        def mean_with_default(l, default):
            if len(l) == 0:
                return default
            return np.mean(l)

        stats = self._dqn.stats
        stats["loss"] = mean_with_default(self._losses, None)
        stats["grad_norm"] = mean_with_default(self._grad_norms, None)
        return {"DQN/{}".format(k): v for k, v in stats.items()}

    def state_dict(self):
        """Returns a serializable dictionary containing all the relevant
        details from the class.

        Returns:
            state_dict (dict)
        """
        return {
                "dqn": self._dqn.state_dict(),
                #"replay_buffer": self._replay_buffer,
                "optimizer": self._optimizer.state_dict(),
                "sync_freq": self._sync_freq,
                "min_buffer_size": self._min_buffer_size,
                "batch_size": self._batch_size,
                "update_freq": self._update_freq,
                "max_grad_norm": self._max_grad_norm,
                "updates": self._updates,
        }

    def load_state_dict(self, state_dict):
        self._dqn.load_state_dict(state_dict["dqn"])
        #self._replay_buffer = state_dict["replay_buffer"]
        self._optimizer.load_state_dict(state_dict["optimizer"])
        self._sync_freq = state_dict["sync_freq"]
        self._min_buffer_size = state_dict["min_buffer_size"]
        self._batch_size = state_dict["batch_size"]
        self._update_freq = state_dict["update_freq"]
        self._max_grad_norm = state_dict["max_grad_norm"]
        self._updates = state_dict["updates"]


class DQNPolicy(nn.Module):
    @classmethod
    def from_config(cls, config, env):
        def embedder_factory():
            embedder_config = config.get("embedder")
            if isinstance(env.unwrapped, grid.ComPILEEnv):
                state_embedder = embed.CompILEEmbedder(
                        embedder_config.get("embed_dim"))
            elif isinstance(env.unwrapped, world3d.MultiTask3DEnv):
                state_embedder = embed.World3DEmbedder(
                        embedder_config.get("embed_dim"))
            else:
                raise ValueError()

            if embedder_config.get("type") == "recurrent":
                state_embedder = embed.RecurrentStateEmbedder(
                        state_embedder, embedder_config.get("embed_dim"))
            return state_embedder

        policy_type = config.get("type")
        if policy_type == "vanilla":
            pass
        elif policy_type == "recurrent":
            cls = RecurrentDQNPolicy
        else:
            raise ValueError("Unsupported policy type: {}".format(policy_type))

        epsilon_schedule = schedule.LinearSchedule.from_config(
                config.get("epsilon_schedule"))
        return cls(env.action_space.n, epsilon_schedule,
                   config.get("test_epsilon"), embedder_factory,
                   config.get("discount"))

    def __init__(self, num_actions, epsilon_schedule, test_epsilon,
                             state_embedder_factory, gamma=0.99):
        """DQNPolicy should typically be constructed via from_config, and not
        through the constructor.

        Args:
            num_actions (int): the number of possible actions to take at each
                state
            epsilon_schedule (Schedule): defines rate at which epsilon decays
            test_epsilon (float): epsilon to use during test time (when test is
                True in act)
            state_embedder_factory (Callable --> StateEmbedder): type of state
                embedder to use
            gamma (float): discount factor
        """
        super().__init__()
        self._Q = DuelingNetwork(num_actions, state_embedder_factory())
        self._target_Q = DuelingNetwork(num_actions, state_embedder_factory())
        self._num_actions = num_actions
        self._epsilon_schedule = epsilon_schedule
        self._test_epsilon = test_epsilon
        self._gamma = gamma

        # Used for generating statistics about the policy
        # Average of max Q values
        self._max_q = collections.deque(maxlen=1000)
        self._min_q = collections.deque(maxlen=1000)
        self._losses = collections.defaultdict(
                lambda: collections.deque(maxlen=1000))

    def act(self, state, prev_hidden_state=None, test=False):
        """
        Args:
            state (State)
            test (bool): if True, takes on the test epsilon value
            prev_hidden_state (object | None): unused agent state.
            epsilon (float | None): if not None, overrides the epsilon greedy
            schedule with this epsilon value. Mutually exclusive with test
            flag

        Returns:
            int: action
            hidden_state (None)
        """
        del prev_hidden_state

        q_values, hidden_state = self._Q([state], None)
        if test:
            epsilon = self._test_epsilon
        else:
            epsilon = self._epsilon_schedule.step()
        self._max_q.append(torch.max(q_values).item())
        self._min_q.append(torch.min(q_values).item())
        return epsilon_greedy(q_values, epsilon)[0], None

    def loss(self, experiences, weights):
        """Updates parameters from a batch of experiences

        Minimizing the loss:

            (target - Q(s, a))^2

            target = r if done
                     r + \gamma * max_a' Q(s', a')

        Args:
            experiences (list[Experience]): batch of experiences, state and
                next_state may be LazyFrames or np.arrays
            weights (list[float]): importance weights on each experience

        Returns:
            loss (torch.tensor): MSE loss on the experiences.
        """
        batch_size = len(experiences)
        states = [e.state for e in experiences]
        actions = torch.tensor([e.action for e in experiences]).long()
        next_states = [e.next_state for e in experiences]
        rewards = torch.tensor([e.reward for e in experiences]).float()

        # (batch_size,) 1 if was not done, otherwise 0
        not_done_mask = torch.tensor([1 - e.done for e in experiences]).byte()
        weights = torch.tensor(weights).float()

        current_state_q_values = self._Q(states, None)[0]
        current_state_q_values = current_state_q_values.gather(
                1, actions.unsqueeze(1))

        # DDQN
        best_actions = torch.max(self._Q(next_states, None)[0], 1)[1].unsqueeze(1)
        next_state_q_values = self._target_Q(next_states, None)[0].gather(
                1, best_actions).squeeze(1)
        targets = rewards + self._gamma * (
            next_state_q_values * not_done_mask.float())
        targets.detach_()  # Don't backprop through targets

        td_error = current_state_q_values.squeeze() - targets
        loss = torch.mean((td_error ** 2) * weights)
        self._losses["td_error"].append(loss.detach().cpu().data.numpy())
        return loss

    def sync_target(self):
        """Syncs the target Q values with the current Q values"""
        self._target_Q.load_state_dict(self._Q.state_dict())

    @property
    def stats(self):
        """See comments in constructor for more details about what these stats
        are"""
        def mean_with_default(l, default):
            if len(l) == 0:
                return default
            return np.mean(l)

        stats = {
                "epsilon": self._epsilon_schedule.step(take_step=False),
                "Max Q": mean_with_default(self._max_q, None),
                "Min Q": mean_with_default(self._min_q, None),
        }
        for name, losses in self._losses.items():
            stats[name] = np.mean(losses)
        return stats


class RecurrentDQNPolicy(DQNPolicy):
    """Implements a DQN policy that uses an RNN on the observations."""

    def loss(self, experiences, weights):
        """Updates recurrent parameters from a batch of sequential experiences

        Minimizing the DQN loss:

            (target - Q(s, a))^2

            target = r if done
                     r + \gamma * max_a' Q(s', a')

        Args:
            experiences (list[list[Experience]]): batch of sequences of experiences.
            weights (list[float]): importance weights on each experience

        Returns:
            loss (torch.tensor): MSE loss on the experiences.
        """
        unpadded_experiences = experiences
        experiences, mask = dqn_utils.pad(experiences)
        batch_size = len(experiences)
        seq_len = len(experiences[0])

        hidden_states = [seq[0].agent_state for seq in experiences]
        # Include the next states in here to minimize calls to _Q
        states = [
                [e.state for e in seq] + [seq[-1].next_state] for seq in experiences]
        actions = torch.tensor(
                [e.action for seq in experiences for e in seq]).long()
        next_hidden_states = [seq[0].next_agent_state for seq in experiences]
        next_states = [[e.next_state for e in seq] for seq in experiences]
        rewards = torch.tensor(
                [e.reward for seq in experiences for e in seq]).float()

        # (batch_size,) 1 if was not done, otherwise 0
        not_done_mask = ~(torch.tensor(
                [e.done for seq in experiences for e in seq]).bool())
        weights = torch.tensor(weights).float()

        # (batch_size, seq_len + 1, actions)
        q_values, _ = self._Q(states, hidden_states)
        current_q_values = q_values[:, :-1, :]
        current_q_values = current_q_values.reshape(batch_size * seq_len, -1)
        # (batch_size * seq_len, 1)
        current_state_q_values = current_q_values.gather(1, actions.unsqueeze(1))

        # DDQN
        next_q_values = q_values[:, 1:, :]
        # (batch_size * seq_len, actions)
        next_q_values = next_q_values.reshape(batch_size * seq_len, -1)
        best_actions = torch.max(next_q_values, 1)[1].unsqueeze(1)
        # Using the same hidden states for target
        target_q_values, _ = self._target_Q(next_states, next_hidden_states)
        target_q_values = target_q_values.reshape(batch_size * seq_len, -1)
        next_state_q_values = target_q_values.gather(1, best_actions).squeeze(1)
        targets = rewards + self._gamma * (
                next_state_q_values * not_done_mask.float())
        targets.detach_()  # Don't backprop through targets

        td_error = current_state_q_values.squeeze() - targets
        weights = weights.unsqueeze(1) * mask.float()
        loss = (td_error ** 2).reshape(batch_size, seq_len) * weights
        loss = loss.sum() / mask.sum()  # masked mean
        return loss

    def act(self, state, prev_hidden_state=None, test=False):
        """
        Args:
            state (State)
            test (bool): if True, takes on the test epsilon value
            prev_hidden_state (object | None): unused agent state.
            epsilon (float | None): if not None, overrides the epsilon greedy
            schedule with this epsilon value. Mutually exclusive with test
            flag

        Returns:
            int: action
            hidden_state (None)
        """
        q_values, hidden_state = self._Q([[state]], prev_hidden_state)
        if test:
            epsilon = self._test_epsilon
        else:
            epsilon = self._epsilon_schedule.step()
        self._max_q.append(torch.max(q_values).item())
        self._min_q.append(torch.min(q_values).item())
        return epsilon_greedy(q_values, epsilon)[0], hidden_state


class DQN(nn.Module):
    """Implements the Q-function."""
    def __init__(self, num_actions, state_embedder):
        """
        Args:
            num_actions (int): the number of possible actions at each state
            state_embedder (StateEmbedder): the state embedder to use
        """
        super(DQN, self).__init__()
        self._state_embedder = state_embedder
        self._q_values = nn.Linear(self._state_embedder.embed_dim, num_actions)

    def forward(self, states, hidden_states=None):
        """Returns Q-values for each of the states.

        Args:
            states (FloatTensor): shape (batch_size, 84, 84, 4)
            hidden_states (object | None): hidden state returned by previous call to
                forward. Must be called on constiguous states.

        Returns:
            FloatTensor: (batch_size, num_actions)
            hidden_state (object)
        """
        state_embed, hidden_state = self._state_embedder(states, hidden_states)
        return self._q_values(state_embed), hidden_state


class DuelingNetwork(DQN):
    """Implements the following Q-network:

        Q(s, a) = V(s) + A(s, a) - avg_a' A(s, a')
    """
    def __init__(self, num_actions, state_embedder):
        super(DuelingNetwork, self).__init__(num_actions, state_embedder)
        self._V = nn.Linear(self._state_embedder.embed_dim, 1)
        self._A = nn.Linear(self._state_embedder.embed_dim, num_actions)

    def forward(self, states, hidden_states=None):
        state_embedding, hidden_states = self._state_embedder(states, hidden_states)
        V = self._V(state_embedding)
        advantage = self._A(state_embedding)
        mean_advantage = torch.mean(advantage)
        return V + advantage - mean_advantage, hidden_states


def epsilon_greedy(q_values, epsilon):
    """Returns the index of the highest q value with prob 1 - epsilon,
    otherwise uniformly at random with prob epsilon.

    Args:
        q_values (Variable[FloatTensor]): (batch_size, num_actions)
        epsilon (float)

    Returns:
        list[int]: actions
    """
    batch_size, num_actions = q_values.size()
    _, max_indices = torch.max(q_values, 1)
    max_indices = max_indices.cpu().data.numpy()
    actions = []
    for i in range(batch_size):
        if np.random.random() > epsilon:
            actions.append(max_indices[i])
        else:
            actions.append(np.random.randint(0, num_actions))
    return actions
