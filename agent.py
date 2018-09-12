import tensorflow as tf
import numpy as np
from ReplayBuffer import ReplayBuffer
from OUN_noise import OUNoise
class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task, gamma, critic_learning_rate, actor_learning_rate,
    buffer_size, batch_size, tau, num_episodes):
        self.num_episodes = num_episodes
        self.task = task
        self.sess = tf.Session()
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_repeat = task.action_repeat
        self.c_lr = critic_learning_rate
        self.a_lr = actor_learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        # tensor placeholder
        self.s = tf.placeholder(tf.float32, [None, self.action_repeat* self.state_size], name='state')
        self.s_ = tf.placeholder(tf.float32, [None, self.action_repeat* self.state_size], name='next_state')
        #self.action = tf.placeholder(tf.float32, [None, self.action_size], name='action')
        self.r = tf.placeholder(tf.float32, [None, 1], 'reward')
        self.update_counter = 0

        # Noise process
        self.exploration_mu = 1
        self.exploration_theta = 0.3
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = buffer_size

        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        self.tau = tau  # for soft update of target parameters


        with tf.variable_scope('Actor'):
            # input s, output action
            self.a = self.build_actor_model(self.s, scope='eval_net', reuse=None)

            # input next_state, output next_action
            self.a_ = self.build_actor_model(self.s_, scope='target_net', reuse=tf.AUTO_REUSE)


        self.actor_eval_theta_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/eval_net')
        self.actor_target_theta_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/target_net')

        with tf.variable_scope('Critic'):
            # Input (s,a), output Q(s,a)
            self.qvalue = self.build_critic_model(self.s, self.a, 'eval_net', reuse=None)
            # Input (next_state, next_action), output Q(s_,a_)
            self.qvalue_ = self.build_critic_model(self.s_, self.a_, 'target', reuse=tf.AUTO_REUSE)


        self.critic_eval_theta_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic/eval_net')
        self.critic_target_theta_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic/target')
        #('target_q'):
        target_q = self.r + self.gamma * self.qvalue_
        #('TD_error'):
        self.loss = tf.reduce_mean(tf.squared_difference(target_q, self.qvalue))

        #gradients('Q(s,a)_w.r.t_action'):
        q_grad = tf.gradients(ys = self.qvalue, xs=self.a)[0]
        #gradients('action_w.r.t_theta'):
        policy_grads = tf.gradients(ys=self.a, xs=self.actor_eval_theta_params, grad_ys=q_grad)


        self.actor_train = tf.train.AdamOptimizer(-self.a_lr).apply_gradients(zip(policy_grads, self.actor_eval_theta_params))
        self.critic_train = tf.train.AdamOptimizer(self.c_lr).minimize(self.loss, var_list=self.critic_eval_theta_params)

        self.sess.run(tf.global_variables_initializer())

    def build_actor_model(self, input_state, scope, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False

        with tf.variable_scope(scope, reuse = reuse, custom_getter=custom_getter):
            init_w = tf.random_normal_initializer(0., 0.015)
            init_b = tf.constant_initializer(0.1)

            Policy_mue1 = tf.layers.dense(inputs =input_state, units=400, activation=None, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
            Policy_mue1_normal = tf.layers.batch_normalization(Policy_mue1, trainable=trainable)
            Policy_mue1 = tf.nn.relu(Policy_mue1_normal)
            Policy_mue1 = tf.layers.dropout(Policy_mue1, rate=0.5)

            Policy_mue2 = tf.layers.dense(inputs =Policy_mue1, units=200, activation=None, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
            Policy_mue2_normal = tf.layers.batch_normalization(Policy_mue2, trainable=trainable)
            Policy_mue2 = tf.nn.relu(Policy_mue2_normal)
            Policy_mue2 = tf.layers.dropout(Policy_mue2, rate=0.5)

            Policy_mue = tf.layers.dense(inputs =Policy_mue2, units=self.action_size, activation= tf.nn.tanh, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)



        # Scale [0, 1] output for each action dimension to proper range
        mean_action_range = (self.action_high - self.action_low)/2
        action = tf.multiply(Policy_mue[0], mean_action_range) + mean_action_range

        return action

    def build_critic_model(self, state, action,scope, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope(scope, reuse=reuse, custom_getter=custom_getter):
            init_w = tf.random_normal_initializer(0., 0.015)
            init_b = tf.constant_initializer(0.1)
            state_r = tf.reshape(state, [-1, self.state_size*self.action_repeat])
            action = tf.reshape(action, [-1, self.action_size])
            inner_n = 50

            w1_state = tf.get_variable('w1_s', [self.state_size*self.action_repeat, inner_n], trainable=trainable)
            w1_action = tf.get_variable('w1_a', [self.action_size, inner_n], trainable=trainable)
            b1 = tf.get_variable('b1', [1, inner_n],trainable=trainable)
            func_s_a = tf.add(tf.add(tf.matmul(state_r, w1_state) ,tf.matmul(action, w1_action)) , b1)

            
            Q1 = tf.layers.dense(inputs=func_s_a, units=400, activation=None, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
            Q1_normal = tf.layers.batch_normalization(Q1, trainable=trainable)
            Q1 = tf.nn.relu(Q1_normal)
            Q1 = tf.layers.dropout(Q1, rate=0.5)

            Q2 = tf.layers.dense(inputs=Q1, units=300, activation=None, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
            Q2_normal = tf.layers.batch_normalization(Q2, trainable=trainable)
            Q2 = tf.nn.relu(Q2_normal)
            Q2 = tf.layers.dropout(Q2, rate=0.5)
            Q2 = tf.layers.flatten(Q2)


            Q_value = tf.layers.dense(inputs =Q2, units= 1, activation= None, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)

        return Q_value

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.hstack([e.state for e in experiences if e is not None])
        states = np.reshape(states, [-1, self.action_repeat * self.state_size])
        #print(states.shape)
        assert states.shape == (self.batch_size, self.action_repeat * self.state_size)
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        actions = np.reshape(actions, [-1, self.action_size])
        assert actions.shape == (self.batch_size, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.hstack([e.next_state for e in experiences if e is not None])
        next_states = np.reshape(next_states, [-1, self.action_repeat* self.state_size])
        assert next_states.shape == (self.batch_size, self.action_repeat* self.state_size)

        self.sess.run(self.actor_train, feed_dict={self.s:states})
        self.sess.run(self.critic_train,feed_dict={self.s:states, self.r:rewards, self.s_:next_states})

        if self.update_counter % 200 == 1:
            self.params_update('hard')
            self.update_counter = 0
        else:
            self.params_update('soft')

        self.update_counter += 1





    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward state, action, reward, next_state, done
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state


    def params_update(self, replacement):
        """Soft update model parameters."""
        #assert len(self.actor_target_theta_params) == len(self.actor_eval_theta_params), "Local and target model parameters must have the same size"

        if replacement == 'hard':

            [tf.assign(target_c, eval_c) for target_c, eval_c in zip(self.actor_target_theta_params, self.actor_eval_theta_params)]
            [tf.assign(target_a, eval_a) for target_a, eval_a in zip(self.critic_target_theta_params, self.critic_eval_theta_params)]

        else:
            [tf.assign(target_c, (1 - self.tau) * target_c + self.tau * eval_c) for target_c, eval_c in zip(self.critic_target_theta_params, self.critic_eval_theta_params)]
            [tf.assign(target_a, (1 - self.tau) * target_a + self.tau * eval_a) for target_a, eval_a in zip(self.actor_target_theta_params, self.actor_eval_theta_params)]
