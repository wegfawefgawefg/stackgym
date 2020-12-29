from collections import deque

import numpy as np
import gym

'''TODO:
    expose observation space
    expose state space
    expose all gym functions via meta python magic
'''

class StackGym:
    ''' Are you tired of rolling your own framestacking util for openai-gym?
        Be tired no longer my friend. Introducing, the NEW StackGym wrapper class. 
        
        Just provide a state formatter function in case you want to monochrome or shrink your input images, 
        and presto, the StackGym wrapper class will automagically give you stacks of frames to work with.

        By choosing StackGym you made life easy for yourself. Good job.
        
            -Please check example.py for an example state formatter function and usage.
        '''
    def __init__(self, env, frame_stack_size, 
            frame_skip_size=1, state_formatter=None, grayscale=False):
        '''
        -set grayscale=True if it is 1 channel image data
        -set grayscale=False if it is 3 channel rgb images
        '''
        self.env = env
        self.frame_stack_size = frame_stack_size
        self.frame_skip_size = frame_skip_size
        self.buffer_size = self.frame_stack_size * (frame_skip_size + 1)
        self.state_buffer = deque(maxlen=self.buffer_size)
        self.grayscale = grayscale

        if state_formatter is None:
            def identity(state):
                return state
            self.state_formatter = identity
        else:
            self.state_formatter = state_formatter

    def _stack_frames(self):
        states = list(self.state_buffer)[::-(self.frame_skip_size)]
        states = states[:self.frame_stack_size]
        stack = np.stack(states)
        if not self.grayscale:
            stack = stack.reshape((self.frame_stack_size * 3, *stack.shape[-2:]))
        
        return stack

    def step(self, action):
        state_, reward, done, info = self.env.step(action)
        state_ = self.state_formatter(state_)
        self.state_buffer.append(state_)
        stack = self._stack_frames()
        
        return stack, reward, done, info

    def render(self):
        self.env.render()

    def reset(self):
        state = self.env.reset()
        state = self.state_formatter(state)
        for _ in range(self.buffer_size):
            self.state_buffer.append(state)
        stack = self._stack_frames()

        return stack
