from typing import List
from pydantic import BaseModel

# observation model
class Observation(BaseModel):
    options: List


# action model
class Action(BaseModel):
    choice: int


# reward model
class Reward(BaseModel):
    value: float


class ProductEnv:

    def __init__(self):

        self.tasks = [

            [
                ("iphone",52000,4.6,2),
                ("iphone",51000,4.5,3),
                ("iphone",53000,4.8,1)
            ],

            [
                ("laptop",60000,4.2,4),
                ("laptop",58000,4.1,3),
                ("laptop",62000,4.7,2)
            ],

            [
                ("earbuds",2000,4.3,2),
                ("earbuds",1800,4.0,5),
                ("earbuds",2100,4.8,1)
            ]

        ]

        self.index = 0


    def reset(self):

        self.index = 0

        return Observation(
            options=self.tasks[self.index]
        )


    def step(self, action: Action):

        options = self.tasks[self.index]

        cheapest = min(options,key=lambda x:x[1])

        correct_index = options.index(cheapest)

        reward = 1.0 if action.choice == correct_index else 0.0

        self.index += 1

        done = self.index >= len(self.tasks)

        if done:

            obs = Observation(options=[])

        else:

            obs = Observation(
                options=self.tasks[self.index]
            )


        return obs, Reward(value=reward), done, {}


    def state(self):

        return {

            "task_number": self.index

        }
