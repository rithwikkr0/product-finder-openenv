from pydantic import BaseModel

class Observation(BaseModel):
    product: str

class Action(BaseModel):
    choice: int

class Reward(BaseModel):
    score: float


class ProductEnv:

    def __init__(self):

        self.products = [
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
        return Observation(product=str(self.products[self.index]))


    def step(self, action: Action):

        options = self.products[self.index]

        best_price = min(options,key=lambda x:x[1])

        correct_index = options.index(best_price)

        reward = 1.0 if action.choice == correct_index else 0.0

        self.index += 1

        done = self.index >= len(self.products)

        if not done:
            obs = Observation(product=str(self.products[self.index]))
        else:
            obs = Observation(product="done")

        return obs, Reward(score=reward), done, {}


    def state(self):
        return {"index": self.index}
