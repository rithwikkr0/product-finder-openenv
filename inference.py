import os
from openai import OpenAI
from product_env import ProductEnv, Action

API_KEY = os.getenv("HF_TOKEN")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://router.huggingface.co/v1"
)

MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"

env = ProductEnv()

obs = env.reset()

print(f"[START] task=product env=my_env model={MODEL_NAME}")

step = 0
rewards = []

while True:

    step += 1

    prompt = f"""
    choose best product index based on lowest price:
    {obs.product}

    return only number index 0 or 1 or 2
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role":"user","content":prompt}]
    )

    choice = int(response.choices[0].message.content.strip()[0])

    obs, reward, done, _ = env.step(Action(choice=choice))

    rewards.append(reward.score)

    print(f"[STEP] step={step} action={choice} reward={reward.score:.2f} done={str(done).lower()} error=null")

    if done:
        break


score = sum(rewards)/len(rewards)

print(f"[END] success=true steps={step} score={score:.2f} rewards={','.join([format(r,'.2f') for r in rewards])}")
