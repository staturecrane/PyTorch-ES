# Evolutionary Strategies in PyTorch

A set of tools based on [evostra](https://github.com/alirezamika/evostra) for using [OpenAI's evolutionary strategies](https://blog.openai.com/evolution-strategies/) in PyTorch. Keras implementations using evostra will be provided with some or all examples. 

TABLE OF CONTENTS
=================

- [Installation](#installation)
- [Usage](#usage)
- [Run](#run)

## Installation

Your system needs all the prerequisites for the minimal installation of OpenAI gym. These will differ by operating system, so please refer to the [gym repository](https://github.com/openai/gym) for detailed instructions for your build.

Following that, create a conda or virtualenv enviroment and run:

```shell
pip install -r requirements.txt
```

You will find the strategy classes (one as of now) within ```evolution/strategies```. These classes are designed to be used with PyTorch models and take two parameters: a function to get a reward and a list of PyTorch Variables that correspond to parameter layers. This can be achieved in the following manner:

```python
import copy
from functools import partial

from evolution.strategies import EvolutionModule


def get_reward(model, weights):
    """
    This function runs your model and generates a reward
    """
    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data = weights[i]
        except:
            param.data = weights[i].data

    # run environment and return reward as an integer or float
    return 100


model = generate_pytorch_model()
# EvolutionModule runs the population in a ThreadPool, so
# if you need to inject other arguments, you can do that
# using the partial tool
partial_reward_func = partial(get_reward, model=model)

es = EvolutionModule(reward_func, list(model.parameters(), population=10, learning_rate=0.001, sigma=0.1, threadcount=8)
final_weights = es.run(100)
```

## Run

You can run the examples in the following manner:

```shell
PYTHONPATH=. python examples/cartpole/train_pytorch.py
```