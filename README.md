# Evolutionary Strategies in PyTorch

![](https://media.giphy.com/media/30pEMgYfiPliU87swt/giphy.gif)

A set of tools based on [evostra](https://github.com/alirezamika/evostra) for using [OpenAI's evolutionary strategies](https://blog.openai.com/evolution-strategies/) in PyTorch. Keras implementations using evostra will be provided with some examples.

TABLE OF CONTENTS
=================

- [Installation](#installation)
- [Usage](#usage)
- [Run](#run)

## Installation

Your system needs all the prerequisites for the minimal installation of OpenAI gym. These will differ by operating system, so please refer to the [gym repository](https://github.com/openai/gym) for detailed instructions for your build. You also need to install the PyTorch distribution of your [choice](http://pytorch.org/). You can trigger CUDA ops by passing in ```-c``` or ```--cuda``` to the training examples.

Following that, create a conda or virtualenv enviroment and run:

```shell
pip install -r requirements.txt
```

## Usage

You will find the strategy classes (one as of now) within ```evolutionary_strategies/strategies```. These classes are designed to be used with PyTorch models and take two parameters: a function to get a reward and a list of PyTorch Variables that correspond to parameter layers. This can be achieved in the following manner:

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
partial_func = partial(get_reward, model=model)
mother_parameters = list(model.parameters())

es = EvolutionModule(
    mother_parameters, partial_func, population_size=100,
    sigma=0.1, learning_rate=0.001,
    reward_goal=200, consecutive_goal_stopping=20,
    threadcount=10, cuda=cuda, render_test=True
)
```
  
* EvolutionModule
    - init
        - parameters (list of PyTorch Variables)
        - reward_function => float (runs episode and returns a reward)
        - population_size=50
        - sigma=0.1
        - learning_rate=0.001
        - decay=1.0
        - sigma_decay=1.0
        - threadcount=4
        - render_test=False
        - cuda=False
        - reward_goal=None
        - consecutive_goal_stopping=None (stops after n tests consecutively return rewards equal-to or greater-than goal)
        - save_path=None (path to save weights at test times)
    - run
        - iterations
        - print_step=10 (frequency with which to run test and save weights)

## Run

You can run the examples in the following manner:

```shell
PYTHONPATH=. python evolutionary_strategies/examples/cartpole/train_pytorch.py --weights_path cartpole_weights.p
```

## Examples

### Lunar Lander

Solved in 1200~ iterations: population=100, sigma=0.01, learning_rate=0.001.

![](https://media.giphy.com/media/30pEMgYfiPliU87swt/giphy.gif)

### Cartpole

Solved in 200 iterations: population=10, sigma=0.1, learning_rate=0.001.

![](https://media.giphy.com/media/5h9xfw3BXvztG4HVBi/giphy.gif)

