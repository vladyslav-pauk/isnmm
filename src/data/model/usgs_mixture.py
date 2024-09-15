from numpy import random

import torch
import torch.nn as nn
from torch.distributions import Distribution

from src.modules.network.nonlinear_transform import Network as NonlinearTransform
from src.modules.network.linear_positive import Network as LinearPositive


# todo: implement usgs semi-real data
