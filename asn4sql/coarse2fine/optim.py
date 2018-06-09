"""
Defines the optimizer for coarse2fine
"""

from absl import flags

import torch.optim as optim
from torch.nn.utils import clip_grad_norm

flags.DEFINE_float('learning_rate', 0.002, 'initial learning rate')
flags.DEFINE_float('learning_rate_decay', 0.98,
                   'learning rate decay; starts when no epoch-to-epoch '
                   'improvement is visible or we pass the start_decay_at mark')
flags.DEFINE_integer('start_decay_at', 8, 'start decay at this epoch')
flags.DEFINE_float('max_grad_norm', 5, 'maximum gradient norm')
flags.DEFINE_enum('optim', 'rmsprop',
                  ['sgd', 'adagrad', 'adadelta', 'adam', 'rmsprop'],
                  'optimizer')

class Optim:
    """
    Maintains intermediate optimization state.
    """

    def __init__(self):
        self.last_metric = None
        self.lr = flags.FLAGS.learning_rate
        self.max_grad_norm = flags.FLAGS.max_grad_norm
        self.method = flags.FLAGS.optim
        self.lr_decay = flags.FLAGS.learning_rate_decay
        self.start_decay_at = flags.FLAGS.start_decay_at
        self.start_decay = False
        self._step = 0
        self.betas = [0.9, 0.98]
        self.params = None
        self.optimizer = None

    def _setRate(self, lr):
        self.lr = lr
        self.optimizer.param_groups[0]['lr'] = self.lr

    def step(self):
        "Compute gradients norm."
        self._step += 1

        # Decay method used in tensor2tensor.
        warmup_steps = 4000
        self._setRate(
            flags.FLAGS.learning_rate *
            (flags.FLAGS.rnn_size ** (-0.5) *
             min(self._step ** (-0.5),
                 self._step * warmup_steps**(-1.5))))

        if self.max_grad_norm:
            clip_grad_norm(self.params, self.max_grad_norm)
        self.optimizer.step()

    def updateLearningRate(self, metric, epoch):
        """
        Decay learning rate if val perf does not improve
        or we hit the start_decay_at limit. Should be called every epoch.
        """

        if (self.start_decay_at is not None) and (epoch >= self.start_decay_at):
            self.start_decay = True
        if (self.last_metric is not None) and (metric is not None) and (metric > self.last_metric):
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

        self.last_metric = metric
        self.optimizer.param_groups[0]['lr'] = self.lr

    def set_parameters(self, params):
        """specify the parameters for the otpimization"""
        self.params = [p for p in params if p.requires_grad]
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.params, lr=self.lr, alpha=0.95)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr,
                                        betas=self.betas, eps=1e-9)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)
