"""
Enables synchronous multiprocess data parallelism.
Improves GPU usage by having multiple CPU feeders
to the RNN training process.
"""
from contextlib import closing
import sys

from absl import flags
import torch
import torch.multiprocessing as mp
from torch import optim

from .utils import get_device, disable_contiguous_rnn_warning

# TODO (much later -- try hogwild; don't use any sync whatsoever,
# just wait until a whole epoch is complete)

class SyncTrainer:
    """
    Accepts a model which already has share_memory() activated;
    multiplexes the shared model across several processes which
    independently compute sharded minibatch gradients for
    data-parallel SGD.
    """

    def __init__(self, model, n):
        self.n = n
        self._workers = [_Worker(self.n, i, model) for i in range(self.n)]

    def train(self, examples):
        """
        shard and perform fwd/bwd pass on a batch of examples, returning
        mean loss and accuracy.
        """
        for worker in self._workers:
            worker.train(examples)
        loss, acc, gradnorm = 0, 0, 0
        for worker in self._workers:
            worker_loss, worker_acc, worker_gradnorm = worker.train_finish()
            loss += worker_loss
            acc += worker_acc
            gradnorm += worker_gradnorm
        return loss, acc, gradnorm

    def step(self):
        """step grad on workers (and also zero it)"""
        for worker in self._workers:
            worker.step()
        for worker in self._workers:
            worker.step_finish()

    def zero_grad(self):
        """zero grad on workers"""
        for worker in self._workers:
            worker.zero_grad()
        for worker in self._workers:
            worker.zero_grad_finish()

    def lr(self, lr):
        """update worker lr"""
        for worker in self._workers:
            worker.lr(lr)
        for worker in self._workers:
            worker.lr_finish()
            
    def close(self):
        """close workers"""
        for worker in self._workers:
            worker.close()
        for worker in self._workers:
            worker.close_finish()
        self._workers = []


class _Remote:
    """
    Contains the methods performed by a remote process with a
    shared model.
    """

    def __init__(self, model):
        self.model = model
        self.optimizer = optim.SGD(model.parameters(), lr=0.1)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        """Steps and zeros"""
        self.optimizer.step()
        self.zero_grad()

    def train(self, examples, batch_size):
        return _train(self.model, examples, batch_size)

    def lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class _Worker:
    """
    A training process, over which batches are multiplexed.
    """
    def __init__(self, num_workers, worker_idx, model):
        self._worker_idx = worker_idx
        self._num_workers = num_workers
        fmt = '{: ' + str(len(str(num_workers))) + 'd}'
        self._id_str = ('worker ' + fmt + ' of ' + fmt).format(
            worker_idx, num_workers)

        ctx = mp.get_context('forkserver')
        self._conn, child_conn = ctx.Pipe()
        self._proc = ctx.Process(target=_child_loop, args=(
            self._conn, child_conn, self._id_str, model))
        self._proc.start()
        child_conn.close()

    def _lo(self, batch_size):
        return self._worker_idx * batch_size // self._num_workers

    def _hi(self, batch_size):
        return (self._worker_idx + 1) * batch_size // self._num_workers

    def _push(self, method_name, args, swallow_errors=False):
        try:
            self._conn.send((method_name, args))
        except IOError as e:
            if swallow_errors:
                msg = 'parent swallowing IOError {} from {}\n'.format(
                    str(e), self._id_str)
                print(msg, end='', file=sys.stderr)
            else:
                raise e

    def _pull(self, expected_name):
        method_name, result = self._conn.recv()
        assert method_name == expected_name, (method_name, expected_name)
        return result

    def train(self, examples):
        """remote batch sharding train step"""
        batch_size = len(examples)
        examples = examples[self._lo(batch_size):self._hi(batch_size)]
        self._push('train', (examples, batch_size))

    def train_finish(self):
        """
        wait until remote train completes; return loss, acc
        contribution of this worker's part of the batch.
        """
        return self._pull('train')

    def step(self):
        """Take an opt step and zero the gradient"""
        self._push('step', tuple())
        
    def step_finish(self):
        """wait for remote step to finish"""
        self._pull('step')

    def close(self):
        """initiate remote close"""
        # racy if, so we swallow errors.
        if self._proc.is_alive():
            self._push('close', tuple(), swallow_errors=True)

    def close_finish(self):
        """join remote worker process"""
        self._conn.close()
        self._proc.join()

    def lr(self, lr):
        """adjust learning rate"""
        self._push('lr', (lr,))
        
    def lr_finish(self):
        """wait until lr is adjusted"""
        self._pull('lr')

    def zero_grad(self):
        """zero the gradient"""
        self._push('zero_grad', tuple())

    def zero_grad_finish(self):
        """wait until gradient is zeroed"""
        self._pull('zero_grad')


def _train(model, examples, batch_size):
    agg_loss = 0
    agg_acc = 0
    loss_seed = torch.ones((), device=get_device()) / batch_size
    for ex in examples:
        prepared_ex = model.prepare_example(ex)
        loss, acc = model.forward(prepared_ex)
        # faster to just compute bwd pass and drop used activations
        loss.backward(loss_seed)
        agg_loss += loss.detach().cpu().numpy()
        agg_acc += acc.detach().cpu().numpy()
    grad = torch.cat(
        tuple(p.grad.data.view(-1) for p in model.parameters()))
    # won't be exact grad norm but avg of split grad norm batch
    gradnorm = torch.norm(grad).detach().cpu().numpy()
    agg_loss = agg_loss / batch_size
    agg_acc = agg_acc / batch_size
    return agg_loss, agg_acc, gradnorm


def _child_loop(parent_conn, conn, id_str, model):
    parent_conn.close()
    disable_contiguous_rnn_warning()
    try:
        with closing(conn):
            remote = _Remote(model)
            print('{} up and running\n'.format(id_str), end='')
            sys.stdout.flush()
            while True:
                method_name, args = conn.recv()
                if method_name == 'close':
                    return
                else:
                    method = getattr(remote, method_name)
                    ret = method(*args)
                    try:
                        conn.send((method_name, ret))
                    except IOError:
                        print('{} swallowing IOError\n'.format(id_str),
                              file=sys.stderr, end='')
                        sys.stderr.flush()
    except KeyboardInterrupt:
        print('{} exited cleanly on SIGINT\n'.format(id_str), end='',
              file=sys.stderr)
        sys.stderr.flush()
