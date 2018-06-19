"""
Enables synchronous multiprocess data parallelism.
Improves GPU usage by having multiple CPU feeders
to the RNN training process.
"""
from contextlib import closing
import sys

import torch
import torch.multiprocessing as mp
from torch import optim

from . import log
from .utils import get_device, disable_contiguous_rnn_warning

# TODO (much later -- try hogwild; don't use any sync whatsoever,
# just wait until a whole epoch is complete)


class SharedGPU:
    """
    Accepts a model which already has share_memory() activated;
    multiplexes the shared model across several processes which
    independently compute sharded minibatch gradients for
    data-parallel SGD on the same GPU.

    If num workers is 0 then do everything in-process.

    This is useful for recurrent models which have lots of python
    process interactions during training and evaluation.
    """

    def __init__(self, model, n):
        self.n = n
        self._workers = [_Worker(self.n, i, model) for i in range(self.n)]
        self._local = None if n else _Remote(model)
        if self._local:
            log.debug('using a within-process worker')

    def set_mode(self, evaluation=False):
        """set the mode to eval if evaluation, else to train"""
        if self._local:
            self._local.set_mode(evaluation)
            return
        for worker in self._workers:
            worker.set_mode(evaluation)
        for worker in self._workers:
            worker.set_mode_finish(evaluation)

    def train(self, examples):
        """
        shard and perform fwd/bwd pass on a batch of examples, returning
        mean loss and accuracy.
        """
        if self._local:
            return self._local.train(examples, len(examples))
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
        if self._local:
            self._local.step()
            return
        for worker in self._workers:
            worker.step()
        for worker in self._workers:
            worker.step_finish()

    def zero_grad(self):
        """zero grad on workers"""
        if self._local:
            self._local.zero_grad()
            return
        for worker in self._workers:
            worker.zero_grad()
        for worker in self._workers:
            worker.zero_grad_finish()

    def lr(self, lr):
        """update worker lr"""
        if self._local:
            self._local.lr(lr)
            return
        for worker in self._workers:
            worker.lr(lr)
        for worker in self._workers:
            worker.lr_finish()

    def diagnose(self, examples):
        """
        returns a list of diagnostics from model evaluation on each example
        """
        if self._local:
            return self._local.diagnose(examples)
        for worker in self._workers:
            worker.diagnose(examples)
        diagnostics = []
        for worker in self._workers:
            diagnostics.extend(worker.diagnose_finish())
        return diagnostics

    def close(self):
        """close workers"""
        if self._local:
            return
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
        """zero the optimizer grad"""
        self.optimizer.zero_grad()

    def set_mode(self, evaluation=False):
        """set the model evaluation mode"""
        if evaluation:
            self.model.eval()
        else:
            self.model.train()

    def step(self):
        """Steps and zeros"""
        self.optimizer.step()
        self.zero_grad()

    def train(self, examples, batch_size):
        """take sgd steps over the given examples"""
        return _train(self.model, examples, batch_size)

    def lr(self, lr):
        """update the LR"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def diagnose(self, examples):
        """
        run diagnostics on each example for the model,
        return a list of diagnostic results for each example
        """
        return _diagnose(self.model, examples)


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
        self._proc = ctx.Process(
            target=_child_loop,
            args=(self._conn, child_conn, self._id_str, model))
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

    def diagnose(self, examples):
        """remote batch sharding diagnose step"""
        batch_size = len(examples)
        examples = examples[self._lo(batch_size):self._hi(batch_size)]
        self._push('diagnose', (examples, ))

    def diagnose_finish(self):
        """
        wait until remote diagnose completes; return loss, acc
        contribution of this worker's part of the batch.
        """
        return self._pull('diagnose')

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
        self._push('lr', (lr, ))

    def lr_finish(self):
        """wait until lr is adjusted"""
        self._pull('lr')

    def zero_grad(self):
        """zero the gradient"""
        self._push('zero_grad', tuple())

    def zero_grad_finish(self):
        """wait until gradient is zeroed"""
        self._pull('zero_grad')

    def set_mode(self, evaluation):
        """zero the gradient"""
        self._push('set_mode', (evaluation, ))

    def set_mode_finish(self):
        """wait until gradient is zeroed"""
        self._pull('set_mode')


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
    grad = tuple(
        p.grad.data.view(-1) for p in model.parameters()
        if p.grad is not None and p.grad.nelement() > 0)
    grad = grad or [torch.Tensor()]
    # won't be exact grad norm but avg of split grad norm batch
    grad = torch.cat(grad)
    gradnorm = torch.norm(grad).detach().cpu().numpy()

    agg_loss = agg_loss / batch_size
    agg_acc = agg_acc / batch_size
    return agg_loss, agg_acc, gradnorm


def _diagnose(model, examples):
    diagnostics = []
    for ex in examples:
        prepared_ex = model.prepare_example(ex)
        ex_diagnostics = model.diagnose(prepared_ex)
        with torch.no_grad():
            ex_diagnostics = {
                k: (v.cpu().detach().numpy(), fmt)
                for k, (v, fmt) in ex_diagnostics.items()
            }
        diagnostics.append(ex_diagnostics)
    return diagnostics


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
                        print(
                            '{} swallowing IOError\n'.format(id_str),
                            file=sys.stderr,
                            end='')
                        sys.stderr.flush()
    except KeyboardInterrupt:
        print(
            '{} exited cleanly on SIGINT\n'.format(id_str),
            end='',
            file=sys.stderr)
        sys.stderr.flush()
