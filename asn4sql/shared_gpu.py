"""
Enables synchronous multiprocess data parallelism.
Improves GPU usage by having multiple CPU feeders
to the RNN training process.

Combine this with Nvidia's MPS to get parallelism in settings
where you need the GPU but the execution is highly dynamic
(e.g., not a regular feed-forward fixed-length or sequence-packed
input which can be batched).

https://stackoverflow.com/questions/34709749/
"""
from contextlib import closing
import sys

import track
import torch
import torch.multiprocessing as mp

from .utils import get_device, disable_contiguous_rnn_warning


class SharedGPU:
    """
    Accepts a model which already has share_memory() activated;
    multiplexes the shared model across several processes which
    independently compute sharded minibatch gradients for
    data-parallel gradient-based optimization on the same GPU.

    If num workers is 0 then do everything in-process.

    This is useful for recurrent models which have lots of python
    process interactions during training and evaluation.

    This class assumes it is the only thing modifying the gradients
    for the model parameters.
    """

    def __init__(self, optimizer, model, n):
        self.n = n
        grad_length = _grad_length(model)
        self._cat_grad = torch.zeros(
            [grad_length], requires_grad=False, device=get_device())
        self._cat_grad.share_memory_()
        model = model.share_memory()
        self._workers = [
            _Worker(self.n, i, model, self._cat_grad) for i in range(self.n)
        ]
        self.model = model
        self.optimizer = optimizer
        self._local = None if n else _Remote(model, self._cat_grad)
        if self._local:
            track.debug('using a within-process worker')

    def set_mode(self, evaluation=False):
        """set the mode to eval if evaluation, else to train"""
        if self._local:
            self._local.set_mode(evaluation)
            return
        for worker in self._workers:
            worker.set_mode(evaluation)
        for worker in self._workers:
            worker.set_mode_finish()

    def train(self, examples):
        """
        shard and perform fwd/bwd pass on a batch of examples, returning
        mean loss, mean accuracy, and current grad norm.
        """
        if self._local:
            loss, acc = self._local.train(examples, len(examples))
        else:
            for worker in self._workers:
                worker.train(examples)
            loss, acc = 0, 0
            for worker in self._workers:
                worker_loss, worker_acc = worker.train_finish()
                loss += worker_loss
                acc += worker_acc
        return loss, acc, self._cat_grad.norm()

    def zero_grad(self):
        """zero out the current grad"""
        self._cat_grad.zero_()
        if self._local:
            for p in self.model.parameters():
                p.grad = None

    def step(self):
        """update the model according to the current grad"""
        _cat_grad_to_model_grads(self._cat_grad, self.model)
        self.optimizer.step()

    def lr(self, new_lr):
        """
        update optimizer lr (not really related to sharing the GPU),
        just a convenience method"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

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

    def __init__(self, model, cat_grad):
        self.model = model
        self.cat_grad = cat_grad

    def set_mode(self, evaluation=False):
        """set the model evaluation mode"""
        if evaluation:
            self.model.eval()
        else:
            self.model.train()

    def train(self, examples, batch_size):
        """
        take fwd/bwd pass over the given examples,
        writing grads to cat_grad
        """
        return _train(self.model, examples, batch_size, self.cat_grad)

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

    def __init__(self, num_workers, worker_idx, model, cat_grad):
        self._worker_idx = worker_idx
        self._num_workers = num_workers
        fmt = '{: ' + str(len(str(num_workers))) + 'd}'
        self._id_str = ('worker ' + fmt + ' of ' + fmt).format(
            worker_idx, num_workers)

        ctx = mp.get_context('forkserver')
        self._conn, child_conn = ctx.Pipe()
        self._proc = ctx.Process(
            target=_child_loop,
            args=(self._conn, child_conn, self._id_str, model, cat_grad))
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

    def close(self):
        """initiate remote close"""
        # racy if, so we swallow errors.
        if self._proc.is_alive():
            self._push('close', tuple(), swallow_errors=True)

    def close_finish(self):
        """join remote worker process"""
        self._conn.close()
        self._proc.join()

    def set_mode(self, evaluation):
        """zero the gradient"""
        self._push('set_mode', (evaluation, ))

    def set_mode_finish(self):
        """wait until gradient is zeroed"""
        self._pull('set_mode')


def _train(model, examples, batch_size, cat_grad):
    agg_loss = 0
    agg_acc = 0
    model.zero_grad()
    loss_seed = torch.ones((), device=get_device()) / batch_size
    # backward pass and *.grad seem to be independent per child process
    for ex in examples:
        prepared_ex = model.prepare_example(ex)
        loss, acc = model.forward(prepared_ex)
        # faster to just compute bwd pass and drop used activations
        loss.backward(loss_seed)
        agg_loss += loss.detach().cpu().numpy()
        agg_acc += acc.detach().cpu().numpy()
    _model_grads_to_cat_grad(model, cat_grad)
    agg_loss = agg_loss / batch_size
    agg_acc = agg_acc / batch_size
    return agg_loss, agg_acc


def _diagnose(model, examples):
    diagnostics = []
    for ex in examples:
        prepared_ex = model.prepare_example(ex)
        ex_diagnostics = model.diagnose(prepared_ex)
        diagnostics.append(ex_diagnostics)
    return diagnostics


def _child_loop(parent_conn, conn, id_str, model, cat_grad):
    parent_conn.close()
    disable_contiguous_rnn_warning()
    try:
        with closing(conn):
            remote = _Remote(model, cat_grad)
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


def _param_grads(model):
    grad_idxs = []
    grad_lens = []
    grad_shapes = []
    idx = 0
    for p in model.parameters():
        size = p.numel()
        if not p.requires_grad or not size:
            grad_idxs.append(idx)
            grad_lens.append(0)
            grad_shapes.append(tuple())
            continue
        grad_idxs.append(idx)
        grad_lens.append(size)
        grad_shapes.append(p.size())
        idx += size
    return grad_idxs, grad_lens, grad_shapes


def _cat_grad_to_model_grads(cat_grad, model):
    grad_idxs, grad_lens, grad_shapes = _param_grads(model)
    with torch.no_grad():
        for grad_idx, grad_len, grad_shape, p in zip(grad_idxs, grad_lens,
                                                     grad_shapes,
                                                     model.parameters()):
            if grad_len:
                begin = grad_idx
                end = grad_idx + grad_len
                p.grad = cat_grad[begin:end].view(*grad_shape)


def _model_grads_to_cat_grad(model, cat_grad):
    grad_idxs, grad_lens, _ = _param_grads(model)
    with torch.no_grad():
        for grad_idx, grad_len, p in zip(grad_idxs, grad_lens,
                                         model.parameters()):
            if grad_len and p.grad is not None:
                begin = grad_idx
                end = grad_idx + grad_len
                cat_grad[begin:end] = p.grad.data.view(-1)


def _grad_length(model):
    _, grad_lens, _ = _param_grads(model)
    return sum(grad_lens)
