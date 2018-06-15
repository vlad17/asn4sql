from contextlib import closing
import sys

import torch
from torch import optim
from .utils import get_device

def _train(model, examples, batch_size):
    agg_loss = torch.zeros((), device=get_device())
    agg_acc = 0
    loss_seed = torch.ones((), device=get_device()) / batch_size
    for ex in examples:
        prepared_ex = model.prepare_example(ex)
        loss, acc = model.forward(prepared_ex)
        agg_loss += loss
        agg_acc += acc.detach().cpu().numpy()
    agg_loss /= batch_size
    agg_loss.backward()
    grad = torch.cat(
        tuple(p.grad.data.view(-1) for p in model.parameters()))
    gradnorm = torch.norm(grad)
    print(gradnorm)
    agg_loss = agg_loss.detach().cpu().numpy()
    agg_acc = agg_acc / batch_size
    return agg_loss, agg_acc


def _child_loop(parent_conn, conn, id_str, model):
    parent_conn.close()
    opt = optim.SGD(model.parameters(), lr=0.1)
    try:
        with closing(conn):
            while True:
                method_name, args = conn.recv()
                if method_name == 'close':
                    return
                elif method_name == 'opt':
                    opt.step()
                    conn.send(('opt', None))
                else:
                    opt.zero_grad()
                    assert method_name == 'train'
                    examples, batch_size = args
                    loss, acc = _train(model, examples, batch_size)
                    try:
                        conn.send(('train', (loss, acc)))
                    except IOError:
                        print('{} swallowing IOError\n'.format(id_str),
                              file=sys.stderr, end='')
                        sys.stderr.flush()
    except KeyboardInterrupt:
        print('{} exited cleanly on SIGINT\n'.format(id_str), end='',
              file=sys.stderr)
        sys.stderr.flush()
