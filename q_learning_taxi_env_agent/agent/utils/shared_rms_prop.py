import torch
from torch.optim.optimizer import Optimizer

class SharedRMSprop(Optimizer):
    """Implements RMSprop algorithm with shared states.
    """
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(SharedRMSprop, self).__init__(params, defaults)

        # State initialization (and sharing)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1).share_memory_() # Global step counter for this parameter
                state['square_avg'] = torch.zeros_like(p.data).share_memory_()
                if group['momentum'] > 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data).share_memory_()
                if group['centered']:
                    state['grad_avg'] = torch.zeros_like(p.data).share_memory_()

    def share_memory(self):
        # Ensure all state tensors are in shared memory
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['square_avg'].share_memory_()
                if 'momentum_buffer' in state:
                    state['momentum_buffer'].share_memory_()
                if 'grad_avg' in state:
                    state['grad_avg'].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('SharedRMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization is done in __init__
                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1
                step_t = state['step'][0] # Get scalar value

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.add_(buf, alpha=-group['lr'])
                else:
                    p.addcdiv_(grad, avg, value=-group['lr'])

        return loss
