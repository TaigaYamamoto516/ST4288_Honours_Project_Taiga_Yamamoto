import torch
import torch.nn.functional as F
from torch.optim import Optimizer
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def nadam(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps,
         m_schedules,
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float,
         scheduled_decay: float):

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        m_schedule = m_schedules[i]

        bias_correction_2 = 1 - beta2 ** step

        momentum_cache_t = beta1 * (1. - 0.5 * (0.96 ** (step * scheduled_decay)))
        momentum_cache_t_1 = beta1 * (1. - 0.5 * (0.96 ** ((step + 1) * scheduled_decay)))
        
        m_schedule *= momentum_cache_t
        m_schedule_next = m_schedule * momentum_cache_t_1
        m_schedules[i] = m_schedule

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        denom = exp_avg_sq.div(bias_correction_2).sqrt()

        if amsgrad: #Use amsgrad==False (not modified for NAdam)
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt()).add_(eps)
        else:
            denom.add_(eps)
            param.addcdiv_(grad, denom, value=(-lr * (1. - momentum_cache_t) / (1. - m_schedule)))
            param.addcdiv_(exp_avg, denom, value=(-lr * momentum_cache_t_1) / (1. - m_schedule_next))


class NAdam_IKSA_Min(Optimizer):

    def __init__(self, params, function, lr=0.0002, betas=(0.1, 0.99), eps=1e-8,
                 weight_decay=0, scheduled_decay=0.004, amsgrad=False, eps_iksa = 1):

        def f_def(x):
            return x
          
        defaults = dict( function = function, lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, eps_iksa = eps_iksa, scheduled_decay=scheduled_decay)

        super(NAdam_IKSA_Min, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NAdam_IKSA_Min, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, c, running_loss, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            m_schedules = []
            beta1, beta2 = group['betas']
            eps_iksa = group['eps_iksa']
            func = group["function"]

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                    grad = p.grad
                    new_grad = grad/(func(torch.max(torch.zeros(p.data.size(), device = device), running_loss - c)) + eps_iksa)
                    grads.append(new_grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['m_schedule'] = 1
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                            
                    m_schedules.append(state['m_schedule'])
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            nadam(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   amsgrad=group['amsgrad'],
                   beta1=beta1,
                   beta2=beta2,
                   m_schedules=m_schedules,
                   lr=group['lr'],
                   weight_decay=group['weight_decay'],
                   eps=group['eps'],
                   scheduled_decay=group['scheduled_decay'])
        return loss

class NAdam_IKSA_Max(Optimizer):

    def __init__(self, params, function, lr=0.0002, betas=(0.1, 0.99), eps=1e-8,
                 weight_decay=0, scheduled_decay=0.004, amsgrad=False, eps_iksa = 1):

        def f_def(x):
            return x

        defaults = dict( function = function, lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, eps_iksa = eps_iksa, scheduled_decay=scheduled_decay)

        super(NAdam_IKSA_Max, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NAdam_IKSA_Max, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, c, running_loss, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            m_schedules = []
            beta1, beta2 = group['betas']
            eps_iksa = group['eps_iksa']
            func = group["function"]

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                    grad = p.grad
                    new_grad = grad/(func(torch.max(torch.zeros(p.data.size(), device = device), c - running_loss)) + eps_iksa)
                    grads.append(new_grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['m_schedule'] = 1
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    m_schedules.append(state['m_schedule'])
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            nadam(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   amsgrad=group['amsgrad'],
                   beta1=beta1,
                   beta2=beta2,
                   m_schedules=m_schedules,
                   lr=group['lr'],
                   weight_decay=group['weight_decay'],
                   eps=group['eps'],
                   scheduled_decay=group['scheduled_decay'])
        return loss