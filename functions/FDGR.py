import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os
import numpy as np
from datasets import inverse_data_transform, data_transform
import torch.fft as fft

class_num = 951


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
    

def fdgr_diffusion(x, model, b, A_funcs, y, sigma_y, cls_fn=None, classes=None, config=None, args=None):
    with torch.no_grad():
        skip = config.diffusion.num_diffusion_timesteps // config.sampling.T_sampling
        n = x.size(0)
        x0_preds = []
        xs = [x]
        
        times = get_schedule_jump(config.sampling.T_sampling, 1, 1)
        time_pairs = list(zip(times[:-1], times[1:]))
        
        for i, j in tqdm(time_pairs):
            i, j = i * skip, j * skip
            if j < 0: j = -1
            if j < i:
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                beta_t = 1 - at / at_next  # beta_t
                xt = xs[-1].to('cuda')
                if cls_fn == None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda")) * class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)
                if et.size(1) == 6:
                    et = et[:, :3]

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                if sigma_y == 0.:
                    delta_t = 0
                    weight_noise_t = 1
                else:
                    delta_t = (at_next) ** args.gamma
                    weight_noise_t = delta_t
                eta_reg = max(1e-4, sigma_y ** 2 * args.eta_tilde)
                if args.eta_tilde < 0:
                    eta_reg = 1e-4 + args.xi * (sigma_y * 255.0) ** 2
                scale_gLS = args.scale_ls
                guidance_BP = A_funcs.A_pinv_add_eta(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1), eta_reg).reshape(*x0_t.size())
                guidance_LS = A_funcs.At(A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)).reshape(
                    *x0_t.size())

                if args.step_size_mode == 0:

                    step_size_LS = 1
                    step_size_BP = 1

                    if args.mode == 'G+':
                        step_size_reg = beta_t * (1 - at_next) / (1 - at) 
                        
                    elif args.mode == 'G-':
                        step_size_reg = 1 
        
                    step_size = 1

                elif args.step_size_mode == 1:

                    step_size_LS = (1 - at_next) / (1 - at)
                    step_size_BP = (1 - at_next) / (1 - at)

                    step_size_reg = 1 

                    step_size = 1
                    
                elif args.step_size_mode == 2:

                    step_size_LS = (1 - at_next) / (1 - at)
                    step_size_BP = 1
                    step_size = 1
                else:
                    assert 1, "unsupported step-size mode"

                if args.mode not in ['G+', 'G-']:
                    raise ValueError("Invalid mode: must be 'G+' or 'G-'")
                grad_reg = torch.zeros_like(x0_t) 

                if args.mode == 'G+':
                    grad_x, grad_y = backward_difference_color(x0_t)
                    epsilon = 1e-6
                    alpha = args.alpha
                    sigma_g = args.sigma_g
                    lam = args.lam 

                    grad_x_stab = grad_x ** 2 + epsilon
                    grad_y_stab = grad_y ** 2 + epsilon

                    phi_prime_x = alpha * (grad_x_stab ** (alpha /2 - 1) / (2 * sigma_g**2)) * torch.exp(-(grad_x_stab ** (alpha/2)) / (2 * sigma_g**2)) * grad_x
                    phi_prime_y = alpha * (grad_y_stab ** (alpha /2 - 1) / (2 * sigma_g**2)) * torch.exp(-(grad_y_stab ** (alpha/2)) / (2 * sigma_g**2)) * grad_y
                    
                    grad_reg_x = forward_difference_x(phi_prime_x)
                    grad_reg_y = forward_difference_y(phi_prime_y)
                    grad_reg = step_size_reg * (lam * grad_reg_x + lam * grad_reg_y)

                elif args.mode == 'G-': 

                    batch_size, channels, height, width = x0_t.shape

                    alpha = args.alpha
                    sigma_g = args.sigma_g 
                    lam = args.lam 
                    epsilon = 1e-4

                    F = fft.fft2(x0_t)

                    u = torch.fft.fftfreq(width, device=x0_t.device).view(1, 1, 1, width)
                    v = torch.fft.fftfreq(height, device=x0_t.device).view(1, 1, height, 1)

                    grad_u_F = F * (2 * torch.pi * 1j * u.expand_as(F))
                    grad_v_F = F * (2 * torch.pi * 1j * v.expand_as(F))
                    grad_u = fft.ifft2(grad_u_F).real
                    grad_v = fft.ifft2(grad_v_F).real

                    grad_u_sq = grad_u ** 2 + epsilon
                    grad_v_sq = grad_v ** 2 + epsilon
                    grad_magnitude_sq = grad_u_sq + grad_v_sq
                    grad_magnitude_sq_clamp = torch.clamp(grad_magnitude_sq, min=epsilon, max=100.0)

                    phi_prime_u = alpha * (grad_magnitude_sq_clamp ** (alpha / 2 - 1) / (2 * (sigma_g ** 2))) * \
                                  torch.exp(- (grad_magnitude_sq_clamp ** (alpha / 2)) / (2 * (sigma_g ** 2))) * grad_u
                    phi_prime_v = alpha * (grad_magnitude_sq_clamp ** (alpha / 2 - 1) / (2 * (sigma_g ** 2))) * \
                                  torch.exp(- (grad_magnitude_sq_clamp ** (alpha / 2)) / (2 * (sigma_g ** 2))) * grad_v

                    phi_prime_u_F = fft.fft2(phi_prime_u)
                    phi_prime_v_F = fft.fft2(phi_prime_v)

                    div_u_F = phi_prime_u_F * (-2 * torch.pi * 1j * u.expand_as(F))
                    div_v_F = phi_prime_v_F * (-2 * torch.pi * 1j * v.expand_as(F))
                    divergence_F = div_u_F + div_v_F
                    grad_reg_component = fft.ifft2(divergence_F).real
                    grad_reg = step_size_reg * lam * grad_reg_component

                xt_next_tilde = x0_t - step_size * (step_size_BP * (
                        1 - delta_t) * guidance_BP + step_size_LS * delta_t * scale_gLS * guidance_LS + grad_reg)
                
                et_hat = (xt - at.sqrt() * xt_next_tilde) / (1 - at).sqrt()
                c1 = 0
                c2 = 0
                if args.inject_noise:
                    zeta = args.zeta
                    c1 = (1 - at_next).sqrt() * np.sqrt(zeta)
                    c2 = (1 - at_next).sqrt() * np.sqrt(1 - zeta) * weight_noise_t
                xt_next = at_next.sqrt() * xt_next_tilde + c1 * torch.randn_like(x0_t) + c2 * et_hat
                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
            else:
                assert 1, "Unexpected case"
        if sigma_y != 0.:
            xs.append(x0_t.to('cpu'))
    return [xs[-1]], [x0_preds[-1]]


def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1
    t = T_sampling
    ts = []
    while t >= 1:
        t = t - 1
        ts.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)
    ts.append(-1)
    _check_times(ts, -1, T_sampling)
    return ts


def _check_times(times, t_0, T_sampling):
    assert times[0] > times[1], (times[0], times[1])
    assert times[-1] == -1, times[-1]
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)


def forward_difference_x(grad_x):
    batch_size, channels, height, width = grad_x.shape

    grad_x_diff = torch.zeros_like(grad_x)

    grad_x_diff[:, :, :, :-1] = grad_x[:, :, :, :-1] - grad_x[:, :, :, 1:]  
    grad_x_diff[:, :, :, -1] = grad_x[:, :, :, -1] - grad_x[:, :, :, 0] 

    return grad_x_diff


def forward_difference_y(grad_y):
    batch_size, channels, height, width = grad_y.shape

    grad_y_diff = torch.zeros_like(grad_y)

    grad_y_diff[:, :, :-1, :] = grad_y[:, :, :-1, :] - grad_y[:, :, 1:, :] 
    grad_y_diff[:, :, -1, :] = grad_y[:, :, -1, :] - grad_y[:, :, 0, :]

    return grad_y_diff


def backward_difference_color(x0_t):
    batch_size, channels, height, width = x0_t.shape

    grad_x = torch.zeros_like(x0_t)
    grad_y = torch.zeros_like(x0_t)

    grad_x[:, :, :, 1:] = x0_t[:, :, :, 1:] - x0_t[:, :, :, :-1] 
    grad_x[:, :, :, 0] = x0_t[:, :, :, 0] - x0_t[:, :, :, -1]

    grad_y[:, :, 1:, :] = x0_t[:, :, 1:, :] - x0_t[:, :, :-1, :]  
    grad_y[:, :, 0, :] = x0_t[:, :, 0, :] - x0_t[:, :, -1, :] 

    return grad_x, grad_y


def calculate_total_gradient(grad_x, grad_y):
    return torch.sqrt(torch.sum(grad_x ** 2, dim=1) + torch.sum(grad_y ** 2, dim=1))


def compute_gradients(x0_t):
    batch_size, channels, height, width = x0_t.shape

    grad_x = torch.zeros_like(x0_t)
    grad_y = torch.zeros_like(x0_t)

    grad_x[:, :, :, 1:] = x0_t[:, :, :, 1:] - x0_t[:, :, :, :-1] 
    grad_x[:, :, :, 0] = x0_t[:, :, :, 0] - x0_t[:, :, :, -1]

    grad_y[:, :, 1:, :] = x0_t[:, :, 1:, :] - x0_t[:, :, :-1, :] 
    grad_y[:, :, 0, :] = x0_t[:, :, 0, :] - x0_t[:, :, -1, :]  

    return grad_x, grad_y
