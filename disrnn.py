import torch
import torch.distributions as dist
from torch import nn, Tensor

class MLP(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_size: list[int],
            act: nn.Module = nn.ReLU
    ) -> None:
        super(MLP, self).__init__()

        layers = []
        _prev_dim = input_size
        for hidden in hidden_size:
            layers.append(nn.Linear(_prev_dim, hidden))
            layers.append(act())
            _prev_dim = hidden
        layers.append(nn.Linear(_prev_dim, output_size))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        output = self.mlp(x)
        return output
    

class Bottleneck(nn.Module):
    def __init__(self) -> None:
        super(Bottleneck, self).__init__()
        
    def compute_kld(self, mean: Tensor, log_var: Tensor) -> Tensor:
        return 0.5 * torch.sum(-log_var - 1 + (mean ** 2) + log_var.exp(), dim=-1)
    
    def forward(self, x: Tensor, log_var: Tensor, multiplier: Tensor) -> Tensor:
        std = log_var.exp().sqrt()
        mean = multiplier * x
        kld = self.compute_kld(mean, log_var)
        x_tilde = torch.randn_like(x) * std + mean
        return x_tilde, kld


class UpdateMLPs(nn.Module):
    def __init__(
            self,
            num_latents: int,
            input_size: int,
            hidden_size: list[int]
    ) -> None:
        super(UpdateMLPs, self).__init__()

        self.update_mlps = nn.ModuleList([MLP(input_size, 2, hidden_size) for mlp in range(num_latents)])

    def forward(self, x: Tensor, z_t: Tensor) -> Tensor:

        B, D, Z = x.size()
        new_latents = torch.zeros(size=(B, Z))
        for mlp_idx, mlp in enumerate(self.update_mlps):
            mlp_output = mlp(x[..., mlp_idx])
            u, w = mlp_output[:, 0], mlp_output[:, 1].sigmoid()
            z = z_t[:, mlp_idx]
            new_latents[:, mlp_idx] = (1 - w) * z + u * w
        
        return new_latents


class DisRNN(nn.Module):
    def __init__(
            self,
            num_latents: int,
            num_obs: int,
            update_mlp_hidden_size: list[int],
            choice_mlp_hidden_size: list[int]
    ) -> None:
        super(DisRNN, self).__init__()

        self.update_mlp_input_size = num_latents + num_obs
        self.num_latents = num_latents

        self.z0 = nn.Parameter(torch.randn(size=(num_latents,)))
        self.update_bottleneck_multiplier = nn.Parameter(torch.ones(size=(self.update_mlp_input_size, num_latents)))
        self.update_bottleneck_log_var = nn.Parameter(dist.Uniform(-3, -2).sample((self.update_mlp_input_size, num_latents)))
        self.global_bottleneck_log_var = nn.Parameter(dist.Uniform(-3, -2).sample((num_latents,)))
        
        self.global_bottleneck = Bottleneck()
        self.update_bottleneck = nn.ModuleList([Bottleneck() for latent in range(num_latents)])

        self.update_mlp = UpdateMLPs(num_latents, self.update_mlp_input_size, update_mlp_hidden_size)
        self.choice_mlp = MLP(num_latents, 2, choice_mlp_hidden_size)

    
    def forward(
            self,
            latents: Tensor,
            obs: Tensor,
            t0: bool = False,
            eval_mode: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        
        # Prepare model input
        B, _ = latents.size()
        if t0:
            latents = torch.expand_copy(self.z0, (B, self.num_latents))
            x = torch.cat((latents, obs), dim=-1)

            update_mlp_inputs = torch.zeros(size=(B, self.update_mlp_input_size, self.num_latents))
            update_kld = torch.zeros(size=(B, self.num_latents))

            # Put inputs through bottlenecks
            for idx, update_bottleneck in enumerate(self.update_bottleneck):
                x_tilde, kld = update_bottleneck(
                    x,
                    self.update_bottleneck_log_var[:, idx] * (1 - eval_mode),
                    self.update_bottleneck_multiplier[:, idx]
                )

                update_mlp_inputs[..., idx] = x_tilde
                update_kld[:, idx] = kld

            # Update corrupted latents
            new_latents = self.update_mlp(update_mlp_inputs, latents)

            # Global bottleneck (no multiplier used on global bottleneck)
            z_tilde, global_kld = self.global_bottleneck(
                new_latents,
                self.global_bottleneck_log_var * (1 - eval_mode),
                1
            )

            y = self.choice_mlp(z_tilde)

            return y, z_tilde, global_kld + update_kld.sum(-1)
        
    @property
    def update_sigmas(self):
        return self.update_bottleneck_log_var.exp().clone().detach()
    
    @property
    def update_multipliers(self):
        return self.update_bottleneck_multiplier.clone().detach()
    
    @property
    def global_sigmas(self):
        return self.global_bottleneck_log_var.exp().clone().detach()

