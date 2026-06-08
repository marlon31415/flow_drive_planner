import torch
import torch.nn as nn

from box import ConfigBox


def compute_batch_loss(
    params: ConfigBox,
    scene_encoder,
    noise_pred_net,
    noise_scheduler,
    inputs,
    ego_future,
    device,
):
    B = ego_future.shape[0]
    obs_cond = scene_encoder(inputs)

    noise = torch.randn(ego_future.shape, device=device)  # [B, T, 3 or 4]

    ego_current_state = inputs["ego_current_state"][
        ..., :4
    ]  # [B, 10] -> [B, 4], x, y, cos, sin

    t_idx = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,))

    timesteps_float = noise_scheduler.timesteps[t_idx].to(device)

    xt = noise_scheduler.scale_noise(ego_future, timesteps_float, noise)

    xt = torch.cat([ego_current_state.unsqueeze(1), xt], dim=1)

    t_idx = t_idx.to(device)
    v_pred = noise_pred_net(xt, t_idx, global_cond=obs_cond)
    v_pred = v_pred[:, 1:]

    target_v = noise - ego_future
    diffusion_loss = nn.functional.mse_loss(v_pred, target_v)

    # Auxiliary route prediction loss
    route_aux_loss = torch.tensor(0.0, device=device)
    route_logits = obs_cond.get("route_logits")
    if route_logits is not None:
        lanes_is_route = obs_cond["lanes_is_route"].squeeze(-1)  # [B, num_lanes]
        lanes_mask = obs_cond["lanes_mask"]  # [B, num_lanes]
        valid = ~lanes_mask  # True = valid lane
        valid_f = valid.to(dtype=route_logits.dtype)
        bce = nn.functional.binary_cross_entropy_with_logits(
            route_logits, lanes_is_route.float(), reduction="none"
        )

        num_valid = valid_f.sum().clamp_min(1.0)
        route_aux_loss = (bce * valid_f).sum() / num_valid

    lambda_route = 0.01
    loss = diffusion_loss + lambda_route * route_aux_loss
    return (
        loss,
        obs_cond,
        {"diffusion_loss": diffusion_loss, "route_aux_loss": route_aux_loss},
    )
