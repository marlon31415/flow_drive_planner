import torch
import torch.nn as nn
from timm.models.layers import Mlp
from timm.layers import DropPath
from box import ConfigBox

from flow_drive.model.module.mixer import MixerBlock
from route_language_encoder.vocabulary_encoder.encoder import VocabularyRouteEncoder
from route_language_encoder.utils.encoder_utils import (
    to_cumulative_distances,
    truncate_route_description,
)


class Encoder(nn.Module):
    def __init__(self, config: ConfigBox):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        # neighbors + static + lanes
        self.token_num = config.agent_num + config.static_objects_num + config.lane_num

        self.neighbor_encoder = AgentFusionEncoder(
            config.time_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth,
        )
        self.static_encoder = StaticFusionEncoder(
            config.static_objects_state_dim,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
        )
        self.lane_encoder = LaneFusionEncoder(
            config.lane_len,
            route_pool_level=config.route_pool_level,
            max_route_steps=config.max_route_steps,
            route_aux_loss=config.route_aux_loss,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth,
            use_spatial_attn_bias=config.use_spatial_attn_bias,
            route_fusion=getattr(config, "route_fusion", "r2m"),
            route_fusion_binary=getattr(config, "route_fusion_binary", True),
            route_token_pool=getattr(config, "route_token_pool", "cls"),
            route_cumulative_distance=getattr(
                config, "route_cumulative_distance", True
            ),
        )

        self.fusion = FusionEncoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            drop_path_rate=config.encoder_drop_path_rate,
            depth=config.encoder_depth,
        )
        # position embedding encode x, y, cos, sin, type
        self.pos_emb = nn.Linear(7, config.hidden_dim)
        # print("Number of encoder parameters: {:e}".format(sum(p.numel() for p in self.parameters())))
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)

        # Initialize embedding MLP:
        nn.init.normal_(self.neighbor_encoder.type_emb.weight, std=0.02)
        nn.init.normal_(self.lane_encoder.speed_limit_emb.weight, std=0.02)
        nn.init.normal_(self.lane_encoder.traffic_emb.weight, std=0.02)

    def forward(self, inputs):
        ego_anchor = inputs["ego_current_state"][
            ..., :4
        ]  # [B, 4], x, y, cos, sin; ego-local constant (x/y=0, cos/sin=[1,0])
        neighbors = inputs["neighbor_agents_past"]
        static = inputs["static_objects"]
        lanes = inputs["lanes"]
        lanes_speed_limit = inputs["lanes_speed_limit"]
        lanes_has_speed_limit = inputs["lanes_has_speed_limit"]
        lanes_is_route = inputs["lanes_is_route"]
        route_description = inputs["route_description"]
        route_maneuver_positions = inputs["route_maneuver_positions"]
        route_maneuver_headings = inputs.get("route_maneuver_headings")

        B = neighbors.shape[0]

        encoding_neighbors, neighbors_mask, neighbor_pos = self.neighbor_encoder(
            neighbors
        )
        encoding_static, static_mask, static_pos = self.static_encoder(static)
        encoding_lanes, lanes_mask, lane_pos, route_logits = self.lane_encoder(
            lanes,
            lanes_speed_limit,
            lanes_has_speed_limit,
            lanes_is_route,
            route_description,
            route_maneuver_positions,
            ego_anchor,
            route_maneuver_headings,
        )

        encoding_input = [encoding_neighbors, encoding_static, encoding_lanes]
        encoding_mask = [neighbors_mask, static_mask, lanes_mask]

        encoding_input = torch.cat(encoding_input, dim=1)
        encoding_mask = torch.cat(encoding_mask, dim=1).view(-1)

        # positional encoding based on position, orientation, and type of each token
        encoding_pos = torch.cat([neighbor_pos, static_pos, lane_pos], dim=1).view(
            B * self.token_num, -1
        )
        encoding_pos = self.pos_emb(encoding_pos[~encoding_mask])
        encoding_pos_result = torch.zeros(
            (B * self.token_num, self.hidden_dim), device=encoding_pos.device
        )
        encoding_pos_result[~encoding_mask] = encoding_pos  # Fill in valid parts

        encoding_input = encoding_input + encoding_pos_result.view(
            B, self.token_num, -1
        )

        # [B, num_lanes + num_static + num_neighbors, hidden_dim]
        encoder_outputs = self.fusion(
            encoding_input, encoding_mask.view(B, self.token_num)
        )
        # select only valid tokens
        valid_indices = ~encoding_mask.view(
            B, -1
        )  # [B, token_num], where valid_indices is True
        encoding_mask = encoding_mask.view(B, -1)  # [B, token_num]

        # extract only valid tokens from encoder_outputs
        empty_tokens = torch.zeros(
            (B, self.token_num, self.hidden_dim), device=encoder_outputs.device
        )
        empty_tokens[valid_indices] = encoder_outputs[
            valid_indices
        ]  # Fill in valid parts
        encoder_outputs = empty_tokens.view(
            B, -1, self.hidden_dim
        )  # [B, token_num, hidden_dim]

        return {
            "encoding": encoder_outputs,  # [B, token_num, hidden_dim]
            "mask": encoding_mask,  # [B, token_num]
            "route_logits": route_logits,  # [B, num_lanes] or None
            "lanes_is_route": lanes_is_route,  # [B, num_lanes]
            "lanes_mask": lanes_mask,  # [B, num_lanes]
        }


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=dropout,
        )

    def forward(self, x, mask):
        x = x + self.drop_path(self.attn(self.norm1(x), x, x, key_padding_mask=mask)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=dropout,
        )

    def forward(self, x, context, context_mask=None, attn_bias=None):
        """
        x:        [B, N, D]  (queries)
        context:  [B, M, D]  (keys / values)
        attn_bias: [B*H, N, M] optional additive attention bias (float)
        """
        # When both masks are supplied, PyTorch requires matching dtypes.
        # Convert bool key_padding_mask to float so it matches the float attn_bias.
        if (
            context_mask is not None
            and attn_bias is not None
            and context_mask.dtype == torch.bool
        ):
            context_mask = torch.zeros_like(
                context_mask, dtype=attn_bias.dtype
            ).masked_fill_(context_mask, float("-inf"))

        x = x + self.drop_path(
            self.attn(
                self.norm1(x),
                context,
                context,
                key_padding_mask=context_mask,
                attn_mask=attn_bias,
            )[0]
        )

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AgentFusionEncoder(nn.Module):
    def __init__(
        self,
        time_len,
        drop_path_rate=0.3,
        hidden_dim=192,
        depth=3,
        tokens_mlp_dim=64,
        channels_mlp_dim=128,
    ):
        super().__init__()

        self._hidden_dim = hidden_dim
        self._channel = channels_mlp_dim

        self.type_emb = nn.Linear(3, channels_mlp_dim)

        self.channel_pre_project = Mlp(
            in_features=8 + 1,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=time_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.blocks = nn.ModuleList(
            [
                MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x):
        """
        x: B, P, V, D (x, y, cos, sin, vx, vy, w, l, type(3))
        """
        neighbor_type = x[:, :, -1, 8:]
        x = x[..., :8]

        pos = x[:, :, -1, :7].detach().clone()  # x, y, cos, sin
        # neighbor: [1,0,0]
        pos[..., -3:] = 0.0
        pos[..., -3] = 1.0

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = torch.cat([x, (~mask_v).float().unsqueeze(-1)], dim=-1)
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1)
        x = x[valid_indices]

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        # pooling
        x = torch.mean(x, dim=1)

        neighbor_type = neighbor_type.view(B * P, -1)
        neighbor_type = neighbor_type[valid_indices]
        type_embedding = self.type_emb(neighbor_type)  # Type embedding for valid data
        x = x + type_embedding

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts

        return x_result.view(B, P, -1), mask_p.reshape(B, -1), pos.view(B, P, -1)


class StaticFusionEncoder(nn.Module):
    def __init__(self, dim, drop_path_rate=0.3, hidden_dim=192, device="cuda"):
        super().__init__()

        self._hidden_dim = hidden_dim
        self.projection = Mlp(
            in_features=dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x):
        """
        x: B, P, D (x, y, cos, sin, w, l, type(4))
        """
        B, P, _ = x.shape

        pos = x[:, :, :7].detach().clone()  # x, y, cos, sin
        # static: [0,1,0]
        pos[..., -3:] = 0.0
        pos[..., -2] = 1.0

        x_result = torch.zeros((B * P, self._hidden_dim), device=x.device)

        mask_p = torch.sum(torch.ne(x[..., :10], 0), dim=-1).to(x.device) == 0

        valid_indices = ~mask_p.view(-1)

        if valid_indices.sum() > 0:
            x = x.view(B * P, -1)
            x = x[valid_indices]
            x = self.projection(x)
            x_result[valid_indices] = x

        return x_result.view(B, P, -1), mask_p.view(B, P), pos.view(B, P, -1)


class LaneFusionEncoder(nn.Module):
    def __init__(
        self,
        lane_len,
        route_pool_level,
        max_route_steps,
        route_aux_loss,
        drop_path_rate=0.3,
        hidden_dim=192,
        depth=3,
        tokens_mlp_dim=64,
        channels_mlp_dim=128,
        use_spatial_attn_bias=False,
        route_fusion="r2m",
        route_fusion_binary=True,
        route_token_pool="cls",
        route_cumulative_distance=True,
    ):
        super().__init__()

        self.route_pool_level = route_pool_level
        self.max_route_steps = max_route_steps
        self.route_aux_loss = route_aux_loss
        self.route_fusion = route_fusion
        self.route_fusion_binary = route_fusion_binary
        self.route_cumulative_distance = route_cumulative_distance

        if self.route_fusion not in ["m2r", "r2m"]:
            raise ValueError(
                f"Unsupported route_fusion='{self.route_fusion}'. Use 'm2r' or 'r2m'."
            )

        self._lane_len = lane_len
        self._channel = channels_mlp_dim

        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
        self.traffic_emb = nn.Linear(4, channels_mlp_dim)
        self.is_route_emb = nn.Embedding(2, channels_mlp_dim)
        if self.route_pool_level in ["goal", "points"]:
            self.goal_emb = nn.Linear(2, channels_mlp_dim)

        self.channel_pre_project = Mlp(
            in_features=8,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=lane_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.blocks = nn.ModuleList(
            [
                MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

        if self.route_pool_level in ["step", "route"]:
            if self.route_fusion == "m2r":
                self.route_encoder = VocabularyRouteEncoder(
                    hidden_dim=channels_mlp_dim,
                    max_route_steps=max_route_steps,
                    token_pool=route_token_pool,
                )
                self.route_fusion_encoder = RouteFusionEncoder(
                    hidden_dim=channels_mlp_dim,
                    use_spatial_attn_bias=use_spatial_attn_bias,
                )
            else:
                # Keep route conditioning in lane channel space so it can be added to x directly.
                self.route_encoder = VocabularyRouteEncoder(
                    hidden_dim=channels_mlp_dim,
                    max_route_steps=max_route_steps,
                    token_pool=route_token_pool,
                )
                self.route_conditioner = RouteToLaneConditioner(
                    hidden_dim=channels_mlp_dim,
                    depth=depth,
                    drop_path_rate=drop_path_rate,
                    use_spatial_attn_bias=use_spatial_attn_bias,
                )

        if self.route_aux_loss:
            self.route_prediction_head = nn.Sequential(
                nn.LayerNorm(channels_mlp_dim), nn.Linear(channels_mlp_dim, 1)
            )

    def forward(
        self,
        x,
        speed_limit,
        has_speed_limit,
        lanes_is_route,
        route_description,
        route_maneuver_positions,
        ego_anchor,
        route_maneuver_headings=None,
    ):
        """
        x: B, P, V, D (x, y, x'-x, y'-y, x_left-x, y_left-y, x_right-x, y_right-y, traffic(4))
        speed_limit: B, P, 1
        has_speed_limit: B, P, 1
        """
        if self.route_pool_level in ["step", "route"]:
            # Skip the "depart" step and cap both inputs at max_route_steps, in lockstep,
            # so route_description and route_maneuver_positions stay aligned downstream.
            if self.route_cumulative_distance:
                # Before truncation: the running total has to start at the route origin
                # for the distances to mean "from the ego".
                route_description = [
                    to_cumulative_distances(rd) for rd in route_description
                ]
            route_description = [
                truncate_route_description(rd, self.max_route_steps)
                for rd in route_description
            ]
            route_maneuver_positions = route_maneuver_positions[
                :, 1 : self.max_route_steps + 1
            ]
            if route_maneuver_headings is not None:
                route_maneuver_headings = route_maneuver_headings[
                    :, 1 : self.max_route_steps + 1
                ]

        traffic = x[:, :, 0, 8:]
        x = x[..., :8]

        pos = x[:, :, int(self._lane_len / 2), :7].detach().clone()  # x, y, x'-x, y'-y
        heading = torch.atan2(pos[..., 3], pos[..., 2])
        pos[..., 2] = torch.cos(heading)
        pos[..., 3] = torch.sin(heading)
        # lane: [0,0,1]
        pos[..., -3:] = 0.0
        pos[..., -1] = 1.0

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1)
        x = x[valid_indices]

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        x = torch.mean(x, dim=1)

        # Reshape speed_limit and traffic to match flattened dimensions
        speed_limit = speed_limit.view(B * P, 1)
        has_speed_limit = has_speed_limit.view(B * P, 1)
        traffic = traffic.view(B * P, -1)
        lanes_is_route = lanes_is_route.view(B * P, 1)

        # Apply embedding directly to valid speed limit data
        has_speed_limit = has_speed_limit[valid_indices].squeeze(-1)
        speed_limit = speed_limit[valid_indices].squeeze(-1)
        speed_limit_embedding = torch.zeros(
            (speed_limit.shape[0], self._channel), device=x.device
        )

        if has_speed_limit.sum() > 0:
            speed_limit_with_limit = self.speed_limit_emb(
                speed_limit[has_speed_limit].unsqueeze(-1)
            )
            speed_limit_embedding[has_speed_limit] = speed_limit_with_limit

        if (~has_speed_limit).sum() > 0:
            speed_limit_no_limit = self.unknown_speed_emb.weight.expand(
                (~has_speed_limit).sum().item(), -1
            )
            speed_limit_embedding[~has_speed_limit] = speed_limit_no_limit

        # Process traffic lights directly for valid positions
        traffic = traffic[valid_indices]
        traffic_light_embedding = self.traffic_emb(
            traffic
        )  # Traffic light embedding for valid data

        x_base = (
            x + speed_limit_embedding + traffic_light_embedding
        )  # [B * P, channels_mlp_dim]

        # Process route information directly for valid positions
        if self.route_pool_level == "none":
            lanes_is_route = lanes_is_route[valid_indices].squeeze(-1)
            route_embedding = self.is_route_emb(lanes_is_route)
            x = x_base + route_embedding  # [B * P, channels_mlp_dim]

        if self.route_pool_level == "goal":
            goals = torch.zeros((B, 2), device=x.device)
            for i in range(B):
                last_maneuver_pos = route_maneuver_positions[
                    i, route_maneuver_positions[i].nonzero(as_tuple=True)[0][-1]
                ]
                goals[i] = last_maneuver_pos
            goals = goals.repeat_interleave(P, dim=0)[valid_indices]  # [B * P, 2]
            goal_embedding = self.goal_emb(goals)
            x = x_base + goal_embedding  # [B * P, channels_mlp_dim]

        if self.route_pool_level == "points":
            points = route_maneuver_positions[:, : self.max_route_steps]
            points_embedding = self.goal_emb(
                points
            )  # [B, max_route_steps, channels_mlp_dim]

            # Masked mean pooling over valid route steps only.
            valid_step_mask = (points.abs().sum(dim=-1) > 0).unsqueeze(-1)
            valid_step_mask = valid_step_mask.to(dtype=points_embedding.dtype)
            valid_step_count = valid_step_mask.sum(dim=1).clamp_min(1.0)
            points_embedding = (points_embedding * valid_step_mask).sum(
                dim=1
            ) / valid_step_count  # [B, channels_mlp_dim]

            points_embedding = points_embedding.repeat_interleave(P, dim=0)[
                valid_indices
            ]  # [B * P, 2]
            x = x_base + points_embedding  # [B * P, channels_mlp_dim]

        # Condition lane embeddings with route->lane cross-attention in channel space.
        if self.route_pool_level in ["step", "route"] and self.route_fusion == "r2m":
            encoder_output = self.route_encoder(
                route_description,
                route_maneuver_positions,
                pool_level=self.route_pool_level,
                route_maneuver_headings=route_maneuver_headings,
            )
            route_encoding = (
                encoder_output.x
            )  # [B, channels_mlp_dim] or [B, n_route_steps, channels_mlp_dim]
            route_encoding_mask = (
                encoder_output.steps_mask
            )  # [B, channels_mlp_dim] or None

            if route_encoding.dim() == 2:
                route_encoding = route_encoding.unsqueeze(1)

            lane_tokens = torch.zeros(
                (B, P, self._channel), device=x_base.device, dtype=x_base.dtype
            )
            valid_lane_mask = ~mask_p
            lane_tokens[valid_lane_mask] = x_base

            lane_pos_xy = pos[
                :, :, :2
            ]  # [B, P, 2] already in normalized ego-local frame
            step_pos_xy = route_maneuver_positions  # [B, max_route_steps, 2]

            lane_route_delta = self.route_conditioner(
                route=route_encoding,
                lane=lane_tokens,
                lane_mask=mask_p,
                route_mask=route_encoding_mask,
                lane_pos=lane_pos_xy,
                step_pos=step_pos_xy,
            )

            x = x_base + lane_route_delta[valid_lane_mask]

        if self.route_pool_level in ["step", "route"] and self.route_fusion == "m2r":
            encoder_output = self.route_encoder(
                route_description,
                route_maneuver_positions,
                pool_level=self.route_pool_level,
                route_maneuver_headings=route_maneuver_headings,
            )
            route_encoding = (
                encoder_output.x
            )  # [B, channels_mlp_dim] or [B, n_route_steps, channels_mlp_dim]
            route_encoding_mask = encoder_output.steps_mask  # [B, n_route_steps]

            if route_encoding.dim() == 2:
                route_encoding = route_encoding.unsqueeze(1)

            lane_tokens = torch.zeros(
                (B, P, self._channel), device=x_base.device, dtype=x_base.dtype
            )
            valid_lane_mask = ~mask_p
            lane_tokens[valid_lane_mask] = x_base

            lane_pos_xy = pos[
                :, :, :2
            ]  # [B, P, 2] already in normalized ego-local frame
            step_pos_xy = route_maneuver_positions  # [B, max_route_steps, 2]

            lane_tokens = self.route_fusion_encoder(
                lane=lane_tokens,
                route=route_encoding,
                mask=route_encoding_mask,
                lane_pos=lane_pos_xy,
                step_pos=step_pos_xy,
            )

            x = lane_tokens[valid_lane_mask]

        # Auxiliary prediction head: predict which lanes are on the route
        route_logits = None
        if self.route_aux_loss:
            route_logits_valid = self.route_prediction_head(x).squeeze(
                -1
            )  # [B * P, 1] -> [B * P]
            route_logits_valid = torch.clamp(route_logits_valid, -20, 20)

            route_logits_flat = torch.zeros(
                (B * P,),
                device=route_logits_valid.device,
                dtype=route_logits_valid.dtype,
            )
            route_logits_flat[valid_indices] = route_logits_valid  # Fill in valid parts
            route_logits = route_logits_flat.view(B, P)

            # Soft route embedding from logits: interpolate between non-route and route embeddings.
            if self.route_fusion_binary:
                route_prob = torch.sigmoid(route_logits_flat[valid_indices]).unsqueeze(
                    -1
                )
                route_embedding = (1.0 - route_prob) * self.is_route_emb.weight[
                    0
                ].unsqueeze(0) + route_prob * self.is_route_emb.weight[1].unsqueeze(0)
                x = x_base + route_embedding

        x = self.emb_project(self.norm(x))  # [B * P, hidden_dim]

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        x_result = x_result.view(B, P, -1)

        return x_result, mask_p.reshape(B, -1), pos.view(B, P, -1), route_logits


class FusionEncoder(nn.Module):
    def __init__(
        self, hidden_dim=192, num_heads=6, drop_path_rate=0.3, depth=3, device="cuda"
    ):
        super().__init__()

        dpr = drop_path_rate

        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(hidden_dim, num_heads, dropout=dpr)
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):

        mask[:, 0] = False
        for b in self.blocks:
            x = b(x, mask)

        return self.norm(x)


class SpatialBiasComputer(nn.Module):
    """Computes an additive attention bias from pairwise lane↔step relative positions."""

    def __init__(self, num_heads=6, hidden_dim=32):
        super().__init__()
        self.num_heads = num_heads
        self.proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, lane_pos, step_pos):
        """
        Args:
            lane_pos: [B, P, 2]  (lane xy positions, normalized)
            step_pos: [B, S, 2]  (route step xy positions, normalized)
        Returns:
            attn_bias: [B*H, P, S]  additive bias for nn.MultiheadAttention(attn_mask=...)
        """
        # [B, P, S, 2]
        rel = lane_pos[:, :, None, :] - step_pos[:, None, :, :]
        # [B, P, S, num_heads]
        bias = self.proj(rel)
        # -> [B, num_heads, P, S] -> [B*H, P, S]
        B, P, S, H = bias.shape
        bias = bias.permute(0, 3, 1, 2).reshape(B * H, P, S)
        return bias


class RouteFusionEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim=192,
        num_heads=6,
        drop_path_rate=0.3,
        depth=3,
        device="cuda",
        use_spatial_attn_bias=False,
    ):
        super().__init__()

        # Ensure MultiheadAttention is valid for the chosen feature width.
        if hidden_dim % num_heads != 0:
            for candidate in range(min(num_heads, hidden_dim), 0, -1):
                if hidden_dim % candidate == 0:
                    num_heads = candidate
                    break

        self.route_to_lane_attn = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=hidden_dim, heads=num_heads, dropout=drop_path_rate
                )
                for _ in range(depth)
            ]
        )

        self.use_spatial_attn_bias = use_spatial_attn_bias
        if use_spatial_attn_bias:
            self.spatial_bias = SpatialBiasComputer(num_heads=num_heads)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, lane, route, mask=None, lane_pos=None, step_pos=None):
        """
        Forward pass for the encoder module.

        Args:
            lane: Tensor of shape (batch_size, num_lanes, feature_dim) representing lane features.
            route: Tensor of shape (batch_size, 1, feature_dim) or (batch_size, num_steps, feature_dim) representing route features.
            mask: Tensor of shape (batch_size, num_steps) representing padding mask for route attention,
                  where True indicates positions to be masked out.
            lane_pos: Tensor of shape (batch_size, num_lanes, 2) — lane xy positions (normalized).
            step_pos: Tensor of shape (batch_size, num_steps, 2) — route step xy positions (normalized).

        Returns:
            Tensor of shape (batch_size, num_lanes, feature_dim) representing normalized lane features
            after cross-attention with route features.
        """
        attn_bias = None
        if self.use_spatial_attn_bias and lane_pos is not None and step_pos is not None:
            attn_bias = self.spatial_bias(lane_pos, step_pos)  # [B*H, P, S]

        for attn_block in self.route_to_lane_attn:
            lane = attn_block(lane, route, mask, attn_bias=attn_bias)

        return self.norm(lane)


class RouteToLaneConditioner(nn.Module):
    """Route queries attend to lane keys/values, then pool route context back to a lane delta."""

    def __init__(
        self,
        hidden_dim=128,
        num_heads=6,
        drop_path_rate=0.3,
        depth=3,
        use_spatial_attn_bias=False,
    ):
        super().__init__()

        # Ensure MultiheadAttention is always valid for the chosen channel width.
        if hidden_dim % num_heads != 0:
            for candidate in range(min(num_heads, hidden_dim), 0, -1):
                if hidden_dim % candidate == 0:
                    num_heads = candidate
                    break

        self.route_to_lane_attn = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=hidden_dim, heads=num_heads, dropout=drop_path_rate
                )
                for _ in range(depth)
            ]
        )

        self.use_spatial_attn_bias = use_spatial_attn_bias
        if use_spatial_attn_bias:
            self.spatial_bias = SpatialBiasComputer(num_heads=num_heads)

        self.route_out = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        )
        # Start close to zero contribution and let optimization learn influence.
        self.route_gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, route, lane, lane_mask=None, route_mask=None, lane_pos=None, step_pos=None
    ):
        """
        route: [B, S, C] queries
        lane: [B, P, C] keys/values
        lane_mask: [B, P], True means masked lane
        route_mask: [B, S], True means masked route step
        returns: [B, P, C] lane additive delta
        """
        attn_bias = None
        if (
            self.use_spatial_attn_bias
            and lane_pos is not None
            and step_pos is not None
            and route.shape[1] == step_pos.shape[1]
        ):
            # SpatialBiasComputer returns [B*H, P, S], transpose to [B*H, S, P] for route-query attention.
            attn_bias = self.spatial_bias(lane_pos, step_pos).transpose(1, 2)

        for attn_block in self.route_to_lane_attn:
            route = attn_block(route, lane, context_mask=lane_mask, attn_bias=attn_bias)

        if route_mask is not None:
            valid = (~route_mask).unsqueeze(-1).to(route.dtype)
            denom = valid.sum(dim=1).clamp_min(1.0)
            route_global = (route * valid).sum(dim=1) / denom
        else:
            route_global = route.mean(dim=1)

        route_global = self.route_out(route_global)
        gate = torch.sigmoid(self.route_gate)
        return gate * route_global.unsqueeze(1).expand_as(lane)
