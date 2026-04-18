from __future__ import annotations

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell that performs convolutions instead of matrix multiplications."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        cur_state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(combined_conv, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)


class ConvLSTMForecaster(nn.Module):
    """ConvLSTM-based forecaster that preserves spatial structure."""

    def __init__(
        self,
        *,
        input_steps: int,
        in_channels: int,
        out_channels: int,
        spatial_shape: tuple[int, int],
        output_steps: int,
        input_channel_mean: torch.Tensor,
        input_channel_std: torch.Tensor,
        target_channel_mean: torch.Tensor,
        target_channel_std: torch.Tensor,
        target_residual_input_indices: torch.Tensor | None = None,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.0,
        autoregressive_decoder: bool = False,
        residual_encoder: bool = False,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        if lstm_hidden_size < 1:
            raise ValueError("lstm_hidden_size must be at least 1.")
        if lstm_layers < 1:
            raise ValueError("lstm_layers must be at least 1.")
        if in_channels < 1 or out_channels < 1:
            raise ValueError("Channels must be at least 1.")

        if input_channel_mean.ndim != 1 or input_channel_mean.numel() != in_channels:
            raise ValueError("input_channel_mean must be a 1D tensor matching the number of input channels.")
        if input_channel_std.ndim != 1 or input_channel_std.numel() != in_channels:
            raise ValueError("input_channel_std must be a 1D tensor matching the number of input channels.")
        if target_channel_mean.ndim != 1 or target_channel_mean.numel() != out_channels:
            raise ValueError("target_channel_mean must be a 1D tensor matching the number of target channels.")
        if target_channel_std.ndim != 1 or target_channel_std.numel() != out_channels:
            raise ValueError("target_channel_std must be a 1D tensor matching the number of target channels.")

        if target_residual_input_indices is None:
            target_residual_input_indices = torch.full((out_channels,), -1, dtype=torch.int64)
        if target_residual_input_indices.ndim != 1 or target_residual_input_indices.numel() != out_channels:
            raise ValueError("target_residual_input_indices must be a 1D tensor matching the number of target channels.")

        self.input_steps = input_steps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_shape = spatial_shape
        self.output_steps = output_steps
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.autoregressive_decoder = autoregressive_decoder
        self.residual_encoder = residual_encoder

        self.register_buffer("input_channel_mean", input_channel_mean.to(dtype=torch.float32))
        self.register_buffer("input_channel_std", input_channel_std.to(dtype=torch.float32))
        self.register_buffer("target_channel_mean", target_channel_mean.to(dtype=torch.float32))
        self.register_buffer("target_channel_std", target_channel_std.to(dtype=torch.float32))
        self.register_buffer("target_residual_input_indices", target_residual_input_indices.to(dtype=torch.int64))

        self.encoder_cells = nn.ModuleList(
            [ConvLSTMCell(in_channels if i == 0 else lstm_hidden_size, lstm_hidden_size, kernel_size) for i in range(lstm_layers)]
        )

        decoder_in_channels = out_channels
        self.decoder_cells = nn.ModuleList(
            [ConvLSTMCell(decoder_in_channels if i == 0 else lstm_hidden_size, lstm_hidden_size, kernel_size) for i in range(lstm_layers)]
        ) if autoregressive_decoder else None

        self.output_conv = nn.Conv2d(lstm_hidden_size, out_channels, kernel_size=1)
        # Learned residual skip from the most recent input frame.
        # Zero init keeps behavior unchanged until training learns a useful correction.
        if self.residual_encoder:
            self.residual_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            nn.init.zeros_(self.residual_projection.weight)
            if self.residual_projection.bias is not None:
                nn.init.zeros_(self.residual_projection.bias)
        else:
            self.residual_projection = None

    def _normalize_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = inputs.shape
        inputs_reshaped = inputs.reshape(batch * steps, channels, height, width)
        mean = self.input_channel_mean.view(1, channels, 1, 1)
        std = self.input_channel_std.view(1, channels, 1, 1)
        normalized = (inputs_reshaped - mean) / std
        return normalized.reshape(batch, steps, channels, height, width)

    def _denormalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = targets.shape
        targets_reshaped = targets.reshape(batch * steps, channels, height, width)
        mean = self.target_channel_mean.view(1, channels, 1, 1)
        std = self.target_channel_std.view(1, channels, 1, 1)
        denormalized = targets_reshaped * std + mean
        return denormalized.reshape(batch, steps, channels, height, width)

    def _remove_residual_baseline(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        residual_mask = self.target_residual_input_indices >= 0
        if not torch.any(residual_mask):
            return targets

        outputs = targets.clone()
        baseline_inputs = inputs[:, -1]
        residual_target_indices = torch.nonzero(residual_mask, as_tuple=False).flatten()
        for target_index in residual_target_indices.tolist():
            input_index = int(self.target_residual_input_indices[target_index].item())
            baseline = baseline_inputs[:, input_index:input_index+1, :, :].unsqueeze(1)
            outputs[:, :, target_index:target_index+1, :, :] = outputs[:, :, target_index:target_index+1, :, :] - baseline
        return outputs

    def _add_residual_baseline(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        residual_mask = self.target_residual_input_indices >= 0
        if not torch.any(residual_mask):
            return targets

        outputs = targets.clone()
        baseline_inputs = inputs[:, -1]
        residual_target_indices = torch.nonzero(residual_mask, as_tuple=False).flatten()
        for target_index in residual_target_indices.tolist():
            input_index = int(self.target_residual_input_indices[target_index].item())
            baseline = baseline_inputs[:, input_index:input_index+1, :, :].unsqueeze(1)
            outputs[:, :, target_index:target_index+1, :, :] = outputs[:, :, target_index:target_index+1, :, :] + baseline
        return outputs

    def _encode(self, inputs: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        batch_size, steps, channels, height, width = inputs.shape
        h = [torch.zeros(batch_size, self.lstm_hidden_size, height, width, device=inputs.device, dtype=inputs.dtype) for _ in range(self.lstm_layers)]
        c = [torch.zeros(batch_size, self.lstm_hidden_size, height, width, device=inputs.device, dtype=inputs.dtype) for _ in range(self.lstm_layers)]

        output_sequence = []
        for t in range(steps):
            x = inputs[:, t, :, :, :]
            for layer_idx in range(self.lstm_layers):
                h[layer_idx], (h[layer_idx], c[layer_idx]) = self.encoder_cells[layer_idx](x, (h[layer_idx], c[layer_idx]))
                x = h[layer_idx]
            output_sequence.append(h[-1])

        return output_sequence, [h, c]

    def _decode_autoregressive(
        self,
        inputs: torch.Tensor,
        hidden_state: list[list[torch.Tensor]],
        teacher_forcing_targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        batch_size, _, height, width = hidden_state[0][0].shape
        h = list(hidden_state[0])
        c = list(hidden_state[1])

        # Pre-compute normalized, residual-removed teacher forcing sequence
        teacher_forcing_sequence = None
        if teacher_forcing_targets is not None:
            # Remove residual baseline, then normalize into the same space the decoder predicts
            adjusted = self._remove_residual_baseline(inputs, teacher_forcing_targets)
            batch, steps, channels, h_size, w_size = adjusted.shape
            flat = adjusted.reshape(batch * steps, channels, h_size, w_size)
            mean = self.target_channel_mean.view(1, channels, 1, 1)
            std = self.target_channel_std.view(1, channels, 1, 1)
            teacher_forcing_sequence = ((flat - mean) / std).reshape(batch, steps, channels, h_size, w_size)

        device = h[0].device
        dtype = h[0].dtype
        predictions = []
        decoder_input = torch.zeros(batch_size, self.out_channels, height, width, device=device, dtype=dtype)

        for step in range(self.output_steps):
            x = decoder_input
            for layer_idx in range(self.lstm_layers):
                h[layer_idx], (h[layer_idx], c[layer_idx]) = self.decoder_cells[layer_idx](x, (h[layer_idx], c[layer_idx]))
                x = h[layer_idx]

            pred = self.output_conv(x)
            predictions.append(pred)

            # Determine next decoder input (in normalized residual space)
            if teacher_forcing_sequence is not None and step + 1 < self.output_steps:
                teacher_frame = teacher_forcing_sequence[:, step, :, :, :]
                if teacher_forcing_ratio >= 1.0:
                    decoder_input = teacher_frame
                elif teacher_forcing_ratio <= 0.0:
                    decoder_input = pred.detach()
                else:
                    use_teacher = torch.rand(batch_size, 1, 1, 1, device=device) < teacher_forcing_ratio
                    decoder_input = torch.where(use_teacher, teacher_frame, pred.detach())
            else:
                decoder_input = pred.detach()

        return torch.stack(predictions, dim=1)

    def forward(
        self,
        inputs: torch.Tensor,
        teacher_forcing_targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError("ConvLSTMForecaster expects [batch, input_steps, channels, height, width] inputs.")

        batch, steps, channels, height, width = inputs.shape
        if steps != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} input steps, received {steps}.")
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, received {channels}.")
        if (height, width) != self.spatial_shape:
            raise ValueError(f"Expected spatial shape {self.spatial_shape}, received {(height, width)}.")

        normalized_inputs = self._normalize_inputs(inputs)
        # Remove the per-channel persistence baseline from input before encoding
        # inputs[:, -1, ssh_channel] is subtracted from the normalized ssh inputs
        # so the encoder sees residual corrections, not raw absolute values
        residual_mask = self.target_residual_input_indices >= 0
        if torch.any(residual_mask):
            for target_index in torch.nonzero(residual_mask, as_tuple=False).flatten().tolist():
                input_index = int(self.target_residual_input_indices[target_index].item())
                baseline = inputs[:, -1, input_index:input_index+1, :, :]
                normalized_baseline = (baseline - self.input_channel_mean[input_index]) / self.input_channel_std[input_index]
                normalized_inputs = normalized_inputs.clone()
                normalized_inputs[:, :, input_index:input_index+1, :, :] = (
                    normalized_inputs[:, :, input_index:input_index+1, :, :] - normalized_baseline.unsqueeze(1)
                )

        encoder_outputs, hidden_state = self._encode(normalized_inputs)

        if self.autoregressive_decoder:
            predicted_normalized = self._decode_autoregressive(inputs, hidden_state, teacher_forcing_targets, teacher_forcing_ratio)
        else:
            last_hidden = encoder_outputs[-1]
            predicted_normalized = self.output_conv(last_hidden).unsqueeze(1).expand(-1, self.output_steps, -1, -1, -1)

        denormalized = self._denormalize_targets(predicted_normalized)
        outputs = self._add_residual_baseline(inputs, denormalized)

        if self.residual_projection is not None:
            last_input = inputs[:, -1, :, :, :]
            skip = self.residual_projection(last_input).unsqueeze(1).expand(-1, self.output_steps, -1, -1, -1)
            outputs = outputs + skip

        return outputs
