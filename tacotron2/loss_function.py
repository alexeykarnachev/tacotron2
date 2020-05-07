from torch import nn


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target, output_lengths):
        squared_error = (target - pred) ** 2
        loss = (squared_error.mean(1).sum(1) / output_lengths).mean()
        return loss


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.custom_mse = MaskedMSELoss()

    def forward(self, model_output, targets, output_lengths):
        mel_target, gate_target = targets[0], targets[1]
        # mel_target.requires_grad = False
        # gate_target.requires_grad = False
        gate_target = gate_target.reshape(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.reshape(-1, 1)
        mel_loss = self.custom_mse(mel_out, mel_target, output_lengths) + \
                   self.custom_mse(mel_out_postnet, mel_target, output_lengths)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss
