import tacotron2.models._layers as taco_layers
import torch


class Denoiser(torch.nn.Module):
    """ Removes model bias from audio produced with waveglow """

    def __init__(self, vocoder, filter_length=1024, n_overlap=4,
                 win_length=1024, mode='zeros', device='cpu'):

        # TODO: decompose (moce bias calculation to vocoder)
        super(Denoiser, self).__init__()
        self.device = device

        self.vocoder_dtype = None
        try:
            self.vocoder_dtype = vocoder.model.upsample.weight.dtype
        except:
            self.vocoder_dtype = vocoder.model.upsample_conv[0].weight.dtype

        self.stft = taco_layers.STFT(
            filter_length=filter_length,
            hop_length=int(filter_length / n_overlap),
            win_length=win_length).to(self.device)
        if mode == 'zeros':
            mel_input = torch.zeros(
                (1, 80, 88),
                dtype=self.vocoder_dtype,
                device=self.device)
        elif mode == 'normal':
            mel_input = torch.randn(
                (1, 80, 88),
                dtype=self.vocoder_dtype,
                device=self.device)
        else:
            raise Exception("Mode {} if not supported".format(mode))

        with torch.no_grad():
            bias_audio = vocoder.infer(mel_input.to(self.device), sigma=0.0).float()
            bias_spec, _ = self.stft.transform(bias_audio)

        self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.1):
        audio_spec, audio_angles = self.stft.transform(audio.to(self.device).float())
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised
