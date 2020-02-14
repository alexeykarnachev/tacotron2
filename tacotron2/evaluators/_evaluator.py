import torch


class BaseEvaluator(object):
    """
    Base evaluator for tacotron + vocoder models pair
    """

    def __init__(self, encoder, vocoder, tokenizer, denoiser=None, device='cpu'):

        self.encoder = encoder
        self.vocoder = vocoder
        self.tokenizer = tokenizer
        self.denoiser = denoiser

        self.device = device

    def synthesize(self, text, embedding):

        sequence = torch.LongTensor(self.tokenizer.encode(text))\
                        .unsqueeze(0)\
                        .to(self.device)

        embedding = torch.FloatTensor(embedding)\
                         .unsqueeze(0)

        mel_outputs, mel_outputs_postnet, _, alignments = self.encoder.inference(sequence, embedding)
        audio = self.vocoder.infer(mel_outputs_postnet, sigma=0.9)
        if self.denoiser:
            audio = self.denoiser(audio, strength=0.05)[:, 0]

        return audio
