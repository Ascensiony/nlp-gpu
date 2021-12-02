import torch.nn as nn
import copy
import functools

from transformers import EncoderDecoderModel as HFEncoderDecoderModel
from model.networks.models.utils import get_n_params
from model.networks.blocks.huggingface.encoder_decoder.beam_search import beam_search


class EncoderDecoderModel(nn.Module):
    """
    If proto is mentioned in encoder and decoder dict, loads pretrained models from proto strings.
    Otherwise, loads a BertGenerationEncoder/BertGenerationDecoder model from encoder and decoder dict.
    """

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__()

        self.enc_dec = HFEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder.pop('proto'), decoder.pop('proto'))

        # Evaluation
        self.enc_dec.beam_search = functools.partial(beam_search, self.enc_dec)

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        input_ids = input_ids.cuda()
        decoder_input_ids = decoder_input_ids.cuda()
        attention_mask = attention_mask.cuda()
        decoder_attention_mask = decoder_attention_mask.cuda()
        out = self.enc_dec(input_ids=input_ids,
                           attention_mask=attention_mask,
                           decoder_input_ids=decoder_input_ids,
                           decoder_attention_mask=decoder_attention_mask,
                           labels=decoder_input_ids)
        out = vars(out)
        return out

    def __repr__(self):
        s = str(type(self.enc_dec.encoder).__name__) + \
            '(' + str(self.enc_dec.encoder.config) + ')\n'
        s += str(type(self.enc_dec.decoder).__name__) + \
            '(' + str(self.enc_dec.decoder.config) + ')\n'
        s += "{}\n".format(get_n_params(self))
        return s
