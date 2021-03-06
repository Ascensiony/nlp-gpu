import tqdm
import torch
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


def evaluation(models, opts, dl):
    hf_models = [model.enc_dec.enc_dec for model in models]

    ref_str = 'decoder_input_ids'
    tokenizer = dl.dataset.tgt_tokenizer
    max_len = dl.dataset.tgt_len

    ref_list = []
    hyp_list = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dl):
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
                encoder_batch = {k: v.cuda() for k, v in batch.items(
                ) if not k.startswith("decoder_")}
                expanded_return_idx = (torch.arange(
                    batch['input_ids'].shape[0]).view(-1, 1).repeat(1, opts.beam_width).view(-1).cuda())
            else:
                batch = {k: v for k, v in batch.items()}
                encoder_batch = {k: v for k, v in batch.items(
                ) if not k.startswith("decoder_")}
                expanded_return_idx = (torch.arange(
                    batch['input_ids'].shape[0]).view(-1, 1).repeat(1, opts.beam_width).view(-1))

            # Run encoder ourselves
            encoders_outputs = [hf.encoder(
                **encoder_batch, return_dict=True) for hf in hf_models]

            # Tile representations
            expanded_model_outputs = [BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=h.last_hidden_state.index_select(0, expanded_return_idx)) for h in
                encoders_outputs]
            attention_masks = [batch["attention_mask"].index_select(
                0, expanded_return_idx) for _ in hf_models]

            # Create model kwargs
            model_kwargs = {"encoder_outputs": encoders_outputs[0],  # Dummy encoder output
                            "model_kwargs_list": [{"encoder_outputs": e,
                                                   "attention_mask": m,
                                                   } for e, m in zip(expanded_model_outputs, attention_masks)
                                                  ],
                            "hf_models": hf_models
                            }

            # Lets go
            hyps = hf_models[0].generate(input_ids=batch["input_ids"],
                                         num_return_sequences=1,
                                         max_length=max_len,
                                         num_beams=opts.beam_width,
                                         length_penalty=opts.length_penalty,
                                         bos_token_id=hf_models[0].decoder.config.bos_token_id,
                                         eos_token_id=hf_models[0].decoder.config.eos_token_id,
                                         pad_token_id=hf_models[0].decoder.config.pad_token_id,
                                         **model_kwargs
                                         )

            refs = batch[ref_str]
            for h, r in zip(hyps, refs):
                hyp_list.append(tokenizer.decode(
                    h, skip_special_tokens=True, clean_up_tokenization_spaces=False))
                ref_list.append(tokenizer.decode(
                    r, skip_special_tokens=True, clean_up_tokenization_spaces=False))

        return {'refs': ref_list, 'hyps': hyp_list}
