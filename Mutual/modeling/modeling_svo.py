from transformers import (ElectraModel, ElectraPreTrainedModel, BertModel, BertPreTrainedModel, RobertaModel)
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, Conv1d
import torch.nn.functional as F
import math
import torch.nn.utils.rnn as rnn_utils

BertLayerNorm = torch.nn.LayerNorm

ACT2FN = {"gelu": F.gelu, "relu": F.relu}

class ElectraForMultipleChoicePlus(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.electra = ElectraModel(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier2 = nn.Linear(config.hidden_size, 2)
        self.svo_weight = 0.5
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        sep_pos = None,
        position_ids=None,
        turn_ids = None,
        svo_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        num_labels = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None 
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # (batch_size * choice, seq_len)
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        sep_pos = sep_pos.view(-1, sep_pos.size(-1)) if sep_pos is not None else None
        turn_ids = turn_ids.view(-1, turn_ids.size(-1)) if turn_ids is not None else None
        # svo_ids = svo_ids.view(-1, svo_ids.size(-1)) if svo_ids is not None else None
        turn_ids = turn_ids.unsqueeze(-1).repeat([1,1,turn_ids.size(1)])
        
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (batch_size * num_choice, 1, 1, seq_len)
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility

        attention_mask = (1.0 - attention_mask) * -10000.0

        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask.squeeze(1),
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0] # (batch_size * num_choice, seq_len, hidden_size)

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:,0]))
        pooled_output = self.dropout(pooled_output)
        
        if num_labels > 2:
            logits = self.classifier(pooled_output)
        else:
            logits = self.classifier2(pooled_output)

        reshaped_logits = logits.view(-1, num_labels) if num_labels > 2 else logits

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        batch_size = sequence_output.size(0)
        sequence_len = sequence_output.size(1)
        hidden_size = sequence_output.size(2)
        max_svo_len = svo_ids.size(2)

        batch_id = 0
        batch_svo_ids = []
        # print(svo_ids.size())
        # print(batch_size)
        svo_ids = svo_ids.view(batch_size, -1)  # merge number_svos * 3
        # print(svo_ids.size())

        for batch in svo_ids:
            offset = batch_id * sequence_len + 1  # 0用来做padding了
            svo_seqs = [offset + item for item in batch if item != -100]
            while (len(svo_seqs) < max_svo_len * 3):
                svo_seqs.append(0)
            batch_svo_ids.append(svo_seqs)
            batch_id += 1

        batch_svo_ids = torch.tensor(batch_svo_ids)
        batch_svo_ids = batch_svo_ids.view(-1)

        sequence_output = sequence_output.view(-1, hidden_size)
        sequence_output = torch.cat([sequence_output.new_zeros((1, hidden_size)), sequence_output], dim=0)

        batch_svo_ids = batch_svo_ids.cuda()
        svo_features = sequence_output.index_select(0, batch_svo_ids)
        svo_features = svo_features.view(batch_size, max_svo_len, 3, hidden_size)
        # eou_features = eou_features.cuda()

        sv_features = svo_features[:, :, 0, :] + svo_features[:, :, 1, :]
        o_features = svo_features[:, :, 2, :]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            cls_loss = loss_fct(reshaped_logits, labels)

            criterion = nn.CosineSimilarity()
            svo_sim = criterion(sv_features.view(-1, hidden_size), o_features.view(-1, hidden_size))
            svo_inds = torch.nonzero(svo_sim)
            svo_sim = svo_sim[svo_inds]
            svo_loss = 1. - svo_sim
            svo_loss = torch.mean(svo_loss)
            loss = cls_loss + self.svo_weight * svo_loss

            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)

class BertForMultipleChoicePlus(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier2 = nn.Linear(config.hidden_size, 2)
        self.svo_weight=0.5
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        sep_pos = None,
        position_ids=None,
        turn_ids = None,
        svo_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        num_labels = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None 
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        orig_attention_mask = attention_mask
        # (batch_size * choice, seq_len)
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        sep_pos = sep_pos.view(-1, sep_pos.size(-1)) if sep_pos is not None else None
        turn_ids = turn_ids.view(-1, turn_ids.size(-1)) if turn_ids is not None else None

        turn_ids = turn_ids.unsqueeze(-1).repeat([1,1,turn_ids.size(1)])
        
        #print("sep_pos:", sep_pos)
        
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        
        #print("size of sequence_output:", sequence_output.size())
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (batch_size * num_choice, 1, 1, seq_len)
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility

        attention_mask = (1.0 - attention_mask) * -10000.0

        outputs = self.bert(
            input_ids,
            attention_mask=orig_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0] # (batch_size * num_choice, seq_len, hidden_size)

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:,0]))
        pooled_output = self.dropout(pooled_output)
        
        if num_labels > 2:
            logits = self.classifier(pooled_output)
        else:
            logits = self.classifier2(pooled_output)

        reshaped_logits = logits.view(-1, num_labels) if num_labels > 2 else logits

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        batch_size = sequence_output.size(0)
        sequence_len = sequence_output.size(1)
        hidden_size = sequence_output.size(2)
        max_svo_len = svo_ids.size(2)

        batch_id = 0
        batch_svo_ids = []
        # print(svo_ids.size())
        # print(batch_size)
        svo_ids = svo_ids.view(batch_size, -1)  # merge number_svos * 3
        # print(svo_ids.size())

        for batch in svo_ids:
            offset = batch_id * sequence_len + 1  # 0用来做padding了
            svo_seqs = [offset + item for item in batch if item != -100]
            while (len(svo_seqs) < max_svo_len * 3):
                svo_seqs.append(0)
            batch_svo_ids.append(svo_seqs)
            batch_id += 1

        batch_svo_ids = torch.tensor(batch_svo_ids)
        batch_svo_ids = batch_svo_ids.view(-1)

        sequence_output = sequence_output.view(-1, hidden_size)
        sequence_output = torch.cat([sequence_output.new_zeros((1, hidden_size)), sequence_output], dim=0)

        batch_svo_ids = batch_svo_ids.cuda()
        svo_features = sequence_output.index_select(0, batch_svo_ids)
        svo_features = svo_features.view(batch_size, max_svo_len, 3, hidden_size)
        # eou_features = eou_features.cuda()

        sv_features = svo_features[:, :, 0, :] + svo_features[:, :, 1, :]
        o_features = svo_features[:, :, 2, :]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            cls_loss = loss_fct(reshaped_logits, labels)

            criterion = nn.CosineSimilarity()
            svo_sim = criterion(sv_features.view(-1, hidden_size), o_features.view(-1, hidden_size))
            svo_inds = torch.nonzero(svo_sim)
            svo_sim = svo_sim[svo_inds]
            svo_loss = 1. - svo_sim
            svo_loss = torch.mean(svo_loss)
            loss = cls_loss + self.svo_weight * svo_loss

            outputs = (loss,) + outputs

        return outputs #(loss), reshaped_logits, (hidden_states), (attentions)

class RobertaForMultipleChoicePlus(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier2 = nn.Linear(config.hidden_size, 2)
        self.svo_weight = 0.5
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        sep_pos = None,
        position_ids=None,
        turn_ids = None,
        svo_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        num_labels = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None 
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        orig_attention_mask = attention_mask
        # (batch_size * choice, seq_len)
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        sep_pos = sep_pos.view(-1, sep_pos.size(-1)) if sep_pos is not None else None
        turn_ids = turn_ids.view(-1, turn_ids.size(-1)) if turn_ids is not None else None
        turn_ids = turn_ids.unsqueeze(-1).repeat([1,1,turn_ids.size(1)])
        
        #print("sep_pos:", sep_pos)
        
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        
        #print("size of sequence_output:", sequence_output.size())
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (batch_size * num_choice, 1, 1, seq_len)
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility

        attention_mask = (1.0 - attention_mask) * -10000.0

        outputs = self.roberta(
            input_ids,
            attention_mask=orig_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0] # (batch_size * num_choice, seq_len, hidden_size)

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:,0]))
        pooled_output = self.dropout(pooled_output)
        
        if num_labels > 2:
            logits = self.classifier(pooled_output)
        else:
            logits = self.classifier2(pooled_output)

        reshaped_logits = logits.view(-1, num_labels) if num_labels > 2 else logits

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        batch_size = sequence_output.size(0)
        sequence_len = sequence_output.size(1)
        hidden_size = sequence_output.size(2)
        max_svo_len = svo_ids.size(2)

        batch_id = 0
        batch_svo_ids = []
        # print(svo_ids.size())
        # print(batch_size)
        svo_ids = svo_ids.view(batch_size, -1)  # merge number_svos * 3
        # print(svo_ids.size())
        for batch in svo_ids:
            offset = batch_id * sequence_len + 1  # 0用来做padding了
            svo_seqs = [offset + item for item in batch if item != -100]
            while (len(svo_seqs) < max_svo_len * 3):
                svo_seqs.append(0)
            batch_svo_ids.append(svo_seqs)
            batch_id += 1

        batch_svo_ids = torch.tensor(batch_svo_ids)
        batch_svo_ids = batch_svo_ids.view(-1)

        sequence_output = sequence_output.view(-1, hidden_size)
        sequence_output = torch.cat([sequence_output.new_zeros((1, hidden_size)), sequence_output], dim=0)

        batch_svo_ids = batch_svo_ids.cuda()
        svo_features = sequence_output.index_select(0, batch_svo_ids)
        svo_features = svo_features.view(batch_size, max_svo_len, 3, hidden_size)
        # eou_features = eou_features.cuda()

        sv_features = svo_features[:, :, 0, :] + svo_features[:, :, 1, :]
        o_features = svo_features[:, :, 2, :]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            cls_loss = loss_fct(reshaped_logits, labels)

            criterion = nn.CosineSimilarity()
            svo_sim = criterion(sv_features.view(-1, hidden_size), o_features.view(-1, hidden_size))
            svo_inds = torch.nonzero(svo_sim)
            svo_sim = svo_sim[svo_inds]
            svo_loss = 1. - svo_sim
            svo_loss = torch.mean(svo_loss)
            loss = cls_loss + self.svo_weight * svo_loss

            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)