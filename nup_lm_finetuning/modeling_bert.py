# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """

from torch.nn import CrossEntropyLoss
import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertPreTrainingHeads, BertForPreTrainingOutput

class BertForSequenceClassificationMTL(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassificationMTL, self).__init__(config)
        num_labels = 2
        max_positions = 20
        uop_weight = 0.1
        self.bert = BertModel(config)
        self.num_labels = num_labels
        self.order_dense = nn.Linear(config.hidden_size, max_positions)
        self.uop_weight = uop_weight
        self.svo_weight = 0.5
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, order_ids=None, eou_ids=None, svo_ids=None, labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output, pooled_output = outputs[:2]

        # calculate positions
        batch_size = sequence_output.size(0)
        sequence_len = sequence_output.size(1)
        hidden_size = sequence_output.size(2)
        max_utterance_len = order_ids.size(1)

        batch_id = 0
        batch_eou_ids = []
        for batch in eou_ids:
            offset = batch_id * sequence_len + 1  # 0用来做padding了
            eou_seqs = [offset + item for item in batch if item != -100]
            while (len(eou_seqs) < max_utterance_len):
                eou_seqs.append(0)
            batch_eou_ids.append(eou_seqs)
            batch_id += 1

        batch_eou_ids = torch.tensor(batch_eou_ids)
        batch_eou_ids = batch_eou_ids.view(-1)

        max_svo_len = svo_ids.size(1)

        batch_id = 0
        batch_svo_ids = []
        svo_ids = svo_ids.view(batch_size, -1)  # merge number_svos * 3

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

        batch_eou_ids = batch_eou_ids.cuda()
        eou_features = sequence_output.index_select(0, batch_eou_ids)
        eou_features = eou_features.view(batch_size, max_utterance_len, hidden_size)
        # eou_features = eou_features.cuda()

        eou_features = self.order_dense(
            eou_features)  # [batch, max_utterance, dim] -> [batch, max_utterance, max_utterance]
        max_utterance = eou_features.size(2)

        batch_svo_ids = batch_svo_ids.cuda()
        svo_features = sequence_output.index_select(0, batch_svo_ids)
        svo_features = svo_features.view(batch_size, max_svo_len, 3, hidden_size)
        # eou_features = eou_features.cuda()

        sv_features = svo_features[:, :, 0, :] + svo_features[:, :, 1, :]
        o_features = svo_features[:, :, 2, :]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            rs_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            position_orders_loss = loss_fct(eou_features.view(-1, max_utterance), order_ids.view(-1))

            criterion = nn.CosineSimilarity()
            svo_sim = criterion(sv_features.view(-1, hidden_size), o_features.view(-1, hidden_size))
            svo_inds = torch.nonzero(svo_sim)
            svo_sim = svo_sim[svo_inds]
            svo_loss = 1. - svo_sim
            svo_loss = torch.mean(svo_loss)

            loss = rs_loss + self.uop_weight * position_orders_loss + self.svo_weight * svo_loss

            return loss
        else:
            return logits

class BertForSequenceClassificationUOP(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassificationUOP, self).__init__(config)
        num_labels = 2
        self.bert = BertModel(config)
        max_positions = 20
        self.uop_weight = 0.1
        self.num_labels = num_labels
        self.order_dense = nn.Linear(config.hidden_size, max_positions)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, order_ids=None, eou_ids=None, labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output, pooled_output = outputs[:2]
        # calculate positions
        batch_size = sequence_output.size(0)
        sequence_len = sequence_output.size(1)
        hidden_size = sequence_output.size(2)
        max_utterance_len = order_ids.size(1)

        batch_id = 0
        batch_eou_ids = []
        for batch in eou_ids:
            offset = batch_id * sequence_len + 1  # 0用来做padding了
            eou_seqs = [offset + item for item in batch if item != -100]
            while (len(eou_seqs) < max_utterance_len):
                eou_seqs.append(0)
            batch_eou_ids.append(eou_seqs)
            batch_id += 1

        batch_eou_ids = torch.tensor(batch_eou_ids)
        batch_eou_ids = batch_eou_ids.view(-1)

        sequence_output = sequence_output.view(-1, hidden_size)
        sequence_output = torch.cat([sequence_output.new_zeros((1, hidden_size)), sequence_output], dim=0)

        batch_eou_ids = batch_eou_ids.cuda()
        eou_features = sequence_output.index_select(0, batch_eou_ids)
        eou_features = eou_features.view(batch_size, max_utterance_len, hidden_size)
        # eou_features = eou_features.cuda()

        eou_features = self.order_dense(
            eou_features)  # [batch, max_utterance, dim] -> [batch, max_utterance, max_utterance]
        max_utterance = eou_features.size(2)


        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            rs_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            position_orders_loss = loss_fct(eou_features.view(-1, max_utterance), order_ids.view(-1))
            loss = rs_loss + self.uop_weight * position_orders_loss

            return loss
        else:
            return logits

class BertForPreTrainingFastMTL(BertPreTrainedModel):
    def __init__(self, config, max_positions, uop_weight):
        super().__init__(config, max_positions, uop_weight)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        max_positions = 20
        uop_weight = 0.1
        self.order_dense = nn.Linear(config.hidden_size, max_positions)
        self.uop_weight = uop_weight
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        next_sentence_label=None,
        position_orders=None,
        eou_ids=None,
        svo_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        # calculate positions
        batch_size = sequence_output.size(0)
        sequence_len = sequence_output.size(1)
        hidden_size = sequence_output.size(2)
        max_utterance_len = position_orders.size(1)
        max_svo_len = svo_ids.size(1)

        batch_id = 0
        batch_eou_ids = []
        for batch in eou_ids:
            offset = batch_id * sequence_len + 1 # 0用来做padding了
            eou_seqs = [offset + item for item in batch if item != -100]
            while (len(eou_seqs) < max_utterance_len):
                eou_seqs.append(0)
            batch_eou_ids.append(eou_seqs)
            batch_id += 1
        batch_eou_ids = torch.tensor(batch_eou_ids)
        batch_eou_ids = batch_eou_ids.view(-1)

        batch_id = 0
        batch_svo_ids = []
        svo_ids = svo_ids.view(batch_size, -1)  # merge number_svos * 3
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

        batch_eou_ids = batch_eou_ids.cuda()
        eou_features = sequence_output.index_select(0, batch_eou_ids)
        eou_features = eou_features.view(batch_size, max_utterance_len, hidden_size)
        # eou_features = eou_features.cuda()

        batch_svo_ids = batch_svo_ids.cuda()
        svo_features = sequence_output.index_select(0, batch_svo_ids)
        svo_features = svo_features.view(batch_size, max_svo_len, 3, hidden_size)

        sv_features = svo_features[:, :, 0, :] + svo_features[:, :, 1, :]
        o_features = svo_features[:, :, 2, :]
        criterion = nn.CosineSimilarity()
        svo_sim = criterion(sv_features.view(-1, hidden_size), o_features.view(-1, hidden_size))
        svo_inds = torch.nonzero(svo_sim)
        svo_sim = svo_sim[svo_inds]
        svo_loss = 1. - svo_sim
        svo_loss = torch.mean(svo_loss)

        eou_features = self.order_dense(eou_features) # [batch, max_utterance, dim] -> [batch, max_utterance, max_utterance]
        max_utterance = eou_features.size(2)
        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            position_orders_loss = loss_fct(eou_features.view(-1, max_utterance), position_orders.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss + self.uop_weight * position_orders_loss  + 0.5*svo_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertForPreTrainingFast(BertPreTrainedModel):
    def __init__(self, config, max_positions, uop_weight):
        super().__init__(config, max_positions, uop_weight)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.order_dense = nn.Linear(config.hidden_size, max_positions)
        self.uop_weight = uop_weight
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        next_sentence_label=None,
        position_orders=None,
        eou_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        # calculate positions
        batch_size = sequence_output.size(0)
        sequence_len = sequence_output.size(1)
        hidden_size = sequence_output.size(2)
        max_utterance_len = position_orders.size(1)

        batch_id = 0
        batch_eou_ids = []
        for batch in eou_ids:
            offset = batch_id * sequence_len + 1 # 0用来做padding了
            eou_seqs = [offset + item for item in batch if item != -100]
            while (len(eou_seqs) < max_utterance_len):
                eou_seqs.append(0)
            batch_eou_ids.append(eou_seqs)
            batch_id += 1

        batch_eou_ids = torch.tensor(batch_eou_ids)
        batch_eou_ids = batch_eou_ids.view(-1)

        sequence_output = sequence_output.view(-1, hidden_size)
        sequence_output = torch.cat([sequence_output.new_zeros((1, hidden_size)), sequence_output], dim=0)

        batch_eou_ids = batch_eou_ids.cuda()
        eou_features = sequence_output.index_select(0, batch_eou_ids)
        eou_features = eou_features.view(batch_size, max_utterance_len, hidden_size)
        # eou_features = eou_features.cuda()

        eou_features = self.order_dense(eou_features) # [batch, max_utterance, dim] -> [batch, max_utterance, max_utterance]
        max_utterance = eou_features.size(2)
        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            position_orders_loss = loss_fct(eou_features.view(-1, max_utterance), position_orders.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss + self.uop_weight * position_orders_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config, max_positions, uop_weight):
        super().__init__(config, max_positions, uop_weight)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.order_dense = nn.Linear(config.hidden_size, max_positions)
        self.uop_weight = uop_weight
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        next_sentence_label=None,
        position_orders=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        # calculate positions
        device = input_ids.device
        batch_size = sequence_output.size(0)
        hidden_size = sequence_output.size(2)
        max_utterance_len = position_orders.size(1)
        sep_embedding = torch.zeros(batch_size, max_utterance_len, hidden_size).to(device)
        for batchi in range(batch_size):
            # every sample in a batch
            sep_pos = []
            for j in range(input_ids.size(1)):
                if input_ids[batchi][j] == 30522: # [EOU] ID: 30522
                    sep_pos.append(j)
            # sep_embedding->utterance representation
            for sep_idx in range(len(sep_pos) - 1):
                sep_embedding[batchi][sep_idx] = sequence_output[batchi, sep_pos[sep_idx], :]

        sep_embedding_dense = self.order_dense(sep_embedding) # [batch, max_utterance, dim] -> [batch, max_utterance, max_utterance]
        max_utterance = sep_embedding_dense.size(2)
        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            position_orders_loss = loss_fct(sep_embedding_dense.view(-1, max_utterance), position_orders.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss + self.uop_weight * position_orders_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
