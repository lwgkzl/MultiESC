import torch
from transformers.models.bert.modeling_bert import _TOKENIZER_FOR_DOC,_CHECKPOINT_FOR_DOC,_CONFIG_FOR_DOC
from transformers.models.bert.modeling_bert import *


# MODE_TYPE = 0
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
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

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# MODEL_TYPE = 1
class BertAttentionForSequenceClassification(BertPreTrainedModel):
    """
    直接过lstm，然后取lstm每个位置的最后一个hidden，去做分类。 忽略了input_ids
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.sentence_encode = torch.nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size,
                                             num_layers=1, bidirectional=True, batch_first=True)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            history_ids=None,
            history_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            history_probs=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        #     position_ids=position_ids,
        #     head_mask=head_mask,
        #     inputs_embeds=inputs_embeds,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        #
        # pooled_output = outputs[1]

        assert token_type_ids is None
        assert position_ids is None
        assert head_mask is None
        assert inputs_embeds is None

        ######################################
        # history ids
        ######################################
        history_shape = history_ids.size()
        history_ids = history_ids.view(-1, history_shape[-1])
        history_attention_mask = history_attention_mask.view(-1, history_shape[-1])
        history_output = self.bert(
            history_ids,
            attention_mask=history_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        history_pooled_output = history_output[1]
        history_pooled_output = history_pooled_output.view(history_shape[0], history_shape[1], -1)  # bs * sen_num * dim
        # history_two_pooled, _ = torch.max(history_pooled_output, dim=1)
        history_pooled_output, (_, _) = self.sentence_encode(history_pooled_output)
        history_attention_mask = history_attention_mask.view(history_shape[0], history_shape[1], -1)
        ###############################
        # 直接过lstm，然后取lstm每个位置的最后一个hidden，去做分类。
        ###############################
        history_mask = (torch.sum(history_attention_mask, -1) != 0).int()
        sequence_len = torch.sum(history_mask, -1) - 1
        # print(len(sequence_len))
        assert len(sequence_len) == history_pooled_output.size(0)
        index = sequence_len.unsqueeze(1).unsqueeze(1).repeat(1, 1, history_pooled_output.size(-1))
        # print(history_pooled_output.size(), index.size())
        last_output = torch.gather(history_pooled_output, dim=1, index=index).squeeze(1)
        last_output = self.dropout(last_output)
        logits = self.classifier(last_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + history_output[2:]
            return ((loss,) + output) if loss is not None else output
        if history_probs is not None:
            logits = logits + history_probs
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=history_output.hidden_states,
            attentions=history_output.attentions,
        )


# MODEL_TYPE = 2
class BertConcatSequenceClassification(BertPreTrainedModel):
    """
    将history直接过lstm，然后取lstm每个位置的最后一个hidden得到user state，concat( inputs ids 和user state去做分类)
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.sentence_encode = torch.nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size,
                                             num_layers=1, bidirectional=True, batch_first=True)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size * 3, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            history_ids=None,
            history_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            history_probs=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
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

        pooled_output = outputs[1]  # 最后两轮的策略序列

        assert token_type_ids is None
        assert position_ids is None
        assert head_mask is None
        assert inputs_embeds is None

        ######################################
        # history ids
        ######################################
        history_shape = history_ids.size()
        history_ids = history_ids.view(-1, history_shape[-1])
        history_attention_mask = history_attention_mask.view(-1, history_shape[-1])
        history_output = self.bert(
            history_ids,
            attention_mask=history_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        history_pooled_output = history_output[1]
        history_pooled_output = history_pooled_output.view(history_shape[0], history_shape[1], -1)  # bs * sen_num * dim
        # history_two_pooled, _ = torch.max(history_pooled_output, dim=1)
        history_pooled_output, (_, _) = self.sentence_encode(history_pooled_output)
        history_attention_mask = history_attention_mask.view(history_shape[0], history_shape[1], -1)
        ###############################
        # 直接过lstm，然后取lstm每个位置的最后一个hidden，去做分类。
        ###############################
        history_mask = (torch.sum(history_attention_mask, -1) != 0).int()
        sequence_len = torch.sum(history_mask, -1) - 1
        # print(len(sequence_len))
        assert len(sequence_len) == history_pooled_output.size(0)
        index = sequence_len.unsqueeze(1).unsqueeze(1).repeat(1, 1, history_pooled_output.size(-1))
        # print(history_pooled_output.size(), index.size())
        last_output = torch.gather(history_pooled_output, dim=1, index=index).squeeze(1)
        last_output = torch.cat([last_output, pooled_output], -1)
        last_output = self.dropout(last_output)
        logits = self.classifier(last_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + history_output[2:]
            return ((loss,) + output) if loss is not None else output
        if history_probs is not None:
            logits = logits + history_probs
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=history_output.hidden_states,
            attentions=history_output.attentions,
        )


# MODEL_TYPE = 3
class BertPoolingSequenceClassification(BertPreTrainedModel):
    """
    将history直接过lstm，然后取lstm的max_pooling得到user state，concat( inputs ids 和user state去做分类)
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.sentence_encode = torch.nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size,
                                             num_layers=1, bidirectional=True, batch_first=True)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size * 3, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            history_ids=None,
            history_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            history_probs=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
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

        pooled_output = outputs[1]  # 最后两轮的策略序列

        assert token_type_ids is None
        assert position_ids is None
        assert head_mask is None
        assert inputs_embeds is None

        ######################################
        # history ids
        ######################################
        history_shape = history_ids.size()
        history_ids = history_ids.view(-1, history_shape[-1])
        history_attention_mask = history_attention_mask.view(-1, history_shape[-1])
        history_output = self.bert(
            history_ids,
            attention_mask=history_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        history_pooled_output = history_output[1]
        history_pooled_output = history_pooled_output.view(history_shape[0], history_shape[1], -1)  # bs * sen_num * dim
        # history_two_pooled, _ = torch.max(history_pooled_output, dim=1)
        history_pooled_output, (_, _) = self.sentence_encode(history_pooled_output)
        last_output = torch.max(history_pooled_output, 1)[0].squeeze(1)
        last_output = torch.cat([last_output, pooled_output], -1)
        last_output = self.dropout(last_output)
        logits = self.classifier(last_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + history_output[2:]
            return ((loss,) + output) if loss is not None else output
        if history_probs is not None:
            logits = logits + history_probs
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=history_output.hidden_states,
            attentions=history_output.attentions,
        )


# MODEL_TYPE = 4 cause lstm
class BertCausePoolingSequenceClassification(BertPreTrainedModel):
    """
    将history直接过lstm，然后取lstm的max_pooling得到user state，concat( inputs ids 和user state去做分类)
    额外加上cause的序列
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.sentence_encode = torch.nn.LSTM(input_size=config.hidden_size * 2, hidden_size=config.hidden_size * 2,
                                             num_layers=1, batch_first=True)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size * 3, config.num_labels)

        self.init_weights()

    def get_list_embedding(
            self,
            history_ids=None,
            history_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):
        history_shape = history_ids.size()
        history_ids = history_ids.view(-1, history_shape[-1])
        history_attention_mask = history_attention_mask.view(-1, history_shape[-1])
        history_output = self.bert(
            history_ids,
            attention_mask=history_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        history_pooled_output = history_output[1]
        history_pooled_output = history_pooled_output.view(history_shape[0], history_shape[1], -1)  # bs * sen_num * dim
        return history_pooled_output

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            history_ids=None,
            history_attention_mask=None,
            cause_ids=None,
            cause_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            history_probs=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
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

        pooled_output = outputs[1]  # 最后两轮的策略序列

        assert token_type_ids is None
        assert position_ids is None
        assert head_mask is None
        assert inputs_embeds is None

        ######################################
        # history ids
        ######################################
        history_pooled_output = self.get_list_embedding(history_ids, history_attention_mask,
                                                        token_type_ids=token_type_ids,
                                                        position_ids=position_ids,
                                                        head_mask=head_mask,
                                                        inputs_embeds=inputs_embeds,
                                                        output_attentions=output_attentions,
                                                        output_hidden_states=output_hidden_states,
                                                        return_dict=return_dict)
        cause_pooed_output = self.get_list_embedding(cause_ids, cause_attention_mask,
                                                     token_type_ids=token_type_ids,
                                                     position_ids=position_ids,
                                                     head_mask=head_mask,
                                                     inputs_embeds=inputs_embeds,
                                                     output_attentions=output_attentions,
                                                     output_hidden_states=output_hidden_states,
                                                     return_dict=return_dict)
        # history_two_pooled, _ = torch.max(history_pooled_output, dim=1)
        combine_output = torch.cat([history_pooled_output, cause_pooed_output], -1)
        combine_output, (_, _) = self.sentence_encode(combine_output)
        last_output = torch.max(combine_output, 1)[0].squeeze(1)
        last_output = torch.cat([last_output, pooled_output], -1)
        last_output = self.dropout(last_output)
        logits = self.classifier(last_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        if history_probs is not None:
            logits = logits + history_probs
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# MODEL_TYPE = 5 先去lstm最后位置的hidden，与input_ids做attention得到最终的表示
class BertAttentionSequenceClassification(BertPreTrainedModel):
    """
    将history直接过lstm，然后取lstm每个位置的最后一个hidden得到user state，用现有strategy与user state做attention得到最终的表示
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.sentence_encode = torch.nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size,
                                             num_layers=1, batch_first=True)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            history_ids=None,
            history_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            history_probs=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
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

        pooled_output = outputs[1]  # 最后两轮的策略序列

        assert token_type_ids is None
        assert position_ids is None
        assert head_mask is None
        assert inputs_embeds is None

        ######################################
        # history ids
        ######################################
        history_shape = history_ids.size()
        history_ids = history_ids.view(-1, history_shape[-1])
        history_attention_mask = history_attention_mask.view(-1, history_shape[-1])
        history_output = self.bert(
            history_ids,
            attention_mask=history_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        history_pooled_output = history_output[1]
        history_pooled_output = history_pooled_output.view(history_shape[0], history_shape[1], -1)  # bs * sen_num * dim
        # history_two_pooled, _ = torch.max(history_pooled_output, dim=1)
        history_pooled_output, (_, _) = self.sentence_encode(history_pooled_output)
        history_attention_mask = history_attention_mask.view(history_shape[0], history_shape[1], -1)
        ###############################
        # 直接过lstm，然后取lstm每个位置的最后一个hidden，去做分类。
        ###############################
        history_mask = (torch.sum(history_attention_mask, -1) != 0).int() # bs * sen_num

        sequence_attention = history_pooled_output.matmul(pooled_output.unsqueeze(-1)).squeeze(-1)
        assert history_mask.size() == sequence_attention.size(), print(f"sequence_len.size: {history_mask.size()}, last_output.size: {sequence_attention.size()}")
        # mased_sequence_attention = sequence_attention * (sequence_len > 0)
        masked_sequence_attention = torch.where(history_mask > 0, sequence_attention, torch.ones_like(sequence_attention)*1e-16)
        combine_hidden = torch.softmax(masked_sequence_attention, -1).unsqueeze(1).matmul(history_pooled_output)
        # last_output = torch.cat([last_output, pooled_output], -1)
        combine_hidden = combine_hidden.squeeze(1)
        combine_hidden = self.dropout(combine_hidden)
        logits = self.classifier(combine_hidden)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + history_output[2:]
            return ((loss,) + output) if loss is not None else output
        if history_probs is not None:
            logits = logits + history_probs
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=history_output.hidden_states,
            attentions=history_output.attentions,
        )



class BertEmbeddings1(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.emotion_embedding = nn.Embedding(70, config.hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
                persistent=False,
            )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, vads=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            if vads is not None:
                emotion_embedding = self.emotion_embedding(vads)
                inputs_embeds = inputs_embeds + emotion_embedding
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModel1(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings1(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        vads=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            vads=vads,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        # if vads is not None:
        #     emotion_embedding = self.emotion_embedding(vads)
        #     embedding_output = embedding_output + emotion_embedding

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

# MODEL_TYPE = 6 先去lstm最后位置的hidden，与input_ids做attention得到最终的表示, 最终结合input_ids的embedding
class BertAddInputSequenceClassification(BertPreTrainedModel):
    """
    将history直接过lstm，然后取lstm每个位置的最后一个hidden得到user state，用现有strategy与user state做attention得到最终的表示
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel1(config)
        self.sentence_encode = torch.nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size,
                                             num_layers=1, batch_first=True)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            history_ids=None,
            history_attention_mask=None,
            vads=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            history_probs=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
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

        pooled_output = outputs[1]  # 最后两轮的策略序列

        assert token_type_ids is None
        assert position_ids is None
        assert head_mask is None
        assert inputs_embeds is None

        ######################################
        # history ids
        ######################################
        history_shape = history_ids.size()
        history_ids = history_ids.view(-1, history_shape[-1])
        vads = vads.view(-1,history_shape[-1])
        history_attention_mask = history_attention_mask.view(-1, history_shape[-1])
        history_output = self.bert(
            history_ids,
            attention_mask=history_attention_mask,
            vads=vads,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        history_pooled_output = history_output[1]
        history_pooled_output = history_pooled_output.view(history_shape[0], history_shape[1], -1)  # bs * sen_num * dim
        # history_two_pooled, _ = torch.max(history_pooled_output, dim=1)
        history_pooled_output, (_, _) = self.sentence_encode(history_pooled_output)
        history_attention_mask = history_attention_mask.view(history_shape[0], history_shape[1], -1)
        ###############################
        # 直接过lstm，然后取lstm每个位置的最后一个hidden，去做分类。
        ###############################
        history_mask = (torch.sum(history_attention_mask, -1) != 0).int()  # bs * sen_num

        sequence_attention = history_pooled_output.matmul(pooled_output.unsqueeze(-1)).squeeze(-1)
        assert history_mask.size() == sequence_attention.size(), print(
            f"sequence_len.size: {history_mask.size()}, last_output.size: {sequence_attention.size()}")
        # mased_sequence_attention = sequence_attention * (sequence_len > 0)
        masked_sequence_attention = torch.where(history_mask > 0, sequence_attention,
                                                torch.ones_like(sequence_attention) * 1e-16)
        combine_hidden = torch.softmax(masked_sequence_attention, -1).unsqueeze(1).matmul(history_pooled_output)
        # last_output = torch.cat([last_output, pooled_output], -1)
        combine_hidden = torch.cat([combine_hidden.squeeze(1), pooled_output], -1)
        combine_hidden = self.dropout(combine_hidden)
        logits = self.classifier(combine_hidden)
        # if not self.training:
        #     print("logits: ", logits.size())
        #     print("logits: ", logits.shape, logits.device)
        #     print(logits)
        #     size = torch.tensor(logits.shape, device=logits.device)
        #     print(size)
        #     print(size[None])
        #     print("what happend? ", torch.tensor(logits.shape, device=logits.device)[None])
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + history_output[2:]
            return ((loss,) + output) if loss is not None else output
        if history_probs is not None:
            logits = logits + history_probs
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=history_output.hidden_states,
            attentions=history_output.attentions,
        )


BERTMODEL_LIST = {
    0: BertForSequenceClassification,
    1: BertAttentionForSequenceClassification,
    2: BertConcatSequenceClassification,
    3: BertPoolingSequenceClassification,
    4: BertCausePoolingSequenceClassification,
    5: BertAttentionSequenceClassification,
    6: BertAddInputSequenceClassification
}