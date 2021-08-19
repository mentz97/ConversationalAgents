import torch
import torch.backends.cudnn
import torch.utils.data
import transformers
import warnings
import statistics
import pprint
import json
import numpy as np
import tqdm

from action_evaluation import evaluate_action_prediction
from dataset import le_actions, mlb_attributes, valAPIFile, testAPIFile


class CustomBertModel(transformers.BertModel):
    def __init__(self, config, num_actions, num_attributes):
        super().__init__(config)
        self.config = config
        self.bert = transformers.BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier_action = torch.nn.Linear(
            config.hidden_size, num_actions)
        self.classifier_attributes = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, num_attributes))
        self.init_weights()

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
        logits_action = self.classifier_action(pooled_output)
        logits_attributes = self.classifier_attributes(pooled_output)

        return logits_action, logits_attributes


class SIMMCModel():
    def __init__(self) -> None:
        self.num_actions = len(le_actions.classes_)
        self.num_attributes = len(mlb_attributes.classes_)

        self.bert = CustomBertModel.from_pretrained('bert-base-uncased',
                                                    output_attentions=False,
                                                    output_hidden_states=False,
                                                    num_actions=self.num_actions,
                                                    num_attributes=self.num_attributes)
        self.bert.to('cuda')

        self.score = 0

    def train(self, train_dataset, val_dataset, epochs, device, lr, wd, ss, savepath: str = None):
        warnings.simplefilter("ignore", UserWarning)

        optimizer = transformers.AdamW(
            self.bert.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=ss)
        criterion_attributes = torch.nn.BCEWithLogitsLoss(reduction='sum')
        criterion_action = torch.nn.CrossEntropyLoss(reduction='sum')

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=16, shuffle=True, num_workers=4)

        torch.backends.cudnn.benchmark

        for epoch in range(epochs):
            print(f"\nEPOCH {epoch}", flush=True)
            self.bert.train()

            go = tqdm.tqdm(train_dataloader, unit='batch', position=0)
            go.set_description(f"LR {scheduler.get_last_lr()}")
            losses = []

            for item in go:
                input_ids = item['input_ids']
                attention_mask = item['attention_mask']
                action = item['action']
                attributes = item['attributes']
                input_ids, attention_mask, action, attributes = input_ids.to(
                    device), attention_mask.to(device), action.to(device), attributes.to(device)

                optimizer.zero_grad()
                predict_action, predict_attributes = self.bert(
                    input_ids, attention_mask=attention_mask)
                loss_action = criterion_action(predict_action, action)
                loss_attributes = criterion_attributes(
                    predict_attributes, attributes)
                loss = loss_action + loss_attributes

                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                go.set_postfix(avg_loss=statistics.mean(losses))

            scheduler.step()

            # Validation
            action_metrics = self.validate(val_dataset, device)
            pprint.pprint(action_metrics)

            newScore = (action_metrics['action_accuracy'] +
                        action_metrics['attribute_accuracy']) / 2
            if savepath is not None and newScore > self.score:
                self.score = newScore
                self.bert.save_pretrained(savepath)

    def validate(self, dataset, device, savepath: str = None):
        warnings.simplefilter("ignore", UserWarning)
        logsoft = torch.nn.LogSoftmax()
        logsoft.to(device)

        criterion_attributes = torch.nn.BCEWithLogitsLoss(reduction='sum')
        criterion_action = torch.nn.CrossEntropyLoss(reduction='sum')

        turn_ids = []
        dialogue_ids = []
        actions_pred = []
        attributes_pred = []

        total_eval_loss = 0

        val_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=True, num_workers=4)

        if savepath is not None:
            model = CustomBertModel.from_pretrained(savepath,
                                                    output_attentions=False,
                                                    output_hidden_states=False,
                                                    num_actions=self.num_actions,
                                                    num_attributes=self.num_attributes)
            model.to('cuda')
        else:
            model = self.bert

        model.eval()

        for batch in tqdm.auto.tqdm(val_dataloader, leave=False):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_action = batch['action'].to(device)
            b_attributes = batch['attributes'].to(device)

            turn_ids.extend(batch['turn_id'].numpy().tolist())
            dialogue_ids.extend(batch['dialogue_id'].numpy().tolist())

            with torch.no_grad():
                predict_action, predict_attributes = model(
                    b_input_ids, attention_mask=b_input_mask)
                loss_action = criterion_action(predict_action, b_action)
                loss_attributes = criterion_attributes(
                    predict_attributes, b_attributes)
                loss = loss_action + loss_attributes

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            predict_action = logsoft(predict_action)
            predict_action = predict_action.to('cpu').numpy()
            predict_attributes = predict_attributes.to('cpu').numpy()
            actions_pred.extend(predict_action)
            attributes_pred.extend(predict_attributes > 0)

        # __valAPIFile not working
        with open(valAPIFile) as gt_actions_json:
            gt_actions = json.load(gt_actions_json)

        model_actions = []
        actions = le_actions.inverse_transform(
            [np.argmax(action) for action in actions_pred])
        action_predictions = [{k: val for k, val in zip(
            le_actions.classes_, action)} for action in actions_pred]
        attribute_predictions = [[k for k, val in zip(mlb_attributes.classes_, attributes) if(
            val == 1)] for attributes in attributes_pred]

        for (dialog_id, turn_id, action, action_prediction, attribute_prediction) in zip(dialogue_ids, turn_ids, actions, action_predictions, attribute_predictions):
            model_actions.append({
                "dialog_id": dialog_id,
                "predictions": [
                    {
                        "action": action,
                        "action_log_prob": action_prediction,
                        "attributes": {
                            "attributes": attribute_prediction
                        },
                        "turn_id": turn_id}
                ]
            })

        action_metrics = evaluate_action_prediction(gt_actions, model_actions)
        action_metrics['validation_loss'] = total_eval_loss / \
            len(val_dataloader)

        if savepath is not None:
            self.score = (action_metrics['action_accuracy'] +
                          action_metrics['attribute_accuracy']) / 2

        return action_metrics

    def test(self, dataset, device, savepath: str):
        warnings.simplefilter("ignore", UserWarning)
        logsoft = torch.nn.LogSoftmax()
        logsoft.to(device)

        criterion_attributes = torch.nn.BCEWithLogitsLoss(reduction='sum')
        criterion_action = torch.nn.CrossEntropyLoss(reduction='sum')

        turn_ids = []
        dialogue_ids = []
        actions_pred = []
        attributes_pred = []

        total_eval_loss = 0

        test_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=True, num_workers=4)

        model = CustomBertModel.from_pretrained(savepath,
                                                output_attentions=False,
                                                output_hidden_states=False,
                                                num_actions=self.num_actions,
                                                num_attributes=self.num_attributes)
        model.to('cuda')
        model.eval()

        for batch in tqdm.auto.tqdm(test_dataloader, leave=False):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_action = batch['action'].to(device)
            b_attributes = batch['attributes'].to(device)

            turn_ids.extend(batch['turn_id'].numpy().tolist())
            dialogue_ids.extend(batch['dialogue_id'].numpy().tolist())

            with torch.no_grad():
                predict_action, predict_attributes = model(
                    b_input_ids, attention_mask=b_input_mask)
                loss_action = criterion_action(predict_action, b_action)
                loss_attributes = criterion_attributes(
                    predict_attributes, b_attributes)
                loss = loss_action + loss_attributes

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            predict_action = logsoft(predict_action)
            predict_action = predict_action.to('cpu').numpy()
            predict_attributes = predict_attributes.to('cpu').numpy()
            actions_pred.extend(predict_action)
            attributes_pred.extend(predict_attributes > 0)

        with open(testAPIFile) as gt_actions_json:  
            gt_actions = json.load(gt_actions_json)

        model_actions = []
        actions = le_actions.inverse_transform(
            [np.argmax(action) for action in actions_pred])
        action_predictions = [{k: val for k, val in zip(
            le_actions.classes_, action)} for action in actions_pred]
        attribute_predictions = [[k for k, val in zip(mlb_attributes.classes_, attributes) if(
            val == 1)] for attributes in attributes_pred]

        for (dialog_id, turn_id, action, action_prediction, attribute_prediction) in zip(dialogue_ids, turn_ids, actions, action_predictions, attribute_predictions):
            model_actions.append({
                "dialog_id": dialog_id,
                "predictions": [
                    {
                        "action": action,
                        "action_log_prob": action_prediction,
                        "attributes": {
                            "attributes": attribute_prediction
                        },
                        "turn_id": turn_id}
                ]
            })

        action_metrics = evaluate_action_prediction(gt_actions, model_actions)
        action_metrics['test_loss'] = total_eval_loss / len(test_dataloader)

        return action_metrics
