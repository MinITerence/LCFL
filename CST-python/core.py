import torch.nn.functional as F
import torch, random
import numpy as np
from fairseq_signals.data.ecg import ecg_utils
import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def naive_train(args, model, train_loader, optimizer, criterion, device='cpu', scheduler=None) :
    model.train()

    total_loss = 0.0
    preds, targets = [], []

    for idx, t in enumerate(train_loader) :
        
        optimizer.zero_grad()

        source = t['net_input']['source'].to(device)
        padding_mask = t['net_input']['padding_mask'].to(device)
        label = t['label'].to(device).float()

        output = model(source)
        loss = criterion(output, label)

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.max_norm)

        optimizer.step()
        total_loss += loss.item()

        torch.cuda.empty_cache()

    if scheduler is not None :
        scheduler.step()

    return float(total_loss) / len(train_loader)

#########################################################

def train_fedprox(args, server_model, model, train_loader, criterion, optimizer, param_name=None, device='cpu', scheduler=None) :
    model.train()

    total_loss = 0.0

    for idx, t in enumerate(train_loader) :
        
        optimizer.zero_grad()

        source = t['net_input']['source'].to(device)
        padding_mask = t['net_input']['padding_mask'].to(device)
        label = t['label'].to(device).float()

        output = model(source)

        loss = criterion(output, label)

        if idx > 0:
            w_diff = torch.tensor(0., device=device)
            if param_name != None :
                for name, w, w_t in zip(param_name, server_model.parameters(), model.parameters()):
                    if "norm" not in name :
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
            else :
                for w, w_t in zip(server_model.parameters(), model.parameters()):
                    w_diff += torch.pow(torch.norm(w - w_t), 2)

            loss += args.mu / 2. * w_diff

        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.max_norm)
        optimizer.step()
        total_loss += loss.item()

    if scheduler is not None :
        scheduler.step()

    return float(total_loss) / len(train_loader)

#########################################################


def train_feddyn(args, model, server_model_param, train_loader, criterion, optimizer, prev_grads, device='cpu', scheduler=None) :
    model.train()

    total_loss = 0.0

    for idx, t in enumerate(train_loader) :
        
        optimizer.zero_grad()

        source = t['net_input']['source'].to(device)
        padding_mask = t['net_input']['padding_mask'].to(device)
        label = t['label'].to(device).float()

        output = model(source)

        loss = criterion(output, label)

        ######
        ## Dynamic regularizer
                
        local_par_list = None
        for param in model.parameters():
            if not isinstance(local_par_list, torch.Tensor):
            # Initially nothing to concatenate
                local_par_list = param.reshape(-1)
            else:
                local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
                                
        loss_algo = args.feddyn_alpha * torch.sum(local_par_list * (-server_model_param + prev_grads))
        loss += loss_algo
        ######

        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.max_norm)
        optimizer.step()
        total_loss += loss.item()

    if scheduler is not None :
        scheduler.step()

    return float(total_loss) / len(train_loader)

#########################################################

def naive_test(args, model, test_loader, score_weights, sinus_rhythm_index, criterion, c, device='cpu') :
    model.eval()

    total_loss = 0
    preds, targets = [], []
    all_labels, all_outputs = [], []

    with torch.no_grad():
        for idx, t in enumerate(test_loader) :
            
            source = t['net_input']['source'].to(device)
            padding_mask = t['net_input']['padding_mask'].to(device)
            label = t['label'].to(device).float()
            output = model(source)
            preds.extend( output.detach().cpu().numpy() )
            targets.extend(label.cpu().numpy())
            loss = criterion(output, label)
            total_loss += loss.item()

            #
            probs = torch.sigmoid(output)
            preds_binary = (probs > 0.5).float().cpu().numpy()
            all_outputs.extend(preds_binary)
            all_labels.extend(label.cpu().numpy())
            # all_labels.append(label.cpu().numpy())
            # all_outputs.append(output.detach().cpu().numpy())

            torch.cuda.empty_cache()

    logging_output = get_score(np.array(preds), np.array(targets), score_weights, sinus_rhythm_index)
    test_loss = float(total_loss) / len(test_loader)
    cinc_score = get_cinc_score(logging_output)

    # # Convert lists of numpy arrays into a single numpy array
    # all_labels_np = np.concatenate(all_labels)
    # all_outputs_np = np.concatenate(all_outputs)
    # Calculate additional metrics
    # outputs_labels = (all_outputs_np > 0.5).astype(int)  # Assuming binary classification
    # precision = precision_score(all_labels_np, outputs_labels, average='macro')
    # recall = recall_score(all_labels_np, outputs_labels, average='macro')
    # f1 = f1_score(all_labels_np, outputs_labels, average='macro')
    # acc = accuracy_score(all_labels_np, outputs_labels)
    
    
    
    preds_array = np.array(all_outputs)
    targets_array = np.array(all_labels)
    # 将预测结果和标签从one-hot编码转换为分类标签
    preds_labels = np.argmax(preds_array, axis=1)
    targets_labels = np.argmax(targets_array, axis=1)
    accuracy = accuracy_score(targets_labels, preds_labels)  
    precision = precision_score(targets_array, preds_array, average='samples')
    recall = recall_score(targets_array, preds_array, average='samples')
    f1 = f1_score(targets_array, preds_array, average='samples')  

    # Prepare data to save to CSV
    results = {
        f'client{c}_test_loss': [test_loss],
        f'client{c}_cinc_score': [cinc_score],
        f'client{c}_precision': [precision],
        f'client{c}_recall': [recall],
        f'client{c}_f1_score': [f1],
        f'client{c}_accuracy': [accuracy]
    }

    df = pd.DataFrame(results)

    # Define path where the CSV file will be saved
    csv_path = os.path.join('./ecg_federated/Resultsss/', 'fedavg_server_model_valid_results_client-{}.csv'.format(c))

    # Check if the file already exists to decide whether to write header
    file_exists = os.path.isfile(csv_path)

    # Append new row to CSV file
    df.to_csv(csv_path, mode='a', header=not file_exists, index=False)



    return test_loss, cinc_score




def cl_naive_test(args, model, test_loader, score_weights, sinus_rhythm_index, criterion, c, device='cpu') :
    model.eval()

    total_loss = 0
    preds, targets = [], []
    all_labels, all_outputs = [], []

    with torch.no_grad():
        for idx, t in enumerate(test_loader) :
            
            source = t['net_input']['source'].to(device)
            label = t['label'].to(device).float()
            output = model(source)
            preds.extend( output.detach().cpu().numpy() )
            targets.extend(label.cpu().numpy())
            loss = criterion(output, label)
            total_loss += loss.item()


            #
            probs = torch.sigmoid(output)
            preds_binary = (probs > 0.5).float().cpu().numpy()
            all_outputs.extend(preds_binary)
            all_labels.extend(label.cpu().numpy())
            # all_labels.append(label.cpu().numpy())
            # all_outputs.append(output.detach().cpu().numpy())

            torch.cuda.empty_cache()
    
    logging_output = get_score(np.array(preds), np.array(targets), score_weights, sinus_rhythm_index)
    test_loss = float(total_loss) / len(test_loader)
    cinc_score = get_cinc_score(logging_output)

    # # Convert lists of numpy arrays into a single numpy array
    # all_labels_np = np.concatenate(all_labels)
    # all_outputs_np = np.concatenate(all_outputs)
    # # Calculate additional metrics
    # outputs_labels = (all_outputs_np > 0.5).astype(int)  # Assuming binary classification
    # precision = precision_score(all_labels_np, outputs_labels, average='macro')
    # recall = recall_score(all_labels_np, outputs_labels, average='macro')
    # f1 = f1_score(all_labels_np, outputs_labels, average='macro')
    # acc = accuracy_score(all_labels_np, outputs_labels)


    preds_array = np.array(all_outputs)
    targets_array = np.array(all_labels)
    # 将预测结果和标签从one-hot编码转换为分类标签
    preds_labels = np.argmax(preds_array, axis=1)
    targets_labels = np.argmax(targets_array, axis=1)
    accuracy = accuracy_score(targets_labels, preds_labels)  
    precision = precision_score(targets_array, preds_array, average='samples')
    recall = recall_score(targets_array, preds_array, average='samples')
    f1 = f1_score(targets_array, preds_array, average='samples')  


    # Prepare data to save to CSV
    results = {
        f'cl_{c}_test_loss': [test_loss],
        f'cl_{c}_cinc_score': [cinc_score],
        f'cl_{c}_precision': [precision],
        f'cl_{c}_recall': [recall],
        f'cl_{c}_f1_score': [f1],
        f'cl_{c}_accuracy': [accuracy]
    }

    df = pd.DataFrame(results)

    # Define path where the CSV file will be saved
    csv_path = os.path.join('./ecg_federated/Resultsss/', 'cl_model_valid_results-{}.csv'.format(c))

    # Check if the file already exists to decide whether to write header
    file_exists = os.path.isfile(csv_path)

    # Append new row to CSV file
    df.to_csv(csv_path, mode='a', header=not file_exists, index=False)



    return test_loss, cinc_score




########################################################################################################################################

def get_score(probs, target, score_weights, sinus_rhythm_index):
    
    logging_output = {}

    probs = torch.tensor(probs)
    target = torch.tensor(target)

    outputs = (probs > 0.5)

    if probs.numel() == 0:
        corr = 0
        count = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
    else:
        count = float(probs.numel())
        corr = (outputs == target).sum().item()

        true = torch.where(target == 1)
        false = torch.where(target == 0)
        tp = outputs[true].sum()
        fn = outputs[true].numel() - tp
        fp = outputs[false].sum()
        tn = outputs[false].numel() - fp

    logging_output["correct"] = corr
    logging_output["count"] = count

    logging_output["tp"] = tp.item()
    logging_output["fp"] = fp.item()
    logging_output["tn"] = tn.item()
    logging_output["fn"] = fn.item()

    # cinc score
    labels = target.cpu().numpy()

    observed_score = ( ecg_utils.compute_scored_confusion_matrix( score_weights, labels, outputs.cpu().numpy() ) )
    correct_score = ( ecg_utils.compute_scored_confusion_matrix( score_weights, labels, labels ) )
    inactive_outputs = np.zeros(outputs.size(), dtype=bool)
    inactive_outputs[:, sinus_rhythm_index] = 1
    inactive_score = ( ecg_utils.compute_scored_confusion_matrix( score_weights, labels, inactive_outputs ) )

    logging_output["o_score"] = observed_score
    logging_output["c_score"] = correct_score
    logging_output["i_score"] = inactive_score

    return logging_output

def get_cinc_score(logging_output):
    cinc_score = 0.0

    c_score = logging_output['c_score']
    i_score = logging_output['i_score']
    o_score = logging_output['o_score']

    # reference : https://github.com/physionetchallenges/evaluation-2021/blob/main/evaluate_model.py
    if c_score - i_score != 0 :
        cinc_score = round( abs(float(o_score- i_score) / (float(c_score - i_score))), 3)
    else :
        cinc_score = 0.0        

    return cinc_score
