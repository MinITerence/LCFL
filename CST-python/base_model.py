from copy import deepcopy
from typing import NoReturn, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor

import torch_ecg  # noqa: F401

from cfg import TrainCfg_ns, ModelCfg, ModelCfg_ns

from torch_ecg.cfg import CFG
from torch_ecg.components.outputs import MultiLabelClassificationOutput
from torch_ecg.models.ecg_crnn import ECG_CRNN
from torch_ecg.utils.misc import add_docstring
from easydict import EasyDict as ED

__all__ = [
    "ECG_CRNN_CINC2021_f1",
]

# reference : https://discuss.pytorch.org/t/how-to-modify-a-pretrained-model/60509/13
def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)
            
        if isinstance(module, old):
            ## simple module
            channel_size = module.weight.shape[0]
            setattr(model, n, torch.nn.GroupNorm(2, channel_size, eps=1e-8) )


def load_crnn(args):
    ECG_CRNN_CINC2021.__DEBUG__ = False
    train_config = deepcopy(TrainCfg_ns)
    train_config.cnn_name = "resnet_nature_comm_bottle_neck_se"
    train_config.rnn_name = "none"
    train_config.attn_name = "none"

    train_config.log_dir = None
    train_config.model_dir = None

    train_config.n_leads = len(train_config.leads)

    tranches = train_config.tranches_for_training
    if tranches:
        classes = train_config.tranche_classes[tranches]
    else:
        classes = train_config.classes

    model_config = deepcopy(ModelCfg_ns.twelve_leads)

    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name
    model_config.clf = ED()
    model_config.clf.out_channels = [
    # not including the last linear layer, whose out channels equals n_classes
    ]
    model_config.clf.bias = True
    model_config.clf.dropouts = 0.0
    model_config.clf.activation = "mish"  # for a single layer `SeqLin`, activation is ignored

    model = ECG_CRNN_CINC2021(
        classes=train_config.classes,
        n_leads=train_config.n_leads,
        config=model_config
    )

    if args.model_type == "resnet_gn" :
        replace_layers(model, torch.nn.BatchNorm1d, torch.nn.GroupNorm(4, 128) ) # third term is dummy value

    return model



def load_crnn_2():
    ECG_CRNN_CINC2021.__DEBUG__ = False
    train_config = deepcopy(TrainCfg_ns)
    train_config.cnn_name = "resnet_nature_comm_bottle_neck_se"
    train_config.rnn_name = "none"
    train_config.attn_name = "none"

    train_config.log_dir = None
    train_config.model_dir = None

    train_config.n_leads = len(train_config.leads)

    tranches = train_config.tranches_for_training
    if tranches:
        classes = train_config.tranche_classes[tranches]
    else:
        classes = train_config.classes

    model_config = deepcopy(ModelCfg_ns.twelve_leads)

    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name
    model_config.clf = ED()
    model_config.clf.out_channels = [
    # not including the last linear layer, whose out channels equals n_classes
    ]
    model_config.clf.bias = True
    model_config.clf.dropouts = 0.0
    model_config.clf.activation = "mish"  # for a single layer `SeqLin`, activation is ignored

    model = ECG_CRNN_CINC2021(
        classes=train_config.classes,
        n_leads=train_config.n_leads,
        config=model_config,
    )

    return model
############################################################################################################

class ECG_CRNN_CINC2021(ECG_CRNN):
    """ """

    __DEBUG__ = False
    __name__ = "ECG_CRNN_CINC2021"

    def __init__(
        self, classes, n_leads, config=None
    ) :
        """
        Parameters
        ----------
        classes: list,
            list of the classes for classification
        n_leads: int,
            number of leads (number of input channels)
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        
        model_config = CFG(deepcopy(ModelCfg))
        model_config.update(deepcopy(config) or {})
        super().__init__(classes, n_leads, model_config)

    def inference(
        self,
        input,
        class_names= False,
        bin_pred_thr= 0.5,
    ) :
        """
        auxiliary function to `forward`, for CINC2021,
        Parameters
        ----------
        input: ndarray or Tensor,
            input tensor, of shape (batch_size, channels, seq_len)
        class_names: bool, default False,
            if True, the returned scalar predictions will be a `DataFrame`,
            with class names for each scalar prediction
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions
        Returns
        -------
        MultiLabelClassificationOutput, with the following items:
            classes: list,
                list of the classes for classification
            thr: float,
                threshold for making binary predictions from scalar predictions
            prob: ndarray or DataFrame,
                scalar predictions, (and binary predictions if `class_names` is True)
            prob: ndarray,
                the array (with values 0, 1 for each class) of binary prediction
        NOTE that when `input` is ndarray, one should make sure that it is transformed,
        e.g. bandpass filtered, normalized, etc.
        """
        if "NSR" in self.classes:
            nsr_cid = self.classes.index("NSR")
        elif "426783006" in self.classes:
            nsr_cid = self.classes.index("426783006")
        else:
            nsr_cid = None
        _input = torch.as_tensor(input, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        # batch_size, channels, seq_len = _input.shape
        prob = self.sigmoid(self.forward(_input))
        pred = (prob >= bin_pred_thr).int()
        prob = prob.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        for row_idx, row in enumerate(pred):
            row_max_prob = prob[row_idx, ...].max()
            if row_max_prob < ModelCfg.bin_pred_nsr_thr and nsr_cid is not None:
                pred[row_idx, nsr_cid] = 1
            elif row.sum() == 0:
                pred[row_idx, ...] = (
                    (
                        (prob[row_idx, ...] + ModelCfg.bin_pred_look_again_tol)
                        >= row_max_prob
                    )
                    & (prob[row_idx, ...] >= ModelCfg.bin_pred_nsr_thr)
                ).astype(int)
        if class_names:
            prob = pd.DataFrame(prob)
            prob.columns = self.classes
            prob["pred"] = ""
            for row_idx in range(len(prob)):
                prob.at[row_idx, "pred"] = np.array(self.classes)[
                    np.where(pred[row_idx] == 1)[0]
                ].tolist()
        return MultiLabelClassificationOutput(
            classes=self.classes,
            thr=bin_pred_thr,
            prob=prob,
            pred=pred,
        )

    def inference_CINC2021(
        self,
        input,
        class_names=False,
        bin_pred_thr=0.5,
    ) :
        """
        alias for `self.inference`
        """
        return self.inference(input, class_names, bin_pred_thr)


class ECG_CRNN_CINC2021_fl(ECG_CRNN):
    
    __DEBUG__ = False
    __name__ = "ECG_CRNN_CINC2021_fl"

    def __init__(self, classes, n_leads, chunksize=1, config=None):
        # 确保传递了所有必要的参数给父类构造函数
        super(ECG_CRNN_CINC2021_fl, self).__init__(classes, n_leads)  # 传递必要的参数给父类构造函数
        model_config = CFG(deepcopy(ModelCfg))
        model_config.update(deepcopy(config) or {})
        self.classes = classes
        self.n_leads = n_leads
        self.chunksize = chunksize
        self.model_config = model_config
        self.total_param, self.coef_size = self._count_param()

    def inference(
        self,
        input,
        class_names= False,
        bin_pred_thr= 0.5,
    ) :
        """
        auxiliary function to `forward`, for CINC2021,
        Parameters
        ----------
        input: ndarray or Tensor,
            input tensor, of shape (batch_size, channels, seq_len)
        class_names: bool, default False,
            if True, the returned scalar predictions will be a `DataFrame`,
            with class names for each scalar prediction
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions
        Returns
        -------
        MultiLabelClassificationOutput, with the following items:
            classes: list,
                list of the classes for classification
            thr: float,
                threshold for making binary predictions from scalar predictions
            prob: ndarray or DataFrame,
                scalar predictions, (and binary predictions if `class_names` is True)
            prob: ndarray,
                the array (with values 0, 1 for each class) of binary prediction
        NOTE that when `input` is ndarray, one should make sure that it is transformed,
        e.g. bandpass filtered, normalized, etc.
        """
        if "NSR" in self.classes:
            nsr_cid = self.classes.index("NSR")
        elif "426783006" in self.classes:
            nsr_cid = self.classes.index("426783006")
        else:
            nsr_cid = None
        _input = torch.as_tensor(input, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        # batch_size, channels, seq_len = _input.shape
        prob = self.sigmoid(self.forward(_input))
        pred = (prob >= bin_pred_thr).int()
        prob = prob.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        for row_idx, row in enumerate(pred):
            row_max_prob = prob[row_idx, ...].max()
            if row_max_prob < ModelCfg.bin_pred_nsr_thr and nsr_cid is not None:
                pred[row_idx, nsr_cid] = 1
            elif row.sum() == 0:
                pred[row_idx, ...] = (
                    (
                        (prob[row_idx, ...] + ModelCfg.bin_pred_look_again_tol)
                        >= row_max_prob
                    )
                    & (prob[row_idx, ...] >= ModelCfg.bin_pred_nsr_thr)
                ).astype(int)
        if class_names:
            prob = pd.DataFrame(prob)
            prob.columns = self.classes
            prob["pred"] = ""
            for row_idx in range(len(prob)):
                prob.at[row_idx, "pred"] = np.array(self.classes)[
                    np.where(pred[row_idx] == 1)[0]
                ].tolist()
        return MultiLabelClassificationOutput(
            classes=self.classes,
            thr=bin_pred_thr,
            prob=prob,
            pred=pred,
        )

    def inference_CINC2021(
        self,
        input,
        class_names=False,
        bin_pred_thr=0.5,
    ) :
        """
        alias for `self.inference`
        """
        return self.inference(input, class_names, bin_pred_thr)

    @torch.no_grad()
    def _count_param(self):
        total_param = sum(p.numel() for p in self.parameters())
        coef_size = ((total_param + self.chunksize - 1) // self.chunksize) * self.chunksize
        return total_param, coef_size
    
    @torch.no_grad()
    def _partition(self):
        all_param = torch.empty(self.coef_size, dtype=next(self.parameters()).dtype, device=next(self.parameters()).device)
        counter = 0

        for param in self.parameters():
            numel = param.numel()
            start = counter
            end = start + numel
            all_param[start:end] = param.view(-1)
            counter = end
    
        # # 确保我们已经使用了所有分配的空间
        # if counter != self.coef_size:
        #     raise ValueError(f"Not all elements in all_param were used. Used {counter}, expected {self.coef_size}")

        return all_param.view(-1, self.chunksize).detach()

    @torch.no_grad()
    def compute_grad(self, global_model):
        for local_param, global_param in zip(self.parameters(), global_model.parameters()):
            if local_param.requires_grad:
                local_param.grad = local_param.data - global_param.data

    @torch.no_grad()
    def param_transform(self, global_model, seed=0):
        self.compute_grad(global_model)
        param = self._partition()  # Assume this returns a tensor of shape [chunk_num, chunk_size]
        torch.manual_seed(seed)
        coef = torch.randn((self.coef_size // self.chunksize, self.chunksize), dtype=param.dtype, device=param.device) # block_num, chunk_num  # Ensure coef has the same shape as param
        #projection = (coef * param).sum(dim=-1, keepdim=True).detach()  # Dot product over the last dimension
        projection = torch.sum(coef * param, dim=(1,), keepdim=True).detach() # block_num, 1
        return projection
    
    
    @torch.no_grad()
    def shape_recovery(self, agg_param):
        counter = 0
        for module in self.modules():  # 使用 self.modules() 来遍历所有子模块
            for name, param in module.named_parameters(recurse=False):  # recurse=False 确保只处理当前模块的参数
                start = counter
                end = start + param.numel()
                block_param = agg_param[start:end]
                reshaped_block_param = torch.reshape(block_param, param.shape)
                param.data.copy_(reshaped_block_param)  # 使用 copy_ 方法来更新参数值
                counter = end


    @torch.no_grad()
    def aggregate(self, projection_list, seed_list):
        param_size = (self.total_param % self.chunksize) + self.total_param
        coef_list = []
        for (seed, proj) in zip(seed_list, projection_list):
            torch.manual_seed(seed)
            coef = torch.randn((self.coef_size // self.chunksize, self.chunksize), dtype=proj.dtype, device=proj.device)
            coef_list.append(coef)

        a = torch.stack(coef_list, dim=0)
        coef_tensor = a.permute(dims=(1, 0, 2))

        b = torch.stack(projection_list, dim=0)
        projection_tensor = b.permute(dims=(1, 0, 2))

        #coef_tensor = torch.permute(torch.stack(coef_list, dim=0), dims=(1, 0, 2)) # user_num, block_num, chunk_num --> block_num, user_num, chunk_num
        #projection_tensor = torch.permute(torch.stack(projection_list, dim=0), dims=(1, 0, 2)) # user_num, block_num, 1 --> block_num, user_num, 1
        agg_param = torch.pinverse(coef_tensor) @ projection_tensor # (block_num, chunk_num, user_num) @ (block_num, user_num, 1) --> block_num, chunk_num, 1
        self.shape_recovery(torch.reshape(agg_param.detach(), (-1,)))




def load_crnn_f1(args):
    ECG_CRNN_CINC2021_fl.__DEBUG__ = False
    train_config = deepcopy(TrainCfg_ns)
    train_config.cnn_name = "resnet_nature_comm_bottle_neck_se"
    train_config.rnn_name = "none"
    train_config.attn_name = "none"

    train_config.log_dir = None
    train_config.model_dir = None

    train_config.n_leads = len(train_config.leads)

    tranches = train_config.tranches_for_training
    if tranches:
        classes = train_config.tranche_classes[tranches]
    else:
        classes = train_config.classes

    model_config = deepcopy(ModelCfg_ns.twelve_leads)

    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name
    model_config.clf = ED()
    model_config.clf.out_channels = [
    # not including the last linear layer, whose out channels equals n_classes
    ]
    model_config.clf.bias = True
    model_config.clf.dropouts = 0.0
    model_config.clf.activation = "mish"  # for a single layer `SeqLin`, activation is ignored

    model = ECG_CRNN_CINC2021_fl(
        classes=train_config.classes,
        n_leads=train_config.n_leads,
        config=model_config,
        chunksize= args.chunksize
    )

    if args.model_type == "resnet_gn" :
        replace_layers(model, torch.nn.BatchNorm1d, torch.nn.GroupNorm(4, 128) ) # third term is dummy value

    return model













