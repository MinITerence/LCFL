import argparse
import glob
import os
import random

import scipy.io

"""
    Usage: python path/to/manifest.py \
            /path/to/signals \
            --subset $subsets \
            --combine_subsets $combine_subsets \
            --dest /path/to/manifest \
            --ext $ext \
            --valid-percent $valid


python manifest.py \
    /home/edlab/sjyang/ecg_preprocessed_data \
    --subset "CPSC2018, CPSC2018_2"
    --combine_subsets "CPSC2018, CPSC2018_2" \
    --dest /home/edlab/sjyang/federated_ecg_manifest \
    --valid-percent 0.1
"""

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",  default='/data/mcc_fj/Match/physionet.org/files/challenge-2021/1.0.3/training',
        help="root directory containing mat files to index"
    )   #root 参数是用来指定一个包含 .mat 文件的根目录的路径，并且这些 .mat 文件将会被索引（处理）。metavar="DIR",
    parser.add_argument(
        "--subset",
        default="chapman_shaoxing, cpsc_2018, georgia, ningbo", #ptb-xl, chapman_shaoxing, cpsc_2018, georgia, ningbo
        type=str,
        help="comma seperated list of data subsets to manifest for pre-training (e.g. CPSC2018, CPSC2018_2, ...), "
             "each of which should be a name of the sub-directory"
    )#default="CPSC2018, CPSC2018_2, Ga, PTBXL, ChapmanShaoxing, Ningbo",      
     #type 参数指定了该参数的类型为字符串 (str)。这意味着无论用户输入什么内容，它都会被解析为一个字符串。help描述告诉用户 --subset 参数应该是一个以逗号分隔的子集名称列表，每个名称对应一个子目录。
    parser.add_argument(
        "--combine_subsets",
        default="chapman_shaoxing, cpsc_2018, georgia, ningbo", #ptb-xl, chapman_shaoxing, cpsc_2018, georgia, ningbo
        type=str,
        help="comma seperated list of data subsets for fine-tuning (e.g. CPSC2018, CPSC2018_2, ...), "
             "each of which should be a name of the sub-directory"
    ) # default="CPSC2018, Ga",这个参数用于指定一组子集，这些子集将在后续处理中被组合在一起以进行微调（fine-tuning）help描述告诉用户 --combine_subsets 应该是一个以逗号分隔的子集名称列表，每个名称代表一个子目录，用于微调过程。

    parser.add_argument(
        "--valid-percent",
        default=0.1,
        type=float,
        help="percentage of data to use as validation and test set (between 0 and 0.5)",
    )#用于指定用于验证和测试的数据的比例metavar="D",
    parser.add_argument(
        "--dest", default="/data/mcc_fj/Match/physionet.org/files/challenge-2021/1.0.3/federated_ecg_manifest/CINC2021", type=str, help="output directory"
    )


    # parser.add_argument(
    #     "--dest", default="/data/mcc_fj/Match/physionet.org/files/challenge-2021/1.0.3/federated_ecg_manifest/PTBXL", type=str, metavar="DIR", help="output directory"
    # )
    parser.add_argument(
        "--ext", default="mat", type=str, metavar="EXT", help="extension to look for"
    )#help告诉用户 --ext 参数用于指定要查找的文件扩展名。
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    parser.add_argument(
        "--path-must-contain",
        default=None,
        type=str,
        metavar="FRAG",
        help="if set, path must contain this substring for a file to be included in the manifest",
    )#用于指定文件路径必须包含的子字符串。这个参数可以用来过滤文件，以便只有路径中包含特定子串的文件会被包括在处理列表中。
    return parser


def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 0.5  #参数验证，确保 valid_percent 参数的值在 0 到 0.5 之间。

    root_path = os.path.realpath(args.root)   #root_path：获取根路径的绝对路径。
    subset = args.subset.replace(' ', '').split(',')  #subset：将 args.subset 字符串拆分为子集名称列表。
    combine_subsets = args.combine_subsets.replace(' ', '').split(',')  #combine_subsets：将 args.combine_subsets 字符串拆分为需要合并的子集名称列表。
    rand = random.Random(args.seed)   #rand：基于指定的种子初始化随机数生成器。

    if not os.path.exists(os.path.join(args.dest, "total")):     #如果目标路径下的 total 和 cinc 目录不存在，则创建它们。
        os.makedirs(os.path.join(args.dest, "total"))
    if not os.path.exists(os.path.join(args.dest, "cinc")):
        os.makedirs(os.path.join(args.dest, "cinc"))
    


    #创建并写入文件：
    # 打开四个文件以写入数据：
    # total/train.tsv：所有数据文件的路径。
    # cinc/train.tsv：训练数据文件的路径。
    # cinc/valid.tsv：验证数据文件的路径。
    # cinc/test.tsv：测试数据文件的路径。
    # 在这些文件中写入根路径。
    with open(os.path.join(args.dest, "total/train.tsv"), "w") as total_f, open(
        os.path.join(args.dest, "cinc/train.tsv"), "w") as train_f, open(
        os.path.join(args.dest, "cinc/valid.tsv"), "w") as valid_f, open(
        os.path.join(args.dest, "cinc/test.tsv"), "w") as test_f:
        print(root_path, file=total_f)
        print(root_path, file=train_f)
        print(root_path, file=valid_f)
        print(root_path, file=test_f)
        
        #定义write函数，用于将文件路径和相关长度信息写入指定的目标文件dest
        #如果文件路径下不包含args.path_must_contain中的内容，则跳过该文件。
        #如果文件扩展名是 .mat，则读取 .mat 文件的内容，并写入文件路径和数据长度（假设数据存储在 feats 键下）。

        def write(fnames, dest):
            for fname in fnames:
                file_path = os.path.realpath(fname)

                if args.path_must_contain and args.path_must_contain not in file_path:
                    continue

                if args.ext == 'mat':
                    data = scipy.io.loadmat(file_path)
                    length = data['feats'].shape[-1]

                    print(
                        "{}".format(os.path.relpath(file_path, root_path)), file=dest, end='\t'
                    )
                    print(length, file=dest)
        #处理每个子集s:
        # 遍历每个子集 s：
        # 使用 glob 模块查找匹配的文件。
        # 如果子集 s 不在 combine_subsets 中，将所有文件路径写入 total_f 文件。
        # 如果子集 s 在 combine_subsets 中：
        # 随机打乱文件名列表。
        # 按比例划分文件名列表为训练集、验证集和测试集。
        # 将这些分割后的文件路径写入 total_f、train_f、valid_f 和 test_f 文件中。
        for s in subset:
            search_path = os.path.join(args.root, s, "**/*." + args.ext)
            fnames = list(glob.iglob(search_path, recursive=True))
            if s not in combine_subsets:
                write(fnames, total_f)
            else:
                rand.shuffle(fnames)

                valid_len = int(len(fnames) * args.valid_percent)
                test_len = int(len(fnames) * args.valid_percent)
                train_len = len(fnames) - (valid_len + test_len)

                train = fnames[:train_len]
                valid = fnames[train_len:train_len + valid_len]
                test = fnames[train_len + valid_len:]

                write(train, total_f)
                write(train, train_f)
                write(valid, valid_f)
                write(test, test_f)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    #这个函数的整体目的是读取指定路径中的数据文件，将这些文件根据子集和指定的比例划分为训练集、验证集和测试集，并将相关的信息写入相应的文件中。通过这种方式，你可以系统地准备数据集用于后续的模型训练和评估。
    print('Starting')
    main(args)
    print('Ending')

