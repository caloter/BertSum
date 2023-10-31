
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from models.encoder import TransformerInterEncoder, Classifier, RNNEncoder
from models.optimizers import Optimizer


def build_optim(args, model, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if args.train_from != '':
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    if args.train_from != '':
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim




import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# 定义模型
class Bert(nn.Module):
    def __init__(self, temp_dir, load_pretrained_bert, bert_config):
        super(Bert, self).__init__()
        if load_pretrained_bert:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel(bert_config)

    def forward(self, x, segs, mask):
        encoded_layers, _ = self.model(x, segs, attention_mask=mask)
        top_vec = encoded_layers[-1]
        return top_vec

# 构建模型
class Summarizer(nn.Module):
    def __init__(self, args, device, load_pretrained_bert=False, bert_config=None):
        super(Summarizer, self).__init__()
        self.args = args
        self.device = device
        if hasattr(args, 'temp_dir') and args.temp_dir is not None:
            self.bert = Bert(args.temp_dir, load_pretrained_bert, bert_config)
        else:
            raise ValueError("Missing 'temp_dir' attribute in 'args' object.")

        # 添加其他初始化代码，如 encoder、optimizer 等

    def forward(self, x, segs, clss, mask, mask_cls, sentence_range=None):
        top_vec = self.bert(x, segs, mask)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        # 添加摘要生成的代码

# 定义优化器的构建函数
def build_optim(args, model, checkpoint):
    # 添加构建优化器的代码

# 加载模型和检查点
def load_model(model_path, device):
    # 添加加载模型和检查点的代码
    # 返回加载的模型、优化器等对象

# 准备输入文本和其他预处理步骤
def prepare_input(text, tokenizer):
    # 添加输入文本的预处理代码

# 执行摘要生成
def generate_summary(model, input_data):
    # 添加摘要生成的代码
    # 返回摘要结果

# 处理输出结果
def process_output(summary_tokens, tokenizer):
    # 添加处理输出结果的代码
    # 返回最终的摘要字符串

if __name__ == '__main__':
    # 设置参数
    args = YourArgsClass(...)  # 你需要替换成你的参数设置

    # 加载模型和检查点
    model, optimizer, checkpoint = load_model('your_model_checkpoint.pth', device)

    # 准备输入文本
    input_text = "Your input text goes here..."
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    input_data = prepare_input(input_text, tokenizer)

    # 执行摘要生成
    summary_tokens = generate_summary(model, input_data)

    # 处理输出结果
    summary_text = process_output(summary_tokens, tokenizer)

    print("Generated Summary:")
    print(summary_text)

