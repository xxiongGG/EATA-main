from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from torch.nn import Module, Linear
import torch.nn as nn


class dha_llm(Module):
    def __init__(self, input_size, configs, device):
        super(dha_llm, self).__init__()

        self.input_dim = input_size
        self.device = device
        self.pretrain = configs.pretrain
        self.dropout = configs.dropout
        self.object_num = configs.object_num
        self.hidden_dim = configs.hidden_size
        self.model_path = configs.model_path

        if configs.pretrain:
            self.gpt2 = GPT2Model.from_pretrained(self.model_path, output_attentions=True, output_hidden_states=True)
        else:
            print("Failed to load the pretrain model.")

        print('gpt2 = {}'.format(self.gpt2))

        # 将输入数据维度转换为LLM输入维度
        self.in_layer = Linear(self.input_dim, 768)
        self.out_layer = Linear(768, self.hidden_dim)
        self.pred_layer = Linear(self.hidden_dim, self.object_num)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(self.dropout)

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):

                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (
                self.gpt2, self.in_layer, self.out_layer, self.sigmoid, self.dropout, self.pred_layer):
            layer.to(device=self.device)
            layer.train()

    def forward(self, input_x):
        llm_input = self.in_layer(input_x)
        llm_output = self.gpt2(inputs_embeds=llm_input).last_hidden_state
        h = self.dropout(llm_output)
        h = self.out_layer(h)
        outputs = self.pred_layer(h)
        outputs = self.sigmoid(outputs)
        return outputs
