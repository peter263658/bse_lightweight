# model_analysis.py
import torch
from collections import OrderedDict

def collect_layer_stats(model: torch.nn.Module):
    """逐層統計參數量 / dtype / 記憶體占用"""
    stats = []
    for name, module in model.named_modules():
        if name == '':               # 跳過最外層
            continue
        # 只看本層自己的參數，不遞迴
        params = list(module.parameters(recurse=False))
        if not params:
            continue
        n_params = sum(p.numel() for p in params)
        dtype    = params[0].dtype
        bytes_per = (torch.finfo(dtype).bits if dtype.is_floating_point
                     else torch.iinfo(dtype).bits) // 8
        size_mb   = n_params * bytes_per / (1024**2)
        size_mb_q = n_params * 1        / (1024**2)   # 8-bit → 1 byte
        stats.append(OrderedDict(
            layer=name,
            cls   = module.__class__.__name__,
            n     = n_params,
            dtype = str(dtype),
            mb    = size_mb,
            mb_q8 = size_mb_q,
        ))
    return stats

def print_table(stats):
    header = f'{"Layer":40}{"Type":20}{"Params":>12}{"DType":>10}{"MB":>12}{"MB@8bit":>12}'
    print(header)
    print('-'*len(header))
    tot_n = tot_mb = tot_mb_q8 = 0
    for s in stats:
        print(f'{s["layer"]:40}{s["cls"]:20}{s["n"]:12,}{s["dtype"]:>10}'
              f'{s["mb"]:12.2f}{s["mb_q8"]:12.2f}')
        tot_n    += s['n']
        tot_mb   += s['mb']
        tot_mb_q8+= s['mb_q8']
    print('-'*len(header))
    print(f'{"TOTAL":60}{tot_n:12,}{"":>10}{tot_mb:12.2f}{tot_mb_q8:12.2f}')

if __name__ == "__main__":
    from DCNN.models.model import Model     # 依你的檔名 / class 名稱調整
    net = Model().cpu().eval()  # 重要：確定在 CPU、eval 模式
    stats = collect_layer_stats(net)
    print_table(stats)
