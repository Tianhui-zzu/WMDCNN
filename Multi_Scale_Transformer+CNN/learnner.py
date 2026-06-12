from torch.optim.lr_scheduler import LambdaLR

factor = 400 # 调和参数


def get_customized_schedule_with_warmup(optimizer, num_warmup_steps, d_model=1.0, last_epoch=-1):
    def lr_lambda(current_step):
        current_step += 1
        arg1 = current_step ** -0.5
        arg2 = current_step * (num_warmup_steps ** -1.5)
        return (d_model ** -0.5) * factor * min(arg1, arg2)

    return LambdaLR(optimizer, lr_lambda, last_epoch)
