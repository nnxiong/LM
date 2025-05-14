import torch
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import openai_map_fn, template_map_fn_factory
from xtuner.engine.hooks import (DatasetInfoHook, EvaluateChatHook,
                                 VarlenAttnArgsToMessageHubHook)
from xtuner.engine.runner import TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.utils import PROMPT_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = '/mnt/data1/zhongrongmiao/InternLM/internlm2_5-1_8b-chat'
use_varlen_attn = False

# Data
# data_files = ['/mnt/data1/zhongrongmiao/InternLM/data_ft/instruction_dataset_train_v1.json']
# data_files = ['/mnt/data1/zhongrongmiao/InternLM/data_ft/instruction_dataset_train_v2.json']
# data_files = ['/mnt/data1/zhongrongmiao/InternLM/data_ft/instruction_dataset_train_v2.json', '/mnt/data1/zhongrongmiao/InternLM/data_ft/instruction_dataset_train_v3.json']
data_files = ['/mnt/data1/zhongrongmiao/InternLM/data_ft/instruction_dataset_train_v3_2.json']
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 2048
pack_to_max_length = True

# Scheduler & Optimizer
batch_size = 2  # per_device
accumulative_counts = 1  # bs = 1 GPU * 1 batch_size_per_device * 16 acc
dataloader_num_workers = 4
max_epochs = 8
optim_type = AdamW
lr = 2e-4
# lr = 1e-6
betas = (0.9, 0.999)
# weight_decay = 0
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 2000
save_total_limit = -1  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 500
SYSTEM = ''

evaluation_inputs = [
    '你是一名医疗场景下的操作输入提取器，需要将用户操作指令中向目标对象输入的文本提取出来，只需要输出提取出的文本，这对我很重要。现有用户操作指令：“在工作流程持续改进，护士知晓岗位职责中备注护理部主任已多次强调岗位职责”，其中用户的动作为“备注”，目标对象为“工作流程持续改进，护士知晓岗位职责”，现在请提取出操作指令中向目标对象输入的文本。',
    '你是一名医疗场景下的操作输入提取器，需要将用户操作指令中向目标对象输入的文本提取出来，只需要输出提取出的文本，这对我很重要。现有用户操作指令："把医疗垃圾转运人员防护用品穿戴情况检查结果备注到医疗规范。今日对医疗垃圾转运人员的防护用品穿戴情况进行了突击检查，发现个别转运人员存在防护服穿戴不规范、口罩佩戴不标准的问题。已对相关转运人员进行了现场纠正和教育，并再次强调了正确穿戴防护用品的重要性，以及不规范操作可能带来的感染风险，在备注中详细记录了此次检查的时间、发现的问题、处理措施和后续要求，以便后续追踪和管理。最后完成了详细的管理与记录。"，其中用户的动作为"备注"，目标对象为“医疗规范”，现在请提取出操作指令中向目标对象输入的文本。',
    '你是一名医疗场景下的目标对象与输入内容提取器，需要提取出用户操作指令中的目标对象与向目标对象输入的文本，这对我很重要。现有用户操作指令：“把夫妻感情很好写入感情记录中”，其中用户的动作为“写入”，现在请提取出操作指令中的目标对象与向目标对象输入的文本，必须要先输出目标对象，并且两者以空格分开输出。',
    '你是一名医疗场景下的目标对象与输入内容提取器，需要提取出用户操作指令中的目标对象与向目标对象输入的文本，这对我很重要。现有用户操作指令：“把患者性格沉默写进病例中”，其中用户的动作为“写进”，现在请提取出操作指令中的目标对象与向目标对象输入的文本，必须要先输出目标对象，并且两者以空格分开输出。',
    '你是一名医疗场景下的目标对象与输入内容提取器，需要提取出用户操作指令中的目标对象与向目标对象输入的文本，这对我很重要。现有用户操作指令：“往护士知晓口头医嘱制度及执行流程输入护士能够正确执行口头医嘱”，其中用户的动作为“输入”，现在请提取出操作指令中的目标对象与向目标对象输入的文本，必须要先输出目标对象，并且两者以空格分开输出。',
    '你是一名医疗场景下的目标对象与输入内容提取器，需要提取出用户操作指令中的目标对象与向目标对象输入的文本，这对我很重要。现有用户操作指令：“往流程情况输入好”，其中用户的动作为“输入”，现在请提取出操作指令中的目标对象与向目标对象输入的文本，必须要先输出目标对象，并且两者以空格分开输出。'
]
#######################################################################
#                      PART 2  Model & Tokenizer                      #
#####################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

model = dict(
    type=SupervisedFinetune,
    use_varlen_attn=use_varlen_attn,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        # quantization_config=dict(
        #     type=BitsAndBytesConfig,
        #     load_in_4bit=True,
        #     load_in_8bit=False,
        #     llm_int8_threshold=6.0,
        #     llm_int8_has_fp16_weight=False,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type='nf4')
            ),
    lora=dict(
        type=LoraConfig,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path='json', data_files=data_files),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=openai_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        system=SYSTEM,
        prompt_template=prompt_template)
]

if use_varlen_attn:
    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
