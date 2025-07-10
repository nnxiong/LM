## 1. 使用epoch 作为单位
```js
curl -X "POST" "http://0.0.0.0:8001/finetune" \
    -H "Content-Type: application/json" \
    -d '{
        "task_id": "task_1",
        "call_back_api": "http://172.17.0.1:5000/callback",
        "input_content": {
            "customized_input":{        
                "train_dataset_path": "/app/finetune_datasets/test/train",  
                "valid_dataset_path": "/app/finetune_datasets/test/test", 
                "test_dataset_path": "/app/finetune_datasets/test/test"  
            }
        },
        "param_dict": {
            "finetune_mode": "full",      
            "config_param": {},         
            "show_frequecy": 2,       
            "measurement": "epoch",     
            "total_epoch_number": 5,    
            "batch_size": 64,          
            "lr": 0.001                
            }  
    }'
```


返回值

```js
{'status': 'success', 'task_id': 'task_1', 'is_final': False, 'idx': 2, 'fig_param': {'x_axis': [2], 'x_label': 'epoch', 'losses': [106.59809218160808], 'accuracies': [0.9657666666666667], 'valid_losses': [7.502392057795078], 'valid_accuracies': [0.9724]}, 'test_accuracy': None, 'output_model_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708024147_task_1/model_files', 'output_log_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708024147_task_1/finetune.log'}
192.168.1.4 - - [08/Jul/2025 10:42:00] "POST /callback HTTP/1.1" 200 -

{'status': 'success', 'task_id': 'task_1', 'is_final': False, 'idx': 4, 'fig_param': {'x_axis': [2, 4], 'x_label': 'epoch', 'losses': [106.59809218160808, 51.49035492935218], 'accuracies': [0.9657666666666667, 0.9829], 'valid_losses': [7.502392057795078, 6.990388248581439], 'valid_accuracies': [0.9724, 0.973]}, 'test_accuracy': None, 'output_model_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708024147_task_1/model_files', 'output_log_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708024147_task_1/finetune.log'}
192.168.1.4 - - [08/Jul/2025 10:42:11] "POST /callback HTTP/1.1" 200 -

{'status': 'success', 'task_id': 'task_1', 'is_final': True, 'idx': 5, 'fig_param': {'x_axis': [2, 4, 5], 'x_label': 'epoch', 'losses': [106.59809218160808, 51.49035492935218, 39.23344399477355], 'accuracies': [0.9657666666666667, 0.9829, 0.9863666666666666], 'valid_losses': [7.502392057795078, 6.990388248581439, 6.788880517240614], 'valid_accuracies': [0.9724, 0.973, 0.975]}, 'test_accuracy': 0.9766, 'output_model_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708024147_task_1/model_files', 'output_log_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708024147_task_1/finetune.log'}
192.168.1.4 - - [08/Jul/2025 10:42:17] "POST /callback HTTP/1.1" 200 
```



## 2. 使用step 作为单位
```js
curl -X "POST" "http://0.0.0.0:8001/finetune" \
    -H "Content-Type: application/json" \
    -d '{
        "task_id": "task_1",
        "call_back_api": "http://172.17.0.1:5000/callback",
        "input_content": {
            "customized_input":{        
                "train_dataset_path": "/app/finetune_datasets/test/train",  
                "valid_dataset_path": "/app/finetune_datasets/test/test", 
                "test_dataset_path": "/app/finetune_datasets/test/test"  
            }
        },
        "param_dict": {
            "finetune_mode": "full",      
            "config_param": {},         
            "show_frequecy": 200,       
            "measurement": "step",     
            "total_epoch_number": 2,    
            "batch_size": 64,          
            "lr": 0.001                
            }  
    }'
```

返回值


```js
{'status': 'success', 'task_id': 'task_1', 'is_final': False, 'idx': 200, 'fig_param': {'x_axis': [200], 'x_label': 'step', 'losses': [124.7792757153511], 'accuracies': [0.834375], 'valid_losses': [24.149120017886162], 'valid_accuracies': [0.912]}, 'test_accuracy': None, 'output_model_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708110553_task_1/model_files', 'output_log_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708110553_task_1/finetune.log'}                                                     
192.168.1.4 - - [08/Jul/2025 11:05:55] "POST /callback HTTP/1.1" 200 -     

{'status': 'success', 'task_id': 'task_1', 'is_final': False, 'idx': 400, 'fig_param': {'x_axis': [200, 400], 'x_label': 'step', 'losses': [124.7792757153511, 180.18749879300594], 'accuracies': [0.834375, 0.8766796875], 'valid_losses': [24.149120017886162, 17.687379088252783], 'valid_accuracies': [0.912, 0.9322]}, 'test_accuracy': None, 'output_model_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708110553_task_1/model_files', 'output_log_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708110553_task_1/finetune.log'}                
192.168.1.4 - - [08/Jul/2025 11:05:57] "POST /callback HTTP/1.1" 200 -    

{'status': 'success', 'task_id': 'task_1', 'is_final': False, 'idx': 600, 'fig_param': {'x_axis': [200, 400, 600], 'x_label':'step', 'losses': [124.7792757153511, 180.18749879300594, 225.46224972233176], 'accuracies': [0.834375, 0.8766796875, 0.8963020833333334], 'valid_losses': [24.149120017886162, 17.687379088252783, 15.488204557448626], 'valid_accuracies': [0.912, 0.9322, 0.9386]}, 'test_accuracy': None, 'output_model_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708110553_task_1/model_files', 'output_log_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708110553_task_1/finetune.log'}     
192.168.1.4 - - [08/Jul/2025 11:06:06] "POST /callback HTTP/1.1" 200 -

...


{'status': 'success', 'task_id': 'task_1', 'is_final': False, 'idx': 1800, 'fig_param': {'x_axis': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800], 'x_label': 'step', 'losses': [124.7792757153511, 180.18749879300594, 225.46224972233176, 263.21922554448247, 9.105998162180185, 34.85259155370295, 59.054693249985576, 81.35298366099596, 101.24457423295826], 'accuracies': [0.834375, 0.8766796875, 0.8963020833333334, 0.908359375, 0.954133064516129, 0.9599236641221374, 0.9617830086580087, 0.9632269637462235, 0.9645265371229699], 'valid_losses': [24.14912001786162, 17.687379088252783, 15.488204557448626, 12.307836800813675, 10.718489112332463, 13.128510132431984, 8.98106194101274, 8.78214100934565, 7.916733741760254], 'valid_accuracies': [0.912, 0.9322, 0.9386, 0.9534, 0.958, 0.9476, 0.9618, 0.966, 0.9698]}, 'test_accuracy': None, 'output_model_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708110553_task_1/model_files', 'output_log_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708110553_task_1/finetune.log'}              
192.168.1.4 - - [08/Jul/2025 11:06:08] "POST /callback HTTP/1.1" 200 -

{'status': 'success', 'task_id': 'task_1', 'is_final': True, 'idx': 1876, 'fig_param': {'x_axis': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 1876], 'x_label': 'step', 'losses': [124.7792757153511, 180.18749879300594, 225.46224972233176, 263.21922554448247, 9.105998162180185, 34.85259155370295, 59.054693249985576, 81.35298366099596, 101.24457423295826, 108.76271163765341], 'accuracies': [0.834375, 0.8766796875, 0.8963020833333334, 0.908359375, 0.954133064516129, 0.9599236641221374, 0.9617830086580087, 0.9632269637462235, 0.9645265371229699, 0.9650166666666666], 'valid_losses': [24.149120017886162, 17.687379088252783, 15.488204557448626, 12.307836800813675, 10.718489112332463, 13.128510132431984, 8.98106194101274, 8.78214100934565, 7.916733741760254, 8.14450735412538], 'valid_accuracies': [0.912, 0.9322, 0.9386, 0.9534, 0.958, 0.9476, 0.9618, 0.966, 0.9698, 0.9688]}, 'test_accuracy': 0.9702, 'output_model_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708110553_task_1/model_files', 'output_log_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708110553_task_1/finetune.log'} 
192.168.1.4 - - [08/Jul/2025 11:06:09] "POST /callback HTTP/1.1" 200 -
```





## 使用公网路径数据集 
```js
curl -X "POST" "http://0.0.0.0:8001/finetune" \
    -H "Content-Type: application/json" \
    -d '{
        "task_id": "task_1",
        "call_back_api": "http://172.17.0.1:5000/callback",
        "input_content": {
            "customized_input":{        
                "train_dataset_path": "https://oxtium-bioford-public.obs.cn-south-1.myhuaweicloud.com/com/oxtium/bioford/test_examples/sam_img-demo.png",  
                "valid_dataset_path": "https://oxtium-bioford-public.obs.cn-south-1.myhuaweicloud.com/com/oxtium/bioford/test_examples/sam_img-demo.png",
                "test_dataset_path": "https://oxtium-bioford-public.obs.cn-south-1.myhuaweicloud.com/com/oxtium/bioford/test_examples/sam_img-demo.png"
            }
        },
        "param_dict": {
            "finetune_mode": "full",      
            "config_param": {},         
            "show_frequecy": 2,       
            "measurement": "epoch",     
            "total_epoch_number": 5,    
            "batch_size": 64,          
            "lr": 0.001                
            }  
    }'
```



```js
{'status': 'success', 'task_id': 'task_1', 'is_final': False, 'idx': 2, 'fig_param': {'x_axis': [2], 'x_label': 'epoch', 'losses': [106.62065229937434], 'accuracies': [0.9658], 'valid_losses': [7.071825040504336], 'valid_accuracies': [0.9722]}, 'test_accuracy': None, 'output_model_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708113946_task_1/model_files', 'output_log_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708113946_task_1/finetune.log'}
192.168.1.4 - - [08/Jul/2025 11:40:00] "POST /callback HTTP/1.1" 200 -

{'status': 'success', 'task_id': 'task_1', 'is_final': False, 'idx': 4, 'fig_param': {'x_axis': [2, 4], 'x_label': 'epoch', 'losses': [106.62065229937434, 52.0320337517187], 'accuracies': [0.9658, 0.983], 'valid_losses': [7.071825040504336, 6.796358909457922], 'valid_accuracies': [0.9722, 0.975]}, 'test_accuracy': None, 'output_model_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708113946_task_1/model_files', 'output_log_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708113946_task_1/finetune.log'}
192.168.1.4 - - [08/Jul/2025 11:40:12] "POST /callback HTTP/1.1" 200 -

{'status': 'success', 'task_id': 'task_1', 'is_final': True, 'idx': 5, 'fig_param': {'x_axis': [2, 4, 5], 'x_label': 'epoch', 'losses': [106.62065229937434, 52.0320337517187, 40.017796917585656], 'accuracies': [0.9658, 0.983, 0.9861833333333333], 'valid_losses': [7.071825040504336, 6.796358909457922, 5.8163201552815735], 'valid_accuracies': [0.9722, 0.975, 0.9784]}, 'test_accuracy': 0.9782, 'output_model_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708113946_task_1/model_files', 'output_log_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708113946_task_1/finetune.log'}
192.168.1.4 - - [08/Jul/2025 11:40:18] "POST /callback HTTP/1.1" 200 -
```























```js
curl -X "POST" "http://0.0.0.0:8001/finetune" \
    -H "Content-Type: application/json" \
    -d '{
        "task_id": "task_1",
        "call_back_api": "http://172.17.0.1:5000/callback",
        "input_content": {
            "customized_input":{        
                "train_dataset_path": "/app/finetune_datasets/test/train",  # 必有字段，值可以是网上路径 or 宿主机路径
                "valid_dataset_path": "/app/finetune_datasets/test/test",  # 必有字段，值可以是网上路径 or 宿主机路径
                "test_dataset_path": "/app/finetune_datasets/test/test"  # 必有字段，值可以是网上路径 or 宿主机路径
            },
        },
        "param_dict": {
            "finetune_mode": "full",      # 必有字段，值str
            "config_param": {},         # 必有字段，值dict
            "show_frequecy": 2,        # 必有字段，值int
            "measurement": "epoch",      # 必有字段，值 "step" or "epoch"
            "total_epoch_number": 5,     # 必有字段，值 int 必须大于等于 show_frequecy字段
            "batch_size": 64,          # 必有字段，int
            "lr": 0.001                # 必有字段，float
            }  
    }'
```












以下参数为生成路径
```python
model_path="/mnt/data1/bioford2/model_path/test",
output_model_path="/mnt/data1/bioford2/finetuned_model_path/test/20250423020919_task_2_1/model_files/",   # 各个模型需要保存best model ckpts和相关文件,并返回路径: e.g. /mnt/data1/bioford2/finetuned_model_path/test/20250423020919_task_2_1/model_files     
output_log_path="/mnt/data1/bioford2/finetuned_model_path/test/20250423020919_task_2_1/finetune.log" 
```


返回值
```json
{"status":"success","task_id":"task_1", "is_final":false, "idx": 1, "fig_param": {"x_axis": [1], "x_label": "epoch", "losses": [285.361563436687], "accuracies": [0.9111833333333333], "valid_losses": [11.166753727942705], "valid_accuracies": [0.957]}, "test_accuracy": None, "output_model_path": "../finetuned_model_path/test/20250423020919_task_2_1/model_files/", "output_log_path": "/mnt/data1/bioford2/finetuned_model_path/test/20250423020919_task_2_1/finetune.log" }
```







```js
{'idx': 1, 'fig_param': {'x_axis': [1], 'x_label': 'epoch', 'losses': [285.361563436687], 'accuracies': [0.9111833333333333], 'valid_losses': [11.166753727942705], 'valid_accuracies': [0.957]}, 'test_accuracy': None, 'output_model_path': '../finetuned_model_path/test/best_model_epoch1.pth', 'output_log_path': '../finetuned_model_path/test/finetune.log'}

{'idx': 2, 'fig_param': {'x_axis': [1, 2], 'x_label': 'epoch', 'losses': [285.361563436687, 108.07768914476037], 'accuracies': [0.9111833333333333, 0.9658833333333333], 'valid_losses': [11.166753727942705, 7.902920858934522], 'valid_accuracies': [0.957, 0.9718]}, 'test_accuracy': None, 'output_model_path': '../finetuned_model_path/test/best_model_epoch2.pth', 'output_log_path': '../finetuned_model_path/test/finetune.log'}

{'idx': 3, 'fig_param': {'x_axis': [1, 2, 3], 'x_label': 'epoch', 'losses': [285.361563436687, 108.07768914476037, 73.21068456652574], 'accuracies': [0.9111833333333333, 0.9658833333333333, 0.9760166666666666], 'valid_losses': [11.166753727942705, 7.902920858934522, 6.396531878970563], 'valid_accuracies': [0.957, 0.9718, 0.9766]}, 'test_accuracy': None, 'output_model_path': '../finetuned_model_path/test/best_model_epoch3.pth', 'output_log_path': '../finetuned_model_path/test/finetune.log'}

{'idx': 4, 'fig_param': {'x_axis': [1, 2, 3, 4], 'x_label': 'epoch', 'losses': [285.361563436687, 108.07768914476037, 73.21068456652574, 53.40357377578039], 'accuracies': [0.9111833333333333, 0.9658833333333333, 0.9760166666666666, 0.9829666666666667], 'valid_losses': [11.166753727942705, 7.902920858934522, 6.396531878970563, 6.121795257786289], 'valid_accuracies': [0.957, 0.9718, 0.9766, 0.976]}, 'test_accuracy': None, 'output_model_path': None, 'output_log_path': '../finetuned_model_path/test/finetune.log'}

{'idx': 5, 'fig_param': {'x_axis': [1, 2, 3, 4, 5], 'x_label': 'epoch', 'losses': [285.361563436687, 108.07768914476037, 73.21068456652574, 53.40357377578039, 39.33394915191457], 'accuracies': [0.9111833333333333, 0.9658833333333333, 0.9760166666666666, 0.9829666666666667, 0.9867333333333334], 'valid_losses': [11.166753727942705, 7.902920858934522, 6.396531878970563, 6.121795257786289, 5.399720215937123], 'valid_accuracies': [0.957, 0.9718, 0.9766, 0.976, 0.9794]}, 'test_accuracy': 0.9806, 'output_model_path': '../finetuned_model_path/test/best_model_epoch5.pth', 'output_log_path': '../finetuned_model_path/test/finetune.log'}
```

    
