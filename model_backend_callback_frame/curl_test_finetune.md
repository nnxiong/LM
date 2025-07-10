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





## 3. 使用公网路径数据集 
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




