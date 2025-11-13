---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:24
- loss:FitMixinLoss
base_model: cross-encoder/ms-marco-TinyBERT-L2
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/ms-marco-TinyBERT-L2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-TinyBERT-L2](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-TinyBERT-L2](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2) <!-- at revision c9187181f395bd1e7907e7764adc9f3aa6afb26a -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['최근 득남/득녀 소식', '인프라운영팀 ( 강남 ) 한민수 사원이 득녀 하였습니다. 건강히 잘 자랄 수 있도록 많은 축하 격려 부탁 드립니다. - 출생일시 : 2022. 06. 30 ( 목 ) 15 : 28 - 아빠 : 한민수 사원 ( 010 - 5408 - 2409 ) 첫째 득녀를 진심으로 축하 합니다.'],
    ['출산 휴가 또는 경조금 제도', '[UNK] 안녕하세요. 보험IT서비스3팀 김정민 대리 님의 결혼소식이 있습니다. 많은 분들의 축하 인사를 부탁 드립니다 : D ♥일시 : 2020년 12월 12일 토요일 오후 12시 30분 → 2021년 03월 27일 토요일 오후 12시 30분 ♥장소 : 서울 서초구 신반포로 23 엘루체컨벤션 6층 스텔라하우스홀 ♥연락처 : 김정민 대리 ( 010 - 8540 - 6710 ) ♥모바일 청접장 : http : / / mcard. barunnfamily. com / B3060013? 7c2a ※ 코로나 2. 5단계로 인하여 2020년 12월 12일에 예정되었던 결혼식이 2021년 3월 27일로 변경되었습니다.'],
    ['출산 휴가 또는 경조금 제도', '안녕하세요, NB lab 경사 소식 안내 드립니다. NB lab의 막둥이를 막 탈출한 김현호 사원님의 결혼을 아래와 같이 안내 드리오니, 다들 아낌없는 축하와 격려 부탁 드립니다! [UNK] 일시 : 2024년 06월 22일 토요일, 17시 30분 [UNK] 장소 : 서울 강남대로 213 8층 / 엘하우스홀 [UNK] 연락처 : 김현호 사원 ( 010 - 2372 - 0741 ) [UNK] 계좌번호 : 하나은행 620211389159 ( 김현호 ) [UNK] 모바일 청첩장 : https : / / bojagicard. com / mcard / popup. php? ecard = kgusgh'],
    ['최근 득남/득녀 소식', '안녕하세요. 아키텍처팀 송용근 대리의 결혼 소식이 있습니다. 많은 분들의 축하 인사를 부탁 드립니다. 일시 : 2022년 10월 03일 ( 개천절 ) 월요일 낮 12시 40분 장소 : 서울특별시 영등포구 국회대로38길 2 더컨벤션 영등포점 1층 그랜드볼룸 연락처 : 송용근 대리 ( 010 - 9043 - 9456 ) 모바일 청첩장 : https : / / bojagicard. com / i / home. php? uid = cmzl25'],
    ['출산 휴가 또는 경조금 제도', '##인트 + 생일포인트 3 만원 ) ) / 12 } + 월정급여액 이외 과세소득항목 Ex ) 과세소득항목 : 통신비, 역량육성비, 직무수당, 기술수당, 직책수당, 교통비, 기타수당, 자녀장학금 등 비과세항목 ( 중식대 20 만원, 6 세 이하 자녀 장학금 20 만원 ) 이외의 모든 지급액 2 ) 세율표상 공제대상 가족의 수 : 2023 년 연말정산 시 기본공제 대상자 수 3 ) 적용일 : 2024 년 2 월 급여부터 문의 : 재무팀 이아로 ( Tel. 02 - 708 - 6815 / E - mail arlee @ kyobodts. co. kr )'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    '최근 득남/득녀 소식',
    [
        '인프라운영팀 ( 강남 ) 한민수 사원이 득녀 하였습니다. 건강히 잘 자랄 수 있도록 많은 축하 격려 부탁 드립니다. - 출생일시 : 2022. 06. 30 ( 목 ) 15 : 28 - 아빠 : 한민수 사원 ( 010 - 5408 - 2409 ) 첫째 득녀를 진심으로 축하 합니다.',
        '[UNK] 안녕하세요. 보험IT서비스3팀 김정민 대리 님의 결혼소식이 있습니다. 많은 분들의 축하 인사를 부탁 드립니다 : D ♥일시 : 2020년 12월 12일 토요일 오후 12시 30분 → 2021년 03월 27일 토요일 오후 12시 30분 ♥장소 : 서울 서초구 신반포로 23 엘루체컨벤션 6층 스텔라하우스홀 ♥연락처 : 김정민 대리 ( 010 - 8540 - 6710 ) ♥모바일 청접장 : http : / / mcard. barunnfamily. com / B3060013? 7c2a ※ 코로나 2. 5단계로 인하여 2020년 12월 12일에 예정되었던 결혼식이 2021년 3월 27일로 변경되었습니다.',
        '안녕하세요, NB lab 경사 소식 안내 드립니다. NB lab의 막둥이를 막 탈출한 김현호 사원님의 결혼을 아래와 같이 안내 드리오니, 다들 아낌없는 축하와 격려 부탁 드립니다! [UNK] 일시 : 2024년 06월 22일 토요일, 17시 30분 [UNK] 장소 : 서울 강남대로 213 8층 / 엘하우스홀 [UNK] 연락처 : 김현호 사원 ( 010 - 2372 - 0741 ) [UNK] 계좌번호 : 하나은행 620211389159 ( 김현호 ) [UNK] 모바일 청첩장 : https : / / bojagicard. com / mcard / popup. php? ecard = kgusgh',
        '안녕하세요. 아키텍처팀 송용근 대리의 결혼 소식이 있습니다. 많은 분들의 축하 인사를 부탁 드립니다. 일시 : 2022년 10월 03일 ( 개천절 ) 월요일 낮 12시 40분 장소 : 서울특별시 영등포구 국회대로38길 2 더컨벤션 영등포점 1층 그랜드볼룸 연락처 : 송용근 대리 ( 010 - 9043 - 9456 ) 모바일 청첩장 : https : / / bojagicard. com / i / home. php? uid = cmzl25',
        '##인트 + 생일포인트 3 만원 ) ) / 12 } + 월정급여액 이외 과세소득항목 Ex ) 과세소득항목 : 통신비, 역량육성비, 직무수당, 기술수당, 직책수당, 교통비, 기타수당, 자녀장학금 등 비과세항목 ( 중식대 20 만원, 6 세 이하 자녀 장학금 20 만원 ) 이외의 모든 지급액 2 ) 세율표상 공제대상 가족의 수 : 2023 년 연말정산 시 기본공제 대상자 수 3 ) 적용일 : 2024 년 2 월 급여부터 문의 : 재무팀 이아로 ( Tel. 02 - 708 - 6815 / E - mail arlee @ kyobodts. co. kr )',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 24 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 24 samples:
  |         | sentence_0                                                                                     | sentence_1                                                                                       | label                                                          |
  |:--------|:-----------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                         | string                                                                                           | float                                                          |
  | details | <ul><li>min: 11 characters</li><li>mean: 13.33 characters</li><li>max: 15 characters</li></ul> | <ul><li>min: 18 characters</li><li>mean: 264.04 characters</li><li>max: 865 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.46</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                   | sentence_1                                                                                                                                                                                                                                                                                                                                                          | label            |
  |:-----------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>최근 득남/득녀 소식</code>     | <code>인프라운영팀 ( 강남 ) 한민수 사원이 득녀 하였습니다. 건강히 잘 자랄 수 있도록 많은 축하 격려 부탁 드립니다. - 출생일시 : 2022. 06. 30 ( 목 ) 15 : 28 - 아빠 : 한민수 사원 ( 010 - 5408 - 2409 ) 첫째 득녀를 진심으로 축하 합니다.</code>                                                                                                                                                                                           | <code>1.0</code> |
  | <code>출산 휴가 또는 경조금 제도</code> | <code>[UNK] 안녕하세요. 보험IT서비스3팀 김정민 대리 님의 결혼소식이 있습니다. 많은 분들의 축하 인사를 부탁 드립니다 : D ♥일시 : 2020년 12월 12일 토요일 오후 12시 30분 → 2021년 03월 27일 토요일 오후 12시 30분 ♥장소 : 서울 서초구 신반포로 23 엘루체컨벤션 6층 스텔라하우스홀 ♥연락처 : 김정민 대리 ( 010 - 8540 - 6710 ) ♥모바일 청접장 : http : / / mcard. barunnfamily. com / B3060013? 7c2a ※ 코로나 2. 5단계로 인하여 2020년 12월 12일에 예정되었던 결혼식이 2021년 3월 27일로 변경되었습니다.</code> | <code>0.0</code> |
  | <code>출산 휴가 또는 경조금 제도</code> | <code>안녕하세요, NB lab 경사 소식 안내 드립니다. NB lab의 막둥이를 막 탈출한 김현호 사원님의 결혼을 아래와 같이 안내 드리오니, 다들 아낌없는 축하와 격려 부탁 드립니다! [UNK] 일시 : 2024년 06월 22일 토요일, 17시 30분 [UNK] 장소 : 서울 강남대로 213 8층 / 엘하우스홀 [UNK] 연락처 : 김현호 사원 ( 010 - 2372 - 0741 ) [UNK] 계좌번호 : 하나은행 620211389159 ( 김현호 ) [UNK] 모바일 청첩장 : https : / / bojagicard. com / mcard / popup. php? ecard = kgusgh</code>          | <code>0.0</code> |
* Loss: [<code>FitMixinLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#fitmixinloss)

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Framework Versions
- Python: 3.13.5
- Sentence Transformers: 5.1.1
- Transformers: 4.57.0
- PyTorch: 2.9.1+cpu
- Accelerate: 1.11.0
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->