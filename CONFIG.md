# Configuration file structure

This file contains the file structure of the configuration files used in the TransFusion project.

- #### split
  - subset: 0, 1, or 2, specify which of the 3 generated splits to use. Default is 0 (for Ego4D uses the challenge provided one which has weaker stratification. 1,2,3 will use generated ones that preserve class distribution).
  - version: leave as 42.
  - type: *group_stratified* for the realistic grouped splits or *stratified* for the easier splits.
  - all_samples_as_train: False/True, whether to use all the available labeled data for model training and use test data as the validation set -> directly obtain test set predictions without resuming from checkpoints.
  - all_samples_as_val: False/True, whether to run model on inference mode on the validation and test data at the specified epoch period. Useful if you want to produce test predictions on the go with the 27k train data samples.

- pretrained: True/False, whether to use pretrained visual encoder or not. Set to `${CODE}/checkpoints/translated_ego4d.pth` (Ego4Dv1) or `${CODE}/checkpoints/translated_ego4dv2.pth` (Ego4Dv2) to use Ego4d pretrained weights.
- finetune: True/False, whether to finetune the last stage of the visual encoder or not. Used for compatibility reasons. Change the **trainable_layers** and **freeze_backbone_at_epoch** entries to replicate this effect.

- #### model configuration
configuration file for the visual encoder. Default is ResNet50 specified in `nao/configs/ego_vis_det_ego4d{v2?}.yml`. Use *adapt_to_detectron:True*, *batch_norm.use:False*, *load_fpn_rpn:True* and *representation_size:1024* to run Res50 compatible with Ego4D weights. If *load_fpn_rpn==False*, then the RCNN stages will be trained from scratch and only the encoder weights will be used.

- ##### aug: augmentation parameters
  - resize_spec: given as h,w pairs. If it's 2 lists, will randomly pick one of the entries for each sample. Must be in increasing order.
  - channel_order: RGB/BGR, **BGR is needed for Ego4d weights**.
- #### dataset:
  - name: one of *ego4djpg* or *ego4djpgv2*. *ego4djpg* uses the full resolution JPGs (extracted only for the prediction frame). *ego4d* or *ego4dv2* uses images from LMDB with 480p height.
  - subsample: how many samples to keep from the datasets. Use *null* to keep all of them. *500_1500_1500* will keep 500 samples for train, 1500 for validation and test.
  - args:
    - label_cutoff: drop classes whose number of appearances is smaller than the specified values.
    - dampen: coefficient to smoothen out class weights, making the gradients less extreme (like in the Skip-Gram negative sampling). 0.25 is the default value. You can also specify values for specific word types explicitly, e.g. "dampen_noun" or "dampen_verb". 
    - use_external_label_mapping: *True*/*False*, set to True to use the Ego4D class mappings (should not change to False).
    - narr_structure: *{gt_narr}*/*{external_0}*/*{external_0}{external_1}*. Used to choose previous action narrations style. *gt_narr* uses the dataset provided annotations (does not use below path). *external_i* versions use the paths specified below.
    - narr_external_paths: A list of paths, one for each *external_i* entry. E.g. *${DATA}/object_lang_ego4d{v2?}.json* for salient objects and *${DATA}/action_lang_ego4d{v2?}.json* for past actions.
    
- #### run:
  - run_test: *True/False*, whether to run the current configuration on test only data. Use with **--resume-from** to obtain test-set predictions.
  - normalization: *own*, *imagenet*, *ego4d_baseline*, own uses dataset based statistics. Ego4d_baseline uses 0-255 mapping, needed when using their model weights.
  - replace_heads: *all/False*. Set to **all** to enable *transfer learning* on different dataset (e.g. Ego4dV1 to Ego4dV2), otherwise leave *False* or commented out.
  - freeze_backbone_at_epoch: *int*, when to freeze backbone and keep only classification/regression heads (e.g. for transfer learning). Set to *-1* to disable it.
  - flow_args: arguments for models with multimodal flow processing (unused). 
  - hand_args: arguments for including hand positions for ttc prediction heads (unused).
  - narration_embeds: arguments to configure narration processing stream.
    - use: *True/False*
    - slowfast_f: *True/False* if set to *True* and use is *True*, will use extracted SlowFast features instead of language. 
    - args:
      - strategy: of the form *prev_int<w>/current* where the int value represents the context length and optionally *w* postfix specifies if individual word tokens are used or only the phrase level embedding from BERT models. **Set to current to use generated captions instead of the GT ones.**
      - pooling: *max/mean* for *glove* type embeddings, *sbert* to use fixed *Sentence-BERT* embeddings, faster. Used to embed a single narration e.g. *take food* into a fixed-length representation.
      - lang_dropout, num_heads, size, out_mlp, out_dropout, normalize: techniques to apply on the output feature vector such as using an extra MLP or vector normalization.
      - model_v: one of *all-MiniLM-L6-v2*, *all-MiniLM-L12-v2*, *all-distilroberta-v1*, *distilgpt2*, *flan-t5-large*.
      - train_ep: *int* epoch from which the language encoder is trained. Set to -1 to freeze it.
    
   - ##### narr_fusion:
      - config: path to the desired configuration file for fusion. Can be *${CODE}/modeling/cross_fusion/ego_fusion/cross_fusion_config_sym_ego_res50_ego4d{v2?}.yml files.
      - fpn_features: list with the int ids of fpn level features to be used.
      - replace_fpn_features: leave as True.
    
    - bg_weight: *float* loss weight for the background class.
    - all_class_w: *True/False*, enables class probability weighted gradients for more focus on rare classes. If *bg_weight == 1*, it uses the mean value for the background class loss. Use *dampen* to make gradients less extreme.
    - criterion: if any of the values is set to 0, it won't be used in the final weight computation. Use non-zero to enable gradient propagation with the specific loss.
      - agg: *mean* weights the loss values with the corresponding value. *sum* simply adds them as if all the weights are 1.
      - bbox: weight for the localization loss. Includes rpn loss from RCNN and also per class bbox regression.
      - obj_prop: weight for rpn loss.
      - lm: weight for the language modeling loss.
      - lm_decay: *float* per epoch decay term for LM value s.t. it can become negligible towards end of training.
      - ttc_beta: *float* beta for the smooth_L1 loss used for ttc.  
    - optimizer: arguments for the optimizer.
      - sep_encoders: option to divide the learning rate of the backbone parameters (div_rate) or the TTC head (ttc_rate).
    - scheduler: arguments for using schedulers on top of the optimizer.

### Transformer fusion configuration file
- cross_fusion_config_sym_ego_res50.yml:
    - model, type: leave to "cross_f" and "cross_transformer"
    - share_encoders: *True/False* whether to share transformer weights across fpn levels or not.
    - narr_out_mode: *embedding/tokens*, *embedding* gets the pooled phrase representation in a single token, *tokens* forwards all word embeddings. *tokens* by default.
    - patch_h, patch_w: lists of patch sizes (the kernel size for a convolution). Smaller patches usually improve performance at the cost of a larger memory footprint.
    - backproj_dropout: dropout applied over the fused visual tokens before regrouping into the 2D feature map shape.
    - pos_embedding: one of *learned/zero/sin1d*. Might be improved with relative ones.
    - forward_language_f: *False/direct/sum*. Whether to copy the language model embeddings or forward them without (*direct*) or with (*sum*) residual connections across FPN levels.
    - vis_mask_type: *local_int/global*. Use local attention only with a window size of 2*int or global, where all visual tokens attend to each other.
    - args: arguments for transformer encoder construction such as *patch_dropout, num_layers, num_heads* etc. Leave *back_to_img_fn* to "regroup".
    - final_norm: *ln/bn/False* apply additional normalization on the output visual tokens, *ln* improves the performance.
    - flow_projection, flow_args, flow_return_layers, temporal_embedding: used for flow multimodal processing, needs to be redone.
    - lm_args: arguments for the language model additional loss on the fused language tokens.
      - pooling:
        - type: *mean/max*, uses pooling over the tokens to obtain a single token representation on which we apply softmax.
        - ln: *True/False* apply layer norm on the above token.
        - repr_size: *int* additional linear layer before classification, leave to 0 to disable.
      - multi: *True/False* whether to apply on all levels (*True*) or just on the last one.
      - use_lm_f: *True/False* whether to apply on the language tokens before fusion (*True*) or after fusion.
