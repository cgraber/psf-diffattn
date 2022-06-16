NUM_GPUS=4 #set based on how many gpus you have. We trained with 4

export DETECTRON2_SAVEDFEAT_WITHID='1'
export DETECTRON2_SAVEDFEAT_WITHDEPTHS='1'
export DETECTRON2_DATASETS='data/'
export DETECTRON2_SAVEDFEAT_FOLDER='data/saved_seq_feats/'
export CUDA_VISIBLE_DEVICES=1,2,4,5

############################
## STEP 1: extract feats ###
############################
working_dir='data/tmp_feats/'
mkdir -p $working_dir
python -u train_net.py --config-file configs/save_feats.yaml \
    --num-gpus 1 \
    --eval-only \
    OUTPUT_DIR $working_dir \
    DATASETS.TEST "('cityscapes_fine_instance_seg_val',)"
    
python -u train_net.py --config-file configs/save_feats.yaml \
    --num-gpus 1 \
    --eval-only \
    OUTPUT_DIR $working_dir \
    DATASETS.TEST "('cityscapes_fine_instance_seg_train',)"

# These scripts create saved features which need to be moved with the other information we provide
mv ${working_dir}*feats.h5 data/saved_seq_feats/



#############################
## STEP 2: train box model ##
#############################
working_dir='experiments/box_forecast/'
mkdir -p $working_dir
python -u train_net.py --config-file configs/forecast_box.yaml \
    --num-gpus $NUM_GPUS \
    --dist-url "tcp://127.0.0.1:12041" \
    OUTPUT_DIR $working_dir \
    DATALOADER.NUM_WORKERS 4 


####################################
## STEP 3: train appearance model ##
####################################
new_working_dir=${working_dir}with_appearance/
python -u train_net.py --config-file ${working_dir}config.yaml \
    --num-gpus $NUM_GPUS \
    --dist-url "tcp://127.0.0.1:12041" \
    MODEL.WEIGHTS ${working_dir}model_final.pth \
    TEST.LOSS_EVAL_PERIOD 2000 \
    OUTPUT_DIR $new_working_dir \
    SOLVER.IMS_PER_BATCH 8 \
    DATALOADER.NUM_WORKERS 4 \
    MODEL.SUB_BATCH_SIZE 1 \
    SOLVER.STEPS "(36000,)" \
    SOLVER.MAX_ITER 48000 \
    MODEL.MASK_FEAT_LOSS_COEF 10.0 \
    MODEL.USE_SEPARATE_MASK_MODEL True \
    MODEL.MASK_DECODER.ENC_TYPE feat_conv \
    MODEL.MASK_DECODER.DEC_TYPE feat_conv \
    MODEL.MASK_ENCODER.ENC_TYPE feat_conv \
    MODEL.MASK_ENCODER.TIME_SIZE 256 \
    MODEL.MASK_ENCODER.EMB_SIZE 256 \
    MODEL.MASK_DECODER.EMB_SIZE 256 \
    MODEL.MASK_ENCODER.DIM_FEEDFORWARD 512 \
    MODEL.MASK_DECODER.DIM_FEEDFORWARD 512 \
    MODEL.MASK_ENCODER.TRANSFORMER_TYPE conv_all_spatial \
    MODEL.MASK_DECODER.TRANSFORMER_TYPE conv_all_spatial \
    MODEL.MASK_MODEL_ONLY True \
    MODEL.MASK_ENCODER.IS_AGENT_AWARE True \
    MODEL.MASK_DECODER.IS_AGENT_AWARE True


###################################
## STEP 4: train refinement head ##
###################################
#note: not possible with data we've provided. here for reference only

#export DETECTRON2_CITYSCAPES_PANSEG_DIR='data/cityscapes/gtFine/'
#saved_bg_logit_path=empty
#saved_bg_depth_path=empty
#final_working_dir=${new_working_dir}with_refine/
#python -u train_net --config-file ${new_working_dir}config.yaml \
#    --num-gpus $NUM_GPUS \
#    MODEL.WEIGHTS ${new_working_dir}model_final.pth \
#    OUTPUT_DIR $final_working_dir \
#    SOLVER.STEPS "(18000,)" \
#    SOLVER.MAX_ITER 24000 \
#    MODEL.META_ARCHITECTURE Forecast2Track_RefineOnly \
#    INPUT.SAVED_BG_LOGIT_PATH $saved_bg_logit_path \
#    INPUT.SAVED_BG_DEPTH_DIR $saved_bg_depth_path \
#    INPUT.TARGET_FRAME_RANGE None \
#    MODEL.FGBGREFINE.NAME FGBGRefiner \
#    MODEL.FGBGREFINE.USE_SINGLE_BG_DEPTH True \
#    MODEL.ADD_FINAL_REFINE True \
#    MODEL.FGBGREFINE.USE_ID_MATCHING True \
#    MODEL.FGBGREFINE.TRAIN_PASTE_INTO_GT_BOX True \
#    MODEL.FGBGREFINE.TRAIN_GT_DEPTH True \
#    MODEL.FGBGREFINE.TRANSFORMER.SOFTMAX_FULL_MATRIX True \
#    SOLVER.CHECKPOINT_PERIOD 4000 \
#    MODEL.FGBGREFINE.USE_DEPTH_REFINE_MODEL True \
#    MODEL.FGBGREFINE.DYN_HEAD.STATIC_EMB_SIZE 128 \
#    MODEL.FGBGREFINE.USE_DEPTH_OFFSET_BIAS True \
#    MODEL.FGBGREFINE.DEPTH_OFFSET_COEF 0.001 \
#    MODEL.FGBGREFINE.USE_GT_PAN_SEG True \
#    MODEL.FGBGREFINE.PASTE_FULL_REZ_INSTANCES True \

