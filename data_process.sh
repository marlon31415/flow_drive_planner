###################################
# User Configuration Section
###################################
SPLIT="train" # specify the split to process ("train", "val", "val14", "interplan")

TRAIN_SET_PATH="/path/to/save/processed/data/$SPLIT" # specify a folder to save the processed training data (will be around 150GB for 1M training data)

###################################

if [ "$SPLIT" = "interplan" ]; then
    DATA_FOLDER="test"
else
    DATA_FOLDER="trainval"
fi
NUPLAN_DATA_PATH="$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/$DATA_FOLDER" # nuplan training data path (e.g., "/data/nuplan-v1.1/splits/trainval")
NUPLAN_MAP_PATH="$NUPLAN_DATA_ROOT/maps" # nuplan map path (e.g., "/data/nuplan-v1.1/maps")

if [ "$SPLIT" = "train" ]; then
    TOTAL_SCENARIOS=1000000
elif [ "$SPLIT" = "val" ]; then
    TOTAL_SCENARIOS=100000
elif [ "$SPLIT" = "val14" ]; then
    TOTAL_SCENARIOS=1118
elif [ "$SPLIT" = "interplan" ]; then
    TOTAL_SCENARIOS=16
else
    echo "Unsupported split: $SPLIT. Supported splits are: train, val, val14, interplan."
    exit 1
fi

python data_process.py \
--data_path $NUPLAN_DATA_PATH \
--map_path $NUPLAN_MAP_PATH \
--save_path $TRAIN_SET_PATH \
--total_scenarios $TOTAL_SCENARIOS \
--split $SPLIT \
