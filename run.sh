# files=("Banana_v1")
files=("Banana_v1" "Banana_v2" "Bottle_v1" "Bottle_v2" "Orange_v1" "Sponge_v1" "Sponge_v2")
# conda init
# conda activate rlds_env
cd utils

for element in "${files[@]}"; do 

    python process_fisheye_hands_voice.py \
    --vrs "/mnt/c/Users/konst/OneDrive/Dokumente/ETH/Jahr 2025 - 2026/Mixed Reality/embodied-CoT-aria/aria_vrs/vrs_data_1/${element}/${element}.vrs" \
    --mps "/mnt/c/Users/konst/OneDrive/Dokumente/ETH/Jahr 2025 - 2026/Mixed Reality/embodied-CoT-aria/aria_vrs/vrs_data_1/${element}/mps_${element}_vrs/hand_tracking/hand_tracking_results.csv" \
    --mps_base "/mnt/c/Users/konst/OneDrive/Dokumente/ETH/Jahr 2025 - 2026/Mixed Reality/embodied-CoT-aria/aria_vrs/vrs_data_1/${element}/mps_${element}_vrs/" \
    --show_gaze \
    --output ./output/${element}_output
done