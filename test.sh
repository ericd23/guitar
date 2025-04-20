#!/usr/bin/env bash
set -euo pipefail                                  # fail fast

LOG="timing_$(date +%Y%m%d_%H%M%S).log"
echo "Headless‑render timing run started $(date)" > "$LOG"
echo "-------------------------------------------------" >> "$LOG"

# ---------------------------------------------------------------------
#  full command list ---------------------------------------------------
# ---------------------------------------------------------------------
commands=(
  ## flight_of_the_bumblebee1
  "python main.py cfg/two_demo.py --note ./assets/notes/flight_of_the_bumblebee1.json --ckpt ./pretrained/flight_of_the_bumblebee1 --test --headless --record out_bumblebee_default.mp4 --width 1920 --height 1080 --fps 30 --capture-stride 1 --frames 200"
  "python main.py cfg/two_demo.py --note ./assets/notes/flight_of_the_bumblebee1.json --ckpt ./pretrained/flight_of_the_bumblebee1 --test --headless --record out_bumblebee_low_res.mp4 --width 960 --height 540 --fps 15 --capture-stride 1 --frames 200"
  "python main.py cfg/two_demo.py --note ./assets/notes/flight_of_the_bumblebee1.json --ckpt ./pretrained/flight_of_the_bumblebee1 --test --headless --record out_bumblebee_stride_4.mp4 --width 1920 --height 1080 --fps 30 --capture-stride 4 --frames 200"
  "python main.py cfg/two_demo.py --note ./assets/notes/flight_of_the_bumblebee1.json --ckpt ./pretrained/flight_of_the_bumblebee1 --test --headless --record out_bumblebee_low_res_stride_4.mp4 --width 960 --height 540 --fps 15 --capture-stride 4 --frames 200"

  ## prelude1_in_c_major_chords
  "python main.py cfg/two_demo.py --note ./assets/notes/prelude1_in_c_major_chords.json --ckpt ./pretrained/prelude1_in_c_major_chords --test --headless --record out_prelude1_default.mp4 --width 1920 --height 1080 --fps 30 --capture-stride 1 --frames 200"
  "python main.py cfg/two_demo.py --note ./assets/notes/prelude1_in_c_major_chords.json --ckpt ./pretrained/prelude1_in_c_major_chords --test --headless --record out_prelude1_low_res.mp4 --width 960 --height 540 --fps 15 --capture-stride 1 --frames 200"
  "python main.py cfg/two_demo.py --note ./assets/notes/prelude1_in_c_major_chords.json --ckpt ./pretrained/prelude1_in_c_major_chords --test --headless --record out_prelude1_stride_4.mp4 --width 1920 --height 1080 --fps 30 --capture-stride 4 --frames 200"
  "python main.py cfg/two_demo.py --note ./assets/notes/prelude1_in_c_major_chords.json --ckpt ./pretrained/prelude1_in_c_major_chords --test --headless --record out_prelude1_low_res_stride_4.mp4 --width 960 --height 540 --fps 15 --capture-stride 4 --frames 200"

  ## corcovado
  "python main.py cfg/two_demo.py --note ./assets/notes/corcovado.json --ckpt ./pretrained/corcovado --test --headless --record out_corcovado_default.mp4 --width 1920 --height 1080 --fps 30 --capture-stride 1 --frames 200"
  "python main.py cfg/two_demo.py --note ./assets/notes/corcovado.json --ckpt ./pretrained/corcovado --test --headless --record out_corcovado_low_res.mp4 --width 960 --height 540 --fps 15 --capture-stride 1 --frames 200"
  "python main.py cfg/two_demo.py --note ./assets/notes/corcovado.json --ckpt ./pretrained/corcovado --test --headless --record out_corcovado_stride_4.mp4 --width 1920 --height 1080 --fps 30 --capture-stride 4 --frames 200"
  "python main.py cfg/two_demo.py --note ./assets/notes/corcovado.json --ckpt ./pretrained/corcovado --test --headless --record out_corcovado_low_res_stride_4.mp4 --width 960 --height 540 --fps 15 --capture-stride 4 --frames 200"
)

# ---------------------------------------------------------------------
#  iterate and time ----------------------------------------------------
# ---------------------------------------------------------------------
for cmd in "${commands[@]}"; do
  echo -e "\n>> $cmd" | tee -a "$LOG"
  /usr/bin/time -p bash -c "$cmd" 2>&1 | tee -a "$LOG"
done

echo -e "\nAll runs completed $(date)" | tee -a "$LOG"