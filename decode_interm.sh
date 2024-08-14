python decode_interm_result.py --dataset DiffusionDB --evade_method corrupters --arch contrast
python decode_interm_result.py --dataset DiffusionDB --evade_method corrupters --arch gaussian_noise
python decode_interm_result.py --dataset DiffusionDB --evade_method corrupters --arch jpeg
python decode_interm_result.py --dataset DiffusionDB --evade_method diffpure --arch dummy
python decode_interm_result.py --dataset DiffusionDB --evade_method diffuser --arch dummy
python decode_interm_result.py --dataset DiffusionDB --evade_method dip --arch vanila
python decode_interm_result.py --dataset DiffusionDB --evade_method vae --arch cheng2020-anchor