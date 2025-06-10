# bin/bash
h5_file=/path/to/your/dcr/h5file.hdf5
data_idx=0
time_integrator=ssprk104  # {forward_euler, backward_euler, crank_nicolson, ssprk22, ssprk33, ssprk104}
python dcr.py --h5_file $h5_file --data_idx $data_idx --n_grid 64 --time_steps 1000 --time_integrator $time_integrator --save_dir results \
            --plot \
            # --plot_ref \
            # --show_fenicsx_log \
