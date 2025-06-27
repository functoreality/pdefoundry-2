#!/bin/bash

# NOTE: This script downloads the data files that are used by
# https://github.com/functoreality/pdeformer-2/blob/main/configs/pretrain/model-L_small-data.yaml
# These are only a subset of the complete PDEFoundry-2 pretraining data.
# If you want to get the complete PDEFoundry-2 dataset directly, using only
# `pdefoundry2_test.sh` and `pdefoundry2_train.sh` is sufficient.

# dcr_base
wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/DiffConvecReac2D/dedalus_v5.1_DiffConvecReac2D_hom_cU1_k1e-03_0.01_seed1.hdf5
wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/DiffConvecReac2D/dedalus_v5.1_DiffConvecReac2D_hom_cU1_k1e-03_0.01_seed2.hdf5
# dcr_disk
wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/DiffConvecReac2D/dedalus_v5.1_DiffConvecReac2D_hom_disk_cU1_k1e-03_0.01_seed0.hdf5
wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/DiffConvecReac2D/dedalus_v5.1_DiffConvecReac2D_hom_disk_cU1_k1e-03_0.01_seed1.hdf5
# wave_npX
wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/Wave2D/dedalus_v5.1_Wave2D_hom_npX_sJ3_cU1_k1e-02_4_seed1.hdf5
wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/Wave2D/dedalus_v5.1_Wave2D_hom_npX_sJ3_cU1_k1e-02_4_seed2.hdf5
# mvdcr_2
wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/MCompn2D/dedalus_v5.1_MCompn2D_hom_nv2_cU1_k1e-03_0.01_seed1.hdf5
wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/MCompn2D/dedalus_v5.1_MCompn2D_hom_nv2_cU1_k1e-03_0.01_seed2.hdf5
