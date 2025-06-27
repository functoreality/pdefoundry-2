#!/bin/bash

echo "It is not recommended to execute this script directly, which will
download the entire dataset of about 40TB.  We strongly suggest that you read
and understand this script, and only execute the part you need, or write your
own customized command to achieve this."
exit 0

# dcr_base
for i in {2..111}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/DiffConvecReac2D/dedalus_v5.1_DiffConvecReac2D_hom_cU1_k1e-03_0.01_seed$i.hdf5
done

# dcr_npX
for i in {54..158}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/DiffConvecReac2D/dedalus_v5.1_DiffConvecReac2D_hom_npX_cU1_k1e-03_0.01_seed$i.hdf5
done

# dcr_npY
for i in {2..51}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/DiffConvecReac2D/dedalus_v5.1_DiffConvecReac2D_hom_npY_cU1_k1e-03_0.01_seed$i.hdf5
done
for i in {161..215}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/DiffConvecReac2D/dedalus_v5.1_DiffConvecReac2D_hom_npY_cU1_k1e-03_0.01_seed$i.hdf5
done

# dcr_disk
for i in {1..200}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/DiffConvecReac2D/dedalus_v5.1_DiffConvecReac2D_hom_disk_cU1_k1e-03_0.01_seed$i.hdf5
done

# dcr_sJ3
for i in {2..126}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/DiffConvecReac2D/dedalus_v5.1_DiffConvecReac2D_hom_sJ3_cU1_k1e-03_0.01_seed$i.hdf5
done

# dcr_inhom
for i in {2..101}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/DiffConvecReac2D/dedalus_v5.1_DiffConvecReac2D_inhom_cU1_k1e-03_0.1_seed$i.hdf5
done

# wave_base
for i in {2..126}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/Wave2D/dedalus_v5.1_Wave2D_hom_sJ3_cU1_k1e-02_4_seed$i.hdf5
done

# wave_npX
for i in {1..90}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/Wave2D/dedalus_v5.1_Wave2D_hom_npX_sJ3_cU1_k1e-02_4_seed$((i * 2 + 1)).hdf5
done

# wave_npY
for i in {2..91}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/Wave2D/dedalus_v5.1_Wave2D_hom_npY_sJ3_cU1_k1e-02_4_seed$((i * 2)).hdf5
done

# wave_disk
for i in {1..200}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/Wave2D/dedalus_v5.1_Wave2D_hom_disk_cU1_k1e-02_4_seed$i.hdf5
done

# wave_inhom
for i in {2..101}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/Wave2D/dedalus_v5.1_Wave2D_inhom_sJ3_cU1_k1e-02_4_seed$i.hdf5
done

# mvdcr_2
for i in {2..161}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/MCompn2D/dedalus_v5.1_MCompn2D_hom_nv2_cU1_k1e-03_0.01_seed$i.hdf5
done

# mvdcr_2_0
for i in {1..100}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/MCompn2D/dedalus_v5.1_MCompn2D_hom_fNL0_nv2_cU1_k1e-03_0.01_seed$i.hdf5
done

# mvdcr_3_1
for i in {1..200}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/MCompn2D/dedalus_v5.1_MCompn2D_hom_fNL1_nv3_cU1_k1e-03_0.01_seed$i.hdf5
done

# mvdcr_4_0
for i in {1..200}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/MCompn2D/dedalus_v5.1_MCompn2D_hom_fNL0_nv4_cU1_k1e-03_0.01_seed$i.hdf5
done

# dcdcr_icV
for i in {2..101}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/DivConstrDCR2D/dedalus_v5.1_DivConstrDCR2D_icV_hom_cU1_k1e-03_0.01_seed$i.hdf5
done

# dcdcr_icA
for i in {2..101}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/DivConstrDCR2D/dedalus_v5.1_DivConstrDCR2D_icA_hom_cU1_k1e-03_0.01_seed$i.hdf5
done

# mvwave_2
for i in {2..121}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/MCWave2D/dedalus_v5.1_MCWave2D_hom_nv2_cU1_k1e-02_4_seed$i.hdf5
done

# mvwave_3
for i in {2..101}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/MCWave2D/dedalus_v5.1_MCWave2D_hom_nv3_cU1_k1e-02_4_seed$i.hdf5
done

# mvwave_4
for i in {2..101}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/MCWave2D/dedalus_v5.1_MCWave2D_hom_nv4_cU1_k1e-02_4_seed$i.hdf5
done

# dcwave_icV
for i in {2..101}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/DivConstrWave2D/dedalus_v5.1_DivConstrWave2D_icV_hom_cU1_k1e-02_4_seed$i.hdf5
done

# dcwave_icA
for i in {2..101}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/DivConstrWave2D/dedalus_v5.1_DivConstrWave2D_icA_hom_cU1_k1e-02_4_seed$i.hdf5
done

# swe
for i in {1..100}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/SWE2D/dedalus_v5.1_SWE2D_hom_cU1_k1e-03_0.01_seed$i.hdf5
done

# elasticsteady
for i in {1..255}; do
  wget -c data-download.obs.cn-northeast-227.dlaicc.com/fenicsx_v2.1/ElasticSteady2D/fenicsx_v2.1_ElasticSteady2D_iso_cU1_k5e-01_2_nf3_ntf1_ftype1_scatFalse_N128_seed$i.hdf5
done
