


cd /net/spice/scratch/dkelley/future_ffdi/data/
cd ffdi_output/rcp2_6_joined
pwd
moo get moose:/adhoc/users/inika.taylor/fire/rcp2_6_joined/* .
cd ../rcp8_5_joined
moo get moose:/adhoc/users/inika.taylor/fire/rcp8_5_joined/* .

cd /scratch/dkelley/future_ffdi/data/threshold_exceedance/
cd rcp2_6
moo get moose:/adhoc/users/inika.taylor/fire/threshold_exceedance/rcp2_6/* .
cd ../rcp8_5
moo get moose:/adhoc/users/inika.taylor/fire/threshold_exceedance/rcp8_5/* .

cd /net/spice/scratch/dkelley/future_ffdi/data/GlobalWarmingLevels/
cd rcp2_6
moo get moose:/adhoc/users/inika.taylor/fire/GlobalWarmingLevels/rcp2_6/* .
cd ../rcp8_5
moo get moose:/adhoc/users/inika.taylor/fire/GlobalWarmingLevels/rcp8_5/* .

pymc_env
cd /home/h02/dkelley/future_ffdi
python calc_threshold_exceedance.py
python calc_consensus_plots.py

#cd GlobalWarmingLevels
#moo ls moose:/adhoc/users/inika.taylor/fire/

