#!/bin/sh

for i in $(seq 1 40)
do
   rm "./portfolio_test_data/risk_model_test_portfolio2_ParEGO_${i}.h5"
done
