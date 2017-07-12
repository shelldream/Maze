python main.py \
    --train_data_path="/export/sdb/shelldream/training_data_2016-04-01.txt" \
    --data_type="csv_with_schema" \
    --schema_file="fmap.schema.bak" \
    --model="xgboost" \
    --task="classification" \
    --parameter="{'n_estimators':100, 'max_depth':5, 'eta': 0.5, 'silent': 0, 'booster':'gbtree', 'objective': 'binary:logistic','eval_metric':'error','rate_drop':0.5, 'skip_rate':0.8, 'gamma': 0.1, 'max_delta_step':0, 'subsample':1, 'colsample_bytree':1, 'alpha':0, 'lambda':1, 'scale_pos_weight':1}" 
