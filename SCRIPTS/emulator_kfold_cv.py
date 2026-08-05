#!/usr/bin/env python
"""
This script was created with Claude based on the architecture provided in notebook B_06_create_NN_emulator.ipynb.

K-fold cross-validation for the COSIPY surrogate.

Reuses the exact architecture, loss, optimizer and callbacks from
B_06_create_NN_emulator.ipynb. 

Inputs expected already in memory or rebuilt from the notebook preprocessing:
  X_all                : (N, 7)        standardized parameter vectors
  Xtime_sl_all         : (N, 58, 9)    param+time features for snowlines
  Xtime_alb_all        : (N, 98, 9)    param+time features for albedo
  y_mb_all             : (N, 1)
  y_tsl_all            : (N, 58)
  y_alb_all            : (N, 98)
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, root_mean_squared_error

K_FOLDS  = 10     
N_REPEATS = 1    # repeated K-fold
SEED = 11
EPOCHS = 300
BATCH = 32

def build_model():
    """Identical architecture to B_06_create_NN_emulator.ipynb."""
    mass_balance_input = Input(shape=(7,),     name="mass_balance_input")
    snowlines_input    = Input(shape=(58, 9),  name="snowlines_input")
    alb_input          = Input(shape=(98, 9),  name="alb_input")

    shared_mb = Dense(64, activation='relu')(mass_balance_input)
    shared_mb = Dense(128, activation='relu')(shared_mb)
    shared_mb = Dropout(0.1)(shared_mb)
    mb_branch = Dense(64, activation='relu')(shared_mb)
    mb_output = Dense(1, name="mass_balance_output")(mb_branch)

    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(snowlines_input)
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(lstm_out)
    shared_sl = Dense(128, activation='relu')(lstm_out)
    sl_branch = Dense(64, activation='relu')(shared_sl)
    snowlines_output = Dense(1, activation='sigmoid')(sl_branch)
    snowlines_output = Reshape((58,), name="snowlines_output")(snowlines_output)

    lstm_alb = Bidirectional(LSTM(64, return_sequences=True))(alb_input)
    lstm_alb = Bidirectional(LSTM(64, return_sequences=True))(lstm_alb)
    shared_alb = Dense(128, activation='relu')(lstm_alb)
    alb_branch = Dense(64, activation='relu')(shared_alb)
    alb_output = Dense(1, activation='sigmoid')(alb_branch)
    alb_output = Reshape((98,), name="alb_output")(alb_output)

    model = Model(inputs=[mass_balance_input, snowlines_input, alb_input],
                  outputs=[mb_output, snowlines_output, alb_output])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={'mass_balance_output': "mse", 'snowlines_output': "huber", 'alb_output': "huber"},
        loss_weights={'mass_balance_output': 1.0, 'snowlines_output': 1.0, 'alb_output': 1.0},
        metrics={'mass_balance_output': keras.metrics.RootMeanSquaredError(),
         'snowlines_output':    keras.metrics.RootMeanSquaredError(),
         'alb_output':          keras.metrics.RootMeanSquaredError()},
    )
    return model

def _expand_time(Xstd, tfeat, n):
    Xe = np.repeat(Xstd[:, None, :], n, axis=1)
    return np.concatenate([Xe, np.tile(tfeat, (Xstd.shape[0], 1, 1))], axis=-1)

def run_kfold(P, y_mb, y_tsl, y_alb, time_features, time_features_alb):
    n_sl, n_alb = len(time_features), len(time_features_alb)
    kf = RepeatedKFold(n_splits=K_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    rows = []
    sla_resid_near_zero = []
    for fold, (tr, te) in enumerate(kf.split(P)):
        # scale INSIDE the fold (fit on train rows only) -> no leakage
        sc = StandardScaler().fit(P[tr])
        Xtr, Xte = sc.transform(P[tr]), sc.transform(P[te])
        Xtr_sl,  Xte_sl  = _expand_time(Xtr, time_features, n_sl),     _expand_time(Xte, time_features, n_sl)
        Xtr_alb, Xte_alb = _expand_time(Xtr, time_features_alb, n_alb),_expand_time(Xte, time_features_alb, n_alb)

        model = build_model()
        es = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, verbose=2)
        lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6, verbose=2)
        model.fit(
            x={'mass_balance_input': Xtr, 'snowlines_input': Xtr_sl, 'alb_input': Xtr_alb},
            y={'mass_balance_output': y_mb[tr], 'snowlines_output': y_tsl[tr], 'alb_output': y_alb[tr]},
            validation_split=0.15,           # early-stopping val INSIDE the train fold
            epochs=EPOCHS, batch_size=BATCH, callbacks=[es, lr], verbose=2)

        pmb, ptsl, palb = model.predict(
            {'mass_balance_input': Xte, 'snowlines_input': Xte_sl, 'alb_input': Xte_alb}, verbose=2)
        # SLA residual behaviour near the snow-covered floor (obs <= 0.1):
        ot, pt = y_tsl[te].ravel(), ptsl.ravel()
        near0 = ot <= 0.1
        if near0.any():
            sla_resid_near_zero.append(float(np.mean(pt[near0] - ot[near0])))  # mean bias near floor
        rows.append(dict(
            fold=fold,
            r2_mb=r2_score(y_mb[te], pmb), rmse_mb=root_mean_squared_error(y_mb[te], pmb),
            r2_tsl=r2_score(ot, pt), rmse_tsl=root_mean_squared_error(ot, pt),
            r2_alb=r2_score(y_alb[te].ravel(), palb.ravel()), rmse_alb=root_mean_squared_error(y_alb[te].ravel(), palb.ravel()),
            sla_bias_near_floor=(float(np.mean(pt[near0]-ot[near0])) if near0.any() else np.nan),
        ))
        print(f"fold {fold}: R2_mb={rows[-1]['r2_mb']:.3f}  R2_tsl={rows[-1]['r2_tsl']:.3f}  R2_alb={rows[-1]['r2_alb']:.3f}")
        tf.keras.backend.clear_session()

    res = pd.DataFrame(rows)
    summary = res.drop(columns='fold').agg(['mean', 'std']).T
    print(f"\n===== {K_FOLDS}-fold x {N_REPEATS} repeats CV summary "
          f"({len(res)} fits, mean +/- std) =====")
    for m in ['r2_mb','r2_tsl','r2_alb','rmse_mb','rmse_tsl','rmse_alb']:
        print(f"  {m:9s}: {summary.loc[m,'mean']:.4f} +/- {summary.loc[m,'std']:.4f}")
    res.to_csv("emulator_kfold_results.csv", index=False)
    summary.to_csv("emulator_kfold_summary.csv")
    return res, summary

if __name__ == "__main__":
    raise SystemExit("Import run_kfold and call it with the notebook arrays (see __main__).")