# %% [markdown]
# # Training and Evaluation in one Notebook for One Model-Database Pair

# %%
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU found")
    print("Memory growth set")
else:
    print("No GPU found")

# %%
import arrow
import socket
from sqlalchemy.orm import Session
from tqdm.notebook import tqdm

from april.anomalydetection import *
from april.database import EventLog
from april.database import Model
from april.database import get_engine
from april.dataset import Dataset
from april.fs import DATE_FORMAT
from april.fs import get_event_log_files

import itertools

from sklearn import metrics


from april.anomalydetection import BINet
from april.anomalydetection.utils import label_collapse
from april.database import Evaluation
from april.bpic12evaluator import Evaluator
from april.fs import get_model_files
from april.fs import PLOT_DIR

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

import pandas as pd
import seaborn as sns
from sqlalchemy.orm import Session
import scikit_posthocs as sp

from april.database import get_engine
from april.fs import PLOT_DIR
from april.utils import microsoft_colors, prettify_dataframe, cd_plot, get_cd
from april.enums import Base, Strategy, Heuristic

sns.set_style('white')
pd.set_option('display.max_rows', 50)
# %config InlineBackend.figure_format = 'retina'


for ltn_epochs in range(1, 5):
    print(f"Running for {ltn_epochs} epochs")
    # %%
    dataset = "bpic12-0.3-1"
    out_dir = PLOT_DIR / f'bpic12_evaluations_{ltn_epochs}_epochs_{arrow.now().format("YYYY-MM-DD-HH-mm-ss")}'
    eval_file = out_dir / 'bpic12_fraction_evaluations.pkl'
    csv_file = out_dir / 'bpic12_fraction_evaluations.csv'
    excel_file = out_dir / 'bpic12_fraction_evaluations.xlsx'
    model_folder = r"D:\LTNcoder\.out\models"
    db = r"D:\LTNcoder\.out\april.db"

    # code to create out_dir if it does not exist
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {out_dir}")
    from april.utils import delete_all_files_in_folder, delete_evaluation_and_model_tables
    delete_all_files_in_folder(model_folder)
    delete_evaluation_and_model_tables(db)


    # %% [markdown]
    # # Training

    # %%
    def fit_and_save(dataset_name, ad, ad_kwargs=None, fit_kwargs=None):
        if ad_kwargs is None:
            ad_kwargs = {}
        if fit_kwargs is None:
            fit_kwargs = {}

        # Save start time
        start_time = arrow.now()

        # Dataset
        dataset = Dataset(dataset_name)

        # AD
        ad = ad(**ad_kwargs)

        # Train and save
        ad.fit(dataset, **fit_kwargs)
        file_name = f'{dataset_name}_{ad.abbreviation}_{start_time.format(DATE_FORMAT)}'
        model_file = ad.save(file_name)

        # Save end time
        end_time = arrow.now()

        # Cache result
        Evaluator(model_file.str_path).cache_result()

        # Calculate training time in seconds
        training_time = (end_time - start_time).total_seconds()

        # Write to database
        engine = get_engine()
        session = Session(engine)

        session.add(Model(creation_date=end_time.datetime,
                        algorithm=ad.name,
                        training_duration=training_time,
                        file_name=model_file.file,
                        training_event_log_id=EventLog.get_id_by_name(dataset_name),
                        training_host=socket.gethostname(),
                        hyperparameters=str(dict(**ad_kwargs, **fit_kwargs))))
        session.commit()
        session.close()

        if isinstance(ad, NNAnomalyDetector):
            from keras.backend import clear_session
            clear_session()
        pass

    # %%
    # ads = [
    #     dict(ad=Bpic12DAE, fit_kwargs=dict(epochs=1, batch_size=100)),
    #     dict(ad=Bpic12LTN, fit_kwargs=dict(epochs=1, batch_size=100, epochs_ltn=1)),
    #     dict(ad=Bpic12LTNFROZEN, fit_kwargs=dict(epochs=1, batch_size=100, epochs_ltn=1)),    
    # ]
    ads = [
        dict(ad=Bpic12DAE, fit_kwargs=dict(epochs=8, batch_size=100)),
        dict(ad=Bpic12LTN, fit_kwargs=dict(epochs=8, batch_size=100, epochs_ltn=ltn_epochs)),
        dict(ad=Bpic12LTNFROZEN, fit_kwargs=dict(epochs=8, batch_size=100, epochs_ltn=ltn_epochs))
        ] + [dict(ad=LTN_ROW_CLASS, fit_kwargs=dict(epochs=8, batch_size=100, epochs_ltn=ltn_epochs))
        for LTN_ROW_CLASS in ltn_row_classes]
    for ad in tqdm(ads, desc="Fitting ADs"):
        fit_and_save(dataset, **ad)


    # %%
    print(AD)

    # %% [markdown]
    # # Evaluation

    # %%
    heuristics = [h for h in Heuristic.keys() if h not in [Heuristic.DEFAULT, Heuristic.MANUAL, Heuristic.RATIO,
                                                        Heuristic.MEDIAN, Heuristic.MEAN]]
    params = [(Base.SCORES, Heuristic.DEFAULT, Strategy.SINGLE), *itertools.product([Base.SCORES], heuristics, Strategy.keys())]

    # %%
    def _evaluate(params):
        # print(f"Evaluating {params}...")
        e, base, heuristic, strategy = params

        session = Session(get_engine())
        model = session.query(Model).filter_by(file_name=e.model_file.name).first()
        session.close()
        # print(f"Model {model} loaded from Session.")

        if Model is None:
            print(f"Models from session: {model}")
        # Generate evaluation frames
        y_pred = e.binarizer.binarize(base=base, heuristic=heuristic, strategy=strategy, go_backwards=False)
        y_true = e.binarizer.get_targets()

        evaluations = []
        for axis in [0, 1, 2]:
            # print(f"Evaluating axis {axis}...")
            for i, attribute_name in enumerate(e.dataset.attribute_keys):
                # print(f"Evaluating attribute {attribute_name}...")
                def get_evaluation(label, precision, recall, f1):
                    return Evaluation(model_id=model.id, file_name=model.file_name,
                                    label=label, perspective=perspective, attribute_name=attribute_name,
                                    axis=axis, base=base, heuristic=heuristic, strategy=strategy,
                                    precision=precision, recall=recall, f1=f1)

                perspective = 'Control Flow' if i == 0 else 'Data'
                if i > 0 and not e.ad_.supports_attributes:
                    # print(f"Skipping attribute {attribute_name} for model {model} as it does not support attributes.")
                    evaluations.append(get_evaluation('Normal', 0.0, 0.0, 0.0))
                    evaluations.append(get_evaluation('Anomaly', 0.0, 0.0, 0.0))
                else:
                    # print(f"Evaluating attribute {attribute_name} for model {model}...")
                    yp = label_collapse(y_pred[:, :, i:i + 1], axis=axis).compressed()
                    yt = label_collapse(y_true[:, :, i:i + 1], axis=axis).compressed()
                    p, r, f, _ = metrics.precision_recall_fscore_support(yt, yp, labels=[0, 1])
                    evaluations.append(get_evaluation('Normal', p[0], r[0], f[0]))
                    evaluations.append(get_evaluation('Anomaly', p[1], r[1], f[1]))

        return evaluations

    def evaluate(model_name):
        # print(f"Evaluating {model_name}...")
        e = Evaluator(model_name)
        # print(f"{e} loaded.")

        _params = []
        for base, heuristic, strategy in params:
            if e.dataset.num_attributes == 1 and strategy in [Strategy.ATTRIBUTE, Strategy.POSITION_ATTRIBUTE]:
                continue
            if isinstance(e.ad_, BINet) and e.ad_.version == 0:
                continue
            if heuristic is not None and heuristic not in e.ad_.supported_heuristics:
                continue
            if strategy is not None and strategy not in e.ad_.supported_strategies:
                continue
            if base is not None and base not in e.ad_.supported_bases:
                continue
            _params.append([e, base, heuristic, strategy])

        return [_e for p in _params for _e in _evaluate(p)]

    # %%
    models = sorted([m.name for m in get_model_files() if m.p == 0.3])# and 'real' in m.name])
    print(f"Available Models: {models}")
    evaluations = []
    for model in tqdm(models, desc='Evaluate'):
        e = evaluate(model)
        evaluations.append(e)
    # Write to database
    session = Session(get_engine())
    for e in evaluations:
        session.bulk_save_objects(e)
        session.commit()
    session.close()

    # %%

    session = Session(get_engine())
    evaluations = session.query(Evaluation).all()
    rows = []

    for ev in tqdm(evaluations):
        # print(f"Evaluation: {ev}")
        m = ev.model
        # print(f"Model: {m}")
        el = ev.model.training_event_log
        # print(f"Event log: {el}")
        rows.append([m.file_name, m.creation_date, m.hyperparameters, m.training_duration, m.training_host, m.algorithm, 
                    el.name, el.base_name, el.percent_anomalies, el.number,
                    ev.axis, ev.base, ev.heuristic, ev.strategy, ev.label, ev.attribute_name, ev.perspective, ev.precision, ev.recall, ev.f1])
    session.close()
    columns = ['file_name', 'date', 'hyperparameters', 'training_duration', 'training_host', 'ad',
            'dataset_name', 'process_model', 'noise', 'dataset_id',
            'axis', 'base', 'heuristic', 'strategy', 'label', 'attribute_name', 'perspective', 'precision', 'recall', 'f1']
    evaluation = pd.DataFrame(rows, columns=columns)

    evaluation.to_pickle(eval_file)

    # %%
    synth_datasets = ['paper', 'p2p', 'small', 'medium', 'large', 'huge', 'gigantic', 'wide']
    bpic_datasets = ['bpic12', 'bpic13', 'bpic15', 'bpic17']
    anonymous_datasets = ['real']
    datasets = synth_datasets + bpic_datasets + anonymous_datasets
    dataset_types = ['Synthetic', 'Real-life']

    h_ads = ads = [ad['ad'].__name__ for ad in ads]

    heuristics = [r'$best$', r'$default$', r'$elbow_\downarrow$', r'$elbow_\uparrow$', 
                r'$lp_\leftarrow$', r'$lp_\leftrightarrow$', r'$lp_\rightarrow$']
    print(ads)

    # %%
    evaluation = evaluation.query(f'ad in {ads} and label == "Anomaly"')

    # %%
    evaluation['perspective-label'] = evaluation['perspective'] + '-' + evaluation['label']
    evaluation['attribute_name-label'] = evaluation['attribute_name'] + '-' + evaluation['label']
    evaluation['dataset_type'] = 'Synthetic'
    evaluation.loc[evaluation['process_model'].str.contains('bpic'), 'dataset_type'] = 'Real-life'
    evaluation.loc[evaluation['process_model'].str.contains('real'), 'dataset_type'] = 'Real-life'

    # %%
    _filtered_evaluation = evaluation.query(f'ad in {h_ads} and (strategy == "{Strategy.ATTRIBUTE}"'
                                        f' or (strategy == "{Strategy.SINGLE}" and process_model == "bpic12")'
                                        f' or (strategy == "{Strategy.SINGLE}" and ad == "Naive+"))')

    # %%
    filtered_evaluation = _filtered_evaluation.query(f'heuristic == "{Heuristic.DEFAULT}"'
                                                    f' or (heuristic == "{Heuristic.LP_MEAN}" and ad not in {h_ads})'
                                                    f' or (heuristic == "{Heuristic.LP_LEFT}" and ad in {h_ads})'
                                                    )

    # %%
    df = filtered_evaluation.query('axis == 0')
    df = prettify_dataframe(df)
    df = df.groupby(['axis', 'process_model', 'dataset_name', 'ad', 'file_name', 'perspective'])[['precision', 'recall', 'f1']].mean().reset_index()
    df = df.groupby(['axis', 'process_model', 'dataset_name', 'ad', 'file_name'])[['precision', 'recall', 'f1']].mean().reset_index()
    df['f1'] = 2 * df['recall'] * df['precision'] / (df['recall'] + df['precision'])

    df = pd.pivot_table(df, index=['axis', 'ad'], columns=['process_model', 'dataset_name'], values=['precision', 'recall', 'f1'])
    df = df.fillna(0)
    df = df.stack(1).stack(1).reset_index()
    df.to_excel(str(out_dir / 'table.xlsx'), index=False)

    # drop rows in column "axis" which have value "Attribute"
    df = df.query('axis != "Attribute"')

    # df = pd.pivot_table(df, index=['axis', 'ad'], columns=['process_model'], values=['precision', 'recall', 'f1'], aggfunc=np.mean)

    df.to_excel(str(excel_file), index=False)
    df.to_csv(str(csv_file), index=False)
    print(df)

    # %%
    # display(df)


