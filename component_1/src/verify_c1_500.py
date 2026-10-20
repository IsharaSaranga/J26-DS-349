"""Independent validation of the C1 500-review run; does not run models.
From component_1/: python src/verify_c1_500.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT/'data'/'c1_real_review_input.csv'
PRED = ROOT/'results'/'inference'/'c1_review_aspect_predictions.csv'
STATUS = ROOT/'results'/'inference'/'c1_review_aspect_predictions_status.csv'
SHARED = ROOT/'results'/'inference'/'c1_shared_output.csv'
CONFIG = ROOT/'artifacts'/'aspect_b4'/'b4_inference_config.json'

for fp in [INPUT, PRED, STATUS, SHARED, CONFIG]:
    if not fp.is_file():
        raise FileNotFoundError(f'Missing required file: {fp}')

src = pd.read_csv(INPUT, dtype={'review_id':'string','product_id':'string'})
pred = pd.read_csv(PRED, dtype={'review_id':'string','product_id':'string'})
status = pd.read_csv(STATUS, dtype={'review_id':'string','product_id':'string'})
shared = pd.read_csv(SHARED, dtype={'review_id':'string','parent_asin':'string'})
conf = json.loads(CONFIG.read_text(encoding='utf-8'))

assert len(src)==500, f'Expected 500 input reviews, got {len(src)}'
assert src['review_id'].is_unique, 'Duplicate input review_id'
assert src['domain'].value_counts().to_dict()=={'phone':250, 'laptop':250}, 'Phone/laptop split changed'
assert src[['review_id','product_id','domain','timestamp','text']].notna().all().all()
assert pred[['review_id','product_id','domain','timestamp','text']].notna().all().all() if len(pred) else True
assert not pred.duplicated(['review_id','canonical_aspect']).any(), 'Duplicate review-aspect predictions'
assert not status.duplicated('review_id').any(), 'Duplicate status review IDs'
assert status.empty or status['status'].eq('no_aspect_detected').all(), 'Unexpected status code'

with_pred = set(pred['review_id'].astype(str))
without_pred = set(status['review_id'].astype(str))
all_input = set(src['review_id'].astype(str))
assert with_pred.isdisjoint(without_pred), 'An ID appears in predictions and status'
assert with_pred | without_pred == all_input, 'Not all input reviews accounted for'
assert len(with_pred) + len(without_pred) == 500
if len(pred):
    original = src.set_index('review_id')
    match = original.loc[pred['review_id']].reset_index(drop=True)
    for col in ('product_id','domain','timestamp','text'):
        assert pred[col].astype(str).reset_index(drop=True).equals(match[col].astype(str)), f'Original {col} differs'
    group_map = conf['semantic_group_by_aspect']
    assert pred['canonical_aspect'].map(group_map).eq(pred['semantic_group']).all(), 'Invalid taxonomy group'
    assert not ((pred.domain=='phone')&(pred.semantic_group=='LAPTOP_SPECIFIC')).any()
    assert not ((pred.domain=='laptop')&(pred.semantic_group=='PHONE_SPECIFIC')).any()
    pcols=['p_negative','p_neutral','p_positive']
    assert pred[pcols+['aspect_probability','sentiment_confidence']].notna().all().all()
    assert np.allclose(pred[pcols].sum(axis=1),1.0,atol=1e-4)
    assert np.allclose(pred['sentiment_score'],pred.p_positive-pred.p_negative,atol=1e-4)
    assert np.allclose(pred['sentiment_confidence'],pred[pcols].max(axis=1),atol=1e-4)
    assert pred['aspect_probability'].between(0,1).all()
    assert (pred['aspect_probability']+1e-7>=pred['aspect_threshold']).all()
    labels = np.array(['negative','neutral','positive'])
    assert (pred['predicted_sentiment'].to_numpy()==labels[np.argmax(pred[pcols].to_numpy(),axis=1)]).all()
assert len(shared)==len(pred), 'Shared row count differs from detailed output'
assert set(shared['review_id'].astype(str))==with_pred, 'Shared IDs differ from predictions'
if len(shared):
    expected = pred.assign(
        parent_asin=pred.product_id,
        review_date=pd.to_datetime(pred.timestamp,unit='ms',utc=True).dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    )
    key=['review_id','canonical_aspect']
    left=expected.set_index(key).sort_index()
    right=shared.set_index(key).sort_index()
    assert left.index.equals(right.index), 'Review-aspect pairs differ in shared output'
    for col in ['parent_asin','review_date','text','domain','semantic_group','predicted_sentiment']:
        assert left[col].astype(str).equals(right[col].astype(str)),f'Shared field mismatch: {col}'
    for col in ['aspect_probability','sentiment_score','sentiment_confidence']:
        assert np.allclose(left[col],right[col],atol=1e-5),f'Shared value mismatch: {col}'

print('C1 500-REVIEW OUTPUT CONTRACT: PASSED')
print('Input reviews:',len(src),'Phone:',(src.domain=='phone').sum(),'Laptop:',(src.domain=='laptop').sum())
print('Reviews with aspects:',len(with_pred),'No-aspect reviews:',len(without_pred))
print('Prediction aspect rows:',len(pred),'Shared aspect rows:',len(shared))
print('Predictions per domain:',pred.domain.value_counts().to_dict())
print('Aspects per semantic group:',pred.semantic_group.value_counts().to_dict())
print('Prediction sentiment counts:',pred.predicted_sentiment.value_counts().to_dict())
print('Output tables valid, but label accuracy/quality not established by this check.')
