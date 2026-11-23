"""Track 1 score 파일을 주최측 evaluation-package(고정 commit)로 다시 채점.

eval 비교표에는 dev 결과를 넣지 않는다. 공개 시스템 수치는 참고용이며 순위 비교가 아니다.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment_dashboard import (COMBOS, ARCH, tag, read_json, write_json,
                                  verify_score_coverage, save_figure, plot_detection)

OFFICIAL_COMMIT = 'fe23d3053a0889b305d98c9e4cadf418c213d972'
OFFICIAL_REPO = 'https://github.com/asvspoof-challenge/asvspoof5.git'
RESULT_SOURCE = 'https://www.eurecom.fr/publication/7829/download/sec-publi-7829.pdf'
ARCH_SOURCE = 'https://arxiv.org/html/2601.03944v3#S4.T3'
REFERENCES = [
    dict(model='T45', backbone='wav2vec2-large', backend='GAT + MFA-Res2Net + LSTM', fusion=6, min_dcf=.0750, eer_pct=2.59, act_dcf=1.0000, cllr=.7923),
    dict(model='T36', backbone='WavLM-Base', backend='MLP', fusion=5, min_dcf=.0936, eer_pct=3.41, act_dcf=1.0000, cllr=.8874),
    dict(model='T27', backbone='WavLM-Base', backend='MHFA + WAP', fusion=3, min_dcf=.0937, eer_pct=3.42, act_dcf=.1375, cllr=.1927),
    dict(model='T23', backbone='LFCC + wav2vec2-large', backend='LCNN + GNN + Conformer', fusion=3, min_dcf=.1124, eer_pct=4.16, act_dcf=1.0000, cllr=.9179),
    dict(model='T43', backbone='wav2vec2-large', backend='AASIST', fusion=2, min_dcf=.1149, eer_pct=4.04, act_dcf=.5729, cllr=.9562),
]


def official_package(repo):
    repo = Path(repo)
    if not (repo/'.git').exists():
        subprocess.run(['git', 'clone', '--depth', '1', OFFICIAL_REPO, str(repo)], check=True)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != OFFICIAL_COMMIT:
        subprocess.run(['git', '-C', str(repo), 'fetch', '--depth', '1', 'origin', OFFICIAL_COMMIT], check=True)
        subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', OFFICIAL_COMMIT], check=True)
    return repo/'evaluation-package'


def score_split(output, combo, split, package):
    import numpy as np
    import pandas as pd
    count = {'dev': 140950, 'eval': 680774}[split]
    score_path = output/'scores'/f'{split}_{tag(combo)}.tsv'
    protocol = output/'protocols'/f'ASVspoof5.{split}.track_1.tsv'
    summary = read_json(score_path.with_name(score_path.stem+'_summary.json'), {})
    if not score_path.exists() or not summary:
        return None
    if summary.get('metric_mode') != 'official' or summary.get('sliding'):
        raise ValueError('Expected larger-is-bonafide scores and no sliding TTA')
    coverage = verify_score_coverage(score_path, protocol, count)
    score_sha = hashlib.sha256(score_path.read_bytes()).hexdigest()
    saved = output/'reports'/f'official_{split}_{combo}.json'
    previous = read_json(saved, {})
    if previous.get('score_sha256') == score_sha and previous.get('official_commit') == OFFICIAL_COMMIT:
        return previous
    scores = pd.read_csv(score_path, sep='\t', header=None, names=['filename','cm-score','label'])
    keys = pd.read_csv(protocol, sep=r'\s+', header=None, usecols=[1,8], names=['filename','cm-label'], keep_default_na=False)
    if not set(keys['cm-label']).issubset({'bonafide','spoof'}):
        raise ValueError('Official key labels are hidden; numerical eval is unavailable')
    destination = output/'scores/official'/combo/split
    destination.mkdir(parents=True, exist_ok=True)
    score_file, key_file = destination/'cm_scores.tsv', destination/'cm_keys.tsv'
    scores[['filename','cm-score']].to_csv(score_file, sep='\t', index=False)
    keys.to_csv(key_file, sep='\t', index=False)
    # 실제 CLI 옵션은 --cm_keys (README에는 예전 표기 --cm_key)
    command = [sys.executable, str(package/'evaluation.py'), '--m', 't1',
               '--cm', str(score_file), '--cm_keys', str(key_file)]
    result = subprocess.run(command, cwd=destination, check=True, capture_output=True, text=True)
    (destination/'official_stdout.txt').write_text(result.stdout+result.stderr)
    sys.path.insert(0, str(package))
    util = importlib.import_module('util')
    metrics = importlib.import_module('calculate_metrics')
    values, labels = util.load_cm_scores_keys(str(score_file), str(key_file))
    md, eer, cllr, ad = metrics.calculate_minDCF_EER_CLLR_actDCF(values, labels, str(destination/'track1_result.txt'), printout=False)
    values = [float(v) for v in (md, eer, cllr, ad)]
    if not np.isfinite(values).all():
        raise ValueError('Non-finite official metrics')
    report = dict(combo=combo, architecture=ARCH[combo], split=split, rows=count,
                  min_dcf=values[0], eer_pct=100*values[1], cllr=values[2], act_dcf=values[3],
                  official_commit=OFFICIAL_COMMIT, official_repository=OFFICIAL_REPO,
                  score_sha256=score_sha, command=command, coverage=coverage,
                  inference=dict(single_segment=True, sliding=False, limit=0), updated=time.time())
    write_json(saved, report)
    return report


def publish_comparison(output, evaluations):
    rows = [dict(origin='published challenge ensemble', condition='open', **r) for r in REFERENCES]
    rows += [dict(origin='this project; train 50%; single model', condition='open',
                  model=r['combo'].upper(), backbone='WavLM-Base', backend=ARCH[r['combo']], fusion=1,
                  **{k:r[k] for k in ('min_dcf','eer_pct','act_dcf','cllr')}) for r in evaluations]
    destination = output/'reports'
    with (destination/'official_eval_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    text = ['# Track 1 official eval comparison', '',
            'Same released eval partition; minDCF is primary. This project uses 50% of train and single models; published top systems are ensembles.',
            'Dev metrics are excluded. No official challenge rank is claimed.', '',
            '| System | Backbone | Backend | Models fused | minDCF | EER (%) | actDCF | Cllr |',
            '|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        text.append(f"| {row['model']} | {row['backbone']} | {row['backend']} | {row['fusion']} | {row['min_dcf']:.4f} | {row['eer_pct']:.2f} | {row['act_dcf']:.4f} | {row['cllr']:.4f} |")
    for combo in COMBOS:
        if combo not in {r['combo'] for r in evaluations}:
            text.append(f'| {combo.upper()} | WavLM-Base | {ARCH[combo]} | 1 | pending | pending | pending | pending |')
    if evaluations:
        text += ['', '## Difference from published WavLM-Base ensembles', '',
                 'Positive differences mean higher error/cost. EER differences are percentage points.', '',
                 '| Project model | Reference | minDCF difference | EER difference (pp) |',
                 '|---|---|---:|---:|']
        for result in evaluations:
            for reference in [r for r in REFERENCES if r['model'] in ('T36','T27')]:
                text.append(f"| {result['combo'].upper()} | {reference['model']} | {result['min_dcf']-reference['min_dcf']:+.4f} | {result['eer_pct']-reference['eer_pct']:+.2f} |")
        text += ['', '## Score calibration and difficult eval groups', '']
        for result in evaluations:
            text.append(f"- {result['combo'].upper()}: raw actDCF - minDCF = {result['act_dcf']-result['min_dcf']:.4f}; raw Cllr = {result['cllr']:.4f} bits.")
            group_rows = read_json(output/'reports'/f'eval_groups_{result["combo"]}.json',{}).get('rows',[])
            for kind in ('attack','codec'):
                difficult = sorted([r for r in group_rows if r['group_type']==kind],key=lambda r:r['min_dcf'],reverse=True)[:3]
                if difficult:
                    text.append('  Difficult '+kind+' groups by minDCF: '+', '.join(f"{r['group']} ({r['min_dcf']:.4f}, EER {r['eer_pct']:.2f}%)" for r in difficult)+'.')
    text += ['', 'Results: '+RESULT_SOURCE, 'Architectures: '+ARCH_SOURCE,
             'Official evaluation code commit: '+OFFICIAL_COMMIT]
    (destination/'official_eval_comparison.md').write_text('\n'.join(text)+'\n')
    return rows


def visualize(output, evaluations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = REFERENCES + [dict(model=r['combo'].upper(), min_dcf=r['min_dcf'], eer_pct=r['eer_pct']) for r in evaluations]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    labels = [r['model'] for r in rows]
    colors = ['#879cb6']*len(REFERENCES) + ['#e1812c']*len(evaluations)
    for ax, metric, title in zip(axes, ['min_dcf','eer_pct'], ['Official minDCF (lower is better)', 'Official EER (%)']):
        bars = ax.bar(labels, [r[metric] for r in rows], color=colors)
        ax.bar_label(bars, fmt='%.4f' if metric=='min_dcf' else '%.2f', padding=3, fontsize=9)
        ax.set_title(title); ax.set_ylim(0, max(r[metric] for r in rows)*1.2)
        ax.grid(axis='y', alpha=.2)
    pending = ', '.join(c.upper() for c in COMBOS if c not in {r['combo'] for r in evaluations})
    fig.suptitle('Track 1 EVAL only: published ensembles (blue) vs this project single models / train 50% (orange)\n'
                 + ('Pending: '+pending if pending else 'All 680,774 trial IDs verified for each model'), fontsize=10)
    save_figure(fig, output/'figures/official_eval_comparison.png')
    plt.close(fig)
    if evaluations:
        plot_detection(output, 'eval')
        import pandas as pd
        fig, panels = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        completed = {r['combo']: r for r in evaluations}
        for panel, combo in zip(panels.flat, COMBOS):
            if combo not in completed:
                panel.text(.5, .5, combo.upper()+' EVAL pending', ha='center', va='center', transform=panel.transAxes)
                panel.set_axis_off()
                continue
            scores = pd.read_csv(output/'scores'/f'eval_{tag(combo)}.tsv',sep='\t',header=None,
                                 names=['filename','score','label'])
            for label, name, color in [(0,'bonafide','#4677a6'),(1,'spoof','#e1812c')]:
                panel.hist(scores.loc[scores.label==label,'score'],bins=60,density=True,
                           alpha=.55,label=name,color=color)
            panel.set_title(combo.upper()+' - all 680,774 EVAL trials')
            panel.set_xlabel('Raw score (higher = bonafide)'); panel.set_ylabel('Density')
            panel.legend(); panel.grid(alpha=.15)
        fig.suptitle('Official EVAL score distributions by model; no eval calibration applied')
        save_figure(fig,output/'figures/eval_score_distributions.png'); plt.close(fig)


def group_analysis(output, result, package):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    combo = result['combo']
    saved = output/'reports'/f'eval_groups_{combo}.json'
    if read_json(saved, {}).get('score_sha256') == result['score_sha256']: return
    sys.path.insert(0, str(package))
    official = importlib.import_module('calculate_metrics')
    scores = pd.read_csv(output/'scores'/f'eval_{tag(combo)}.tsv', sep='\t', header=None,
                         names=['filename','score','label'])
    metadata = pd.read_csv(output/'protocols/ASVspoof5.eval.track_1.tsv', sep=r'\s+', header=None,
                           usecols=[1,3,7,8], names=['filename','codec','attack','key'], keep_default_na=False)
    frame = scores.merge(metadata, on='filename', validate='one_to_one')
    rows = []
    for kind in ['attack','codec']:
        values = sorted(frame.loc[frame.key=='spoof', kind].unique()) if kind=='attack' else sorted(frame[kind].unique())
        for value in values:
            subset = frame[((frame[kind]==value)|(frame.key=='bonafide')) if kind=='attack' else frame[kind]==value]
            nb, ns = int((subset.key=='bonafide').sum()), int((subset.key=='spoof').sum())
            if not nb or not ns: continue
            md, eer, cllr, ad = official.calculate_minDCF_EER_CLLR_actDCF(subset.score.to_numpy(), subset.key.to_numpy(), 'unused', printout=False)
            rows.append(dict(group_type=kind, group=value, bonafide=nb, spoof=ns,
                             min_dcf=float(md), eer_pct=100*float(eer), cllr=float(cllr), act_dcf=float(ad)))
        subset_rows = [r for r in rows if r['group_type']==kind]
        if not subset_rows: continue
        fig, axes = plt.subplots(2, 1, figsize=(max(10, .6*len(subset_rows)),7), constrained_layout=True)
        for ax, metric in zip(axes, ['min_dcf','eer_pct']):
            ax.bar([r['group'] for r in subset_rows], [r[metric] for r in subset_rows], color='#4677a6')
            ax.set_ylabel('minDCF' if metric=='min_dcf' else 'EER (%)'); ax.grid(axis='y',alpha=.2)
            ax.tick_params(axis='x', rotation=45)
        fig.suptitle(f'{combo.upper()} - official EVAL by {kind}\n'
                     + ('Each attack vs ALL eval bonafide trials' if kind=='attack' else 'Bonafide and spoof within the SAME codec'))
        save_figure(fig, output/'figures'/f'eval_{kind}_{combo}.png'); plt.close(fig)
    pd.DataFrame(rows).to_csv(output/'reports'/f'eval_groups_{combo}.csv', index=False)
    write_json(saved, dict(score_sha256=result['score_sha256'], rows=rows,
                           attack_reference='all eval bonafide', codec_reference='same-codec bonafide'))


def analysis_notebook(output, evaluations):
    """그림이 포함된 결과 노트북 생성 (GPU 없이 열람용)."""
    cells = []
    def markdown(identifier, text):
        cells.append(dict(cell_type='markdown', id=identifier, metadata={'id':identifier}, source=text.splitlines(True)))
    markdown('eval-summary', '# ASVspoof5 Track 1 - final EVAL analysis\n\n'
             'Evaluation partition: **680,774 trials per model**, organizer scoring package. '
             'No dev result appears in this comparison.\n\n'
             'A1=BiLSTM adapter, A2=SE-Res2Net adapter; C1=LCNN-big classifier, C2=SENet12 classifier.\n\n'
             f'Completed eval models: {", ".join(r["combo"].upper() for r in evaluations) or "none yet"}. '
             'Missing models remain pending. Generated UTC epoch: '+str(time.time())+'\n')
    markdown('eval-table', (output/'reports/official_eval_comparison.md').read_text())
    paths = [output/'figures/official_eval_comparison.png', output/'figures/eval_detection.png',
             output/'figures/eval_score_distributions.png']
    for result in evaluations:
        paths += [output/'figures'/f'eval_attack_{result["combo"]}.png', output/'figures'/f'eval_codec_{result["combo"]}.png']
    for index, path in enumerate(paths):
        if not path.exists(): continue
        identifier = 'eval-figure-'+str(index)
        markdown(identifier+'-title', '### '+path.stem+'\n')
        cells.append(dict(cell_type='code', id=identifier, metadata={'id':identifier}, execution_count=None,
                          source=[f'# Saved actual EVAL figure: {path.name}\n'],
                          outputs=[dict(output_type='display_data', metadata={'image/png':{'width':900}},
                                        data={'image/png':base64.b64encode(path.read_bytes()).decode(), 'text/plain':[path.name]})]))
    markdown('eval-notes', 'The project uses 50% train and individual models. Reference top systems fuse 2-6 models. '
             'These are post-challenge results and do not establish an official leaderboard rank. '
             'Attack plots use all eval bonafide trials as the reference; codec plots compare within the same codec. '
             'Raw actDCF/CLLR reflect score calibration as well as discrimination.\n')
    notebook = dict(nbformat=4, nbformat_minor=5, metadata={'kernelspec':{'name':'python3','display_name':'Python 3'},
                    'language_info':{'name':'python'}, 'asvspoof_partition':'eval', 'eval_trials_per_model':680774}, cells=cells)
    path = output.parent.parent/'notebooks/results/ASVspoof5_eval_results.ipynb'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f: json.dump(notebook,f,ensure_ascii=False,indent=1)
    if len(evaluations) == len(COMBOS):
        # 결과 파일만 zip으로 묶음 (가중치, 음성, dev 결과 제외)
        import zipfile
        destination = output/'reports/final_eval_analysis.zip'
        temporary = destination.with_suffix('.zip.tmp')
        report_files = [output/'reports/official_eval_comparison.csv',
                        output/'reports/official_eval_comparison.md',
                        output/'reports/official_package_manifest.json']
        for result in evaluations:
            combo = result['combo']
            report_files += [output/'reports'/f'official_eval_{combo}.json',
                             output/'reports'/f'coverage_eval_{combo}.json',
                             output/'reports'/f'eval_groups_{combo}.json',
                             output/'reports'/f'eval_groups_{combo}.csv']
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
            for artifact in [*report_files,*paths]:
                if artifact.exists(): bundle.write(artifact,str(artifact.relative_to(output)))
            bundle.write(path,'notebooks/results/'+path.name)
        temporary.replace(destination)
    return path


def publish_cloud(output, evaluations):
    """주최측 지표를 조합별 W&B run summary에 추가."""
    saved_path = output/'reports/official_wandb_publication.json'
    published = read_json(saved_path, {})
    links = read_json(output/'wandb_sync/links.json', {})
    api = None
    for result in evaluations:
        combo = result['combo']
        if published.get(combo) == result['score_sha256']: continue
        url = links.get(combo, {}).get('url')
        if not url: continue
        if api is None:
            import wandb
            api = wandb.Api(timeout=30)
        run = api.run(url.split('wandb.ai/',1)[1].replace('/runs/','/'))
        run.summary.update({**{f'official_eval/{k}': result[k] for k in
                              ('min_dcf','eer_pct','cllr','act_dcf','rows','official_commit','score_sha256')},
                            'official_eval/comparison_file': str(output/'reports/official_eval_comparison.md')})
        published[combo] = result['score_sha256']
        write_json(saved_path, published)
    return published


def publish_final_figures(output, evaluations):
    if len(evaluations) != len(COMBOS): return False
    signature = hashlib.sha256(json.dumps([(r['combo'],r['score_sha256']) for r in evaluations]).encode()).hexdigest()
    saved = output/'reports/official_analysis_wandb.json'
    if read_json(saved, {}).get('signature') == signature: return True
    import wandb
    run_id = hashlib.sha256((output.name+'/official-eval-analysis-v1').encode()).hexdigest()[:12]
    run = wandb.init(project='asvspoof5-track1',id=run_id,resume='allow',mode='online',
                     name=output.name+'-official-eval-analysis',group=output.name,
                     tags=['official-eval','680774-trials','analysis'],dir=str(output/'wandb_sync'))
    try:
        import pandas as pd
        run.log({'official_eval/comparison':wandb.Table(dataframe=pd.read_csv(output/'reports/official_eval_comparison.csv'))})
        for path in [output/'figures/official_eval_comparison.png',output/'figures/eval_detection.png',
                     output/'figures/eval_score_distributions.png',
                     *sorted((output/'figures').glob('eval_attack_*.png')),
                     *sorted((output/'figures').glob('eval_codec_*.png'))]:
            if path.exists(): run.log({'official_eval/figures/'+path.stem:wandb.Image(str(path))})
        artifact = wandb.Artifact(output.name+'-official-eval',type='evaluation')
        for path in [output/'reports/official_eval_comparison.csv',output/'reports/official_eval_comparison.md',
                     output/'reports/official_package_manifest.json',
                     *sorted((output/'reports').glob('official_eval_a*.json'))]: artifact.add_file(str(path))
        run.log_artifact(artifact)
        url = run.url
        run.finish()
        write_json(saved,dict(signature=signature,url=url,complete=True,updated=time.time()))
        return True
    except Exception:
        run.finish(exit_code=1)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--official-repo', type=Path, default=Path('/content/asvspoof5_official'))
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    package = official_package(args.official_repo)
    preserved = args.output/'source/official_evaluation_package'
    preserved.mkdir(parents=True,exist_ok=True)
    for p in [*package.glob('*.py'),package/'README.md']:
        if p.exists(): shutil.copy2(p,preserved/p.name)
    source_manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('*.py')}
    write_json(args.output/'reports/official_package_manifest.json', dict(commit=OFFICIAL_COMMIT, files=source_manifest))
    previous_signature = None
    while True:
        evaluations, problems = [], []
        for combo in COMBOS:
            for split in ('dev','eval'):
                try:
                    result = score_split(args.output, combo, split, package)
                    if result and split == 'eval': evaluations.append(result)
                except Exception as exc:
                    problems.append(f'{combo}/{split}: {type(exc).__name__}: {exc}')
        publish_comparison(args.output, evaluations)
        signature = [(r['combo'],r['score_sha256']) for r in evaluations]
        if signature != previous_signature:
            try:
                visualize(args.output, evaluations)
                for result in evaluations: group_analysis(args.output, result, package)
                publish_comparison(args.output, evaluations)
                analysis_notebook(args.output, evaluations)
                previous_signature = signature
            except Exception as exc:
                problems.append('eval visualization: '+type(exc).__name__+': '+str(exc))
        try:
            published = publish_cloud(args.output, evaluations)
            figures_uploaded = publish_final_figures(args.output, evaluations) if not problems else False
        except Exception as exc:
            problems.append('official W&B publication: '+type(exc).__name__)
            published = read_json(args.output/'reports/official_wandb_publication.json', {})
            figures_uploaded = False
        complete = len(evaluations) == len(COMBOS) and len(published) == len(COMBOS) and figures_uploaded and not problems
        write_json(args.output/'reports/official_comparison_status.json',
                   dict(complete=complete, eval_models=[r['combo'] for r in evaluations], problems=problems,
                        source=RESULT_SOURCE, official_commit=OFFICIAL_COMMIT, updated=time.time()))
        print('Official eval models:', [r['combo'] for r in evaluations], 'problems:', problems, flush=True)
        if complete or not args.watch: return
        time.sleep(60)


if __name__ == '__main__': main()
