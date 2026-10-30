"""Insert data-driven tables and vector figures into the standalone paper."""
from pathlib import Path
import json,re,hashlib
P=Path(__file__).resolve().parent
r=json.loads((P.parent/'analysis/paper_seeds50_20261030/CPU_RESULTS.json').read_text())
assert r['status']=='passed'
tex=(P/'working_paper.tex').read_text()
def put(name,content):
 global tex
 begin,end=f'% {name}_BEGIN',f'% {name}_END'
 assert tex.count(begin)==tex.count(end)==1
 tex=re.sub(re.escape(begin)+'.*?'+re.escape(end),lambda m:begin+'\n'+content+'\n'+end,tex,flags=re.S)
def mean(mode,method,key):return r['means'][mode+'_'+method][key]
def score(mode,method,seed):return r['runs'][f'{mode}_{method}_s{seed}']['metrics']['primary_crps']['mean']
def ci(contrast,key,scale=1,dec=4):
 v=r['contrasts'][contrast][key]
 return '$'+f"{v['mean']*scale:.{dec}f}\\;[{v['ci_low']*scale:.{dec}f},{v['ci_high']*scale:.{dec}f}]"+'$'
def table(caption,label,cols,header,rows,note=''):
 return '\n'.join([r'\begin{table}[htbp]',r'\centering',r'\small',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+cols+'}',r'\toprule',header+r' \\',r'\midrule',*rows,r'\bottomrule',r'\end{tabular}',r'\par\vspace{4pt}\begin{minipage}{0.97\linewidth}\footnotesize '+note+r'\end{minipage}' if note else '',r'\end{table}'])
rows=[]
for method,label in [('global','Global reference'),('conditional','Conditional reference'),('empirical','Empirical retrieval')]:
 vals=[mean('raw',method,k)*s for k,s in [('primary_crps',1),('full_waveform_energy_fair',1e7),('variogram_p0p5_fair',1e7),('log_tf_band_rms_crps_fair',1)]]
 rows.append(label+' & '+' & '.join(f'{v:.4f}' for v in vals)+r' \\')
put('TABLE_RAW',table('Mean scores for raw generated waveforms. Lower is better in every column.','tab:raw','lrrrr',r'Model & Feature CRPS & Energy $\times10^7$ & Variogram $\times10^7$ & Time--frequency',rows,'Each mean averages three repetitions on 169 events with 16 draws per event. Neural repetitions vary training and generation; empirical repetitions vary sampling. Energy and variogram scores retain their respective archive-unit scales.'))
rows=[]
ks=[('Feature CRPS','primary_crps',1,5),('Energy $\\times10^8$','full_waveform_energy_fair',1e8,4),('Variogram $\\times10^8$','variogram_p0p5_fair',1e8,4),('Time--frequency CRPS','log_tf_band_rms_crps_fair',1,5)]
for name,key,scale,dec in ks:
 rows.append(name+' & '+ci('raw_conditional_minus_raw_global',key,scale,dec)+' & '+ci('raw_conditional_minus_raw_empirical',key,scale,dec)+r' \\')
put('TABLE_DIFFERENCES',table(r'Paired score differences for raw output, with pointwise 95\% intervals. Negative values favor conditional reference.','tab:differences','lrr',r'Score & Conditional $-$ global & Conditional $-$ empirical',rows,'Intervals resample whole proximity groups after averaging scores across repetitions. They condition on these fitted models and ensembles and do not adjust for selection or multiple comparisons.'))
rows=[]
for s in range(3):
 vals=[score('raw',m,s) for m in ['conditional','global','empirical']]+[score('filtered',m,s) for m in ['conditional','global']]
 rows.append(str(s+1)+' & '+' & '.join(f'{v:.6f}' for v in vals)+r' \\')
rows.append(r'\midrule')
vals=[mean('raw',m,'primary_crps') for m in ['conditional','global','empirical']]+[mean('filtered',m,'primary_crps') for m in ['conditional','global']]
rows.append('Mean & '+' & '.join(f'{v:.6f}' for v in vals)+r' \\')
put('TABLE_RUNS',table('Primary feature CRPS by repetition.','tab:runs','lrrrrr',r'Run & Raw cond. & Raw global & Raw empirical & Filtered cond. & Filtered global',rows,'Filtered empirical results are available for the first repetition only, with score 0.473211. The raw and filtered neural scores in each row use the same generated draws before and after filtering.'))
rows=[]
for name,key,scale,dec in ks:
 rows.append(name+' & '+ci('filtered_conditional_minus_raw_conditional',key,scale,dec)+' & '+ci('filtered_global_minus_raw_global',key,scale,dec)+r' \\')
put('TABLE_FILTER',table(r'Filtering effects: filtered minus raw score, with pointwise 95\% intervals. Negative values indicate an improvement after filtering.','tab:filter','lrr',r'Score & Conditional reference & Global reference',rows,'Both comparisons average the paired changes across three runs. Observed recordings are unchanged. Differences on different score scales should not be compared by their absolute size.'))

fig=[r'\begin{figure}[htbp]',r'\centering',r'\begin{tikzpicture}',r'\begin{axis}[width=0.96\linewidth,height=6.2cm,ybar,bar width=10pt,',r'ymin=0,ymax=1.18,ylabel={Mean feature CRPS},',r'xtick={1,2,3,4,5},xticklabels={Amplitude,Timing,Spectrum,Envelope,Autocorrelation},',r'xmin=0.5,xmax=5.5,axis lines*=left,ymajorgrids=true,grid style={gray!18},',r'tick label style={font=\small},label style={font=\small},',r'legend style={at={(0.5,1.04)},anchor=south,legend columns=3,draw=none,font=\small,/tikz/column sep=5pt},',r'enlarge x limits=false]']
for method,color in [('conditional','cond'),('global','glob'),('empirical','emp')]:
 vals=[mean('raw',method,'crps_'+g) for g in ['amplitude','timing','spectrum','envelope','correlation']]
 fig.append(r'\addplot[fill='+color+r',draw='+color+r'] coordinates {'+' '.join(f'({i+1},{v:.10f})' for i,v in enumerate(vals))+'};')
fig += [r'\legend{Conditional reference,Global reference,Empirical retrieval}',r'\end{axis}',r'\end{tikzpicture}',r'\caption{Raw feature scores by group. Each bar averages three repetitions on the same 169 events. The primary score gives equal weight to the five groups. Empirical retrieval has a large spectral advantage, while conditional diffusion has lower amplitude, timing and envelope scores. Bars show means, not uncertainty intervals.}',r'\label{fig:features}',r'\end{figure}']
put('FIG_FEATURES','\n'.join(fig))
fig=[r'\begin{figure}[htbp]',r'\centering']
for mode,title in [('raw','Raw output'),('filtered','Filtered generated output')]:
 fig += [r'\begin{minipage}{0.49\linewidth}\centering',r'\begin{tikzpicture}',r'\begin{axis}[width=\linewidth,height=6.2cm,',r'title={'+title+r'},title style={font=\small\bfseries},',r'xmin=0.5,xmax=3.5,ymin=0.35,ymax=0.525,ytick={0.35,0.40,0.45,0.50},',r'xtick={1,2,3},xticklabels={Cond.,Global,Empirical},',r'axis lines*=left,ymajorgrids=true,grid style={gray!18},',r'tick label style={font=\small},scaled y ticks=false,yticklabel style={/pgf/number format/fixed,/pgf/number format/precision=2},',r'ylabel={Feature CRPS},label style={font=\small}]']
 for x,(method,color) in enumerate([('conditional','cond'),('global','glob'),('empirical','emp')],1):
  seeds=[s for s in range(3) if f'{mode}_{method}_s{s}' in r['runs']]
  vals=[]
  for i,s in enumerate(seeds):
   v=score(mode,method,s);vals.append(v)
   jitter=[-.10,0,.10][i] if len(seeds)>1 else 0
   marker=['*','triangle*','square*'][i]
   fig.append(r'\addplot[only marks,mark='+marker+r',mark size=2.2pt,color='+color+'] coordinates {'+f'({x+jitter},{v:.10f})'+'};')
  m=mean(mode,method,'primary_crps')
  fig.append(r'\addplot[no marks,very thick,color='+color+'] coordinates {'+f'({x-.23},{m:.10f}) ({x+.23},{m:.10f})'+'};')
  fig.append(r'\node[font=\scriptsize,anchor=south] at (axis cs:'+f'{x},{max(vals)+.006}'+'){'+f'{m:.4f}'+'};')
  if len(seeds)==1:fig.append(r'\node[font=\scriptsize,anchor=north] at (axis cs:'+f'{x},{m-.009}'+r'){one repetition};')
 fig += [r'\end{axis}',r'\end{tikzpicture}',r'\end{minipage}%']
fig += [r'\caption{Primary feature CRPS before and after filtering generated waveforms. Markers show repetitions and horizontal lines their means; lower is better. All comparisons use 169 events and 16 draws per event. Filtered empirical retrieval has one repetition, while the other entries have three. The observed waveforms are unchanged.}',r'\label{fig:primary}',r'\end{figure}']
put('FIG_PRIMARY','\n'.join(fig))
(P/'working_paper.tex').write_text(tex)
manifest={'input':'../analysis/paper_seeds50_20261030/CPU_RESULTS.json','input_sha256':hashlib.sha256((P.parent/'analysis/paper_seeds50_20261030/CPU_RESULTS.json').read_bytes()).hexdigest(),'tables':4,'figures':2,'scope':'Aggregate saved results only; no new fitting or waveform scoring.'}
(P/'review/exhibit_sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Inserted four numeric tables and two self-contained vector figures.')
