import './scouting-report.css';
import { useEffect,useMemo,useState } from 'react';
import { Activity, ArrowRight, BarChart3, ChevronDown, CircleHelp, Download, ExternalLink, Filter, GitBranch, Info, Layers, LayoutDashboard, Lightbulb, Menu, MoveUpRight, Search, ShieldCheck, Target, TrendingUp, X } from 'lucide-react';
import xtGrid from './xt-grid.json';

type Pass={team:string;player:string;recipient:string;x:number;y:number;endX:number;endY:number;completed:boolean;minute:number};
type PassModel={trained:boolean;model_type:string;means:number[];scales:number[];coefficients:number[];intercept:number;metrics:{roc_auc:number;accuracy:number;brier:number;baseline_accuracy:number;baseline_brier:number;training_matches:number;test_matches:number;test_passes:number};limitations:string};
type CalibrationBin={lower:number;upper:number;count:number;mean_predicted:number;observed_completion:number};
type ClassificationMetrics={accuracy:number;brier:number;log_loss:number;roc_auc:number|null;incomplete_precision:number;incomplete_recall:number;incomplete_f1:number;completed_f1:number;confusion_matrix:number[][]};
type ModelEvaluation={dataset:string;test_match_count:number;test_pass_count:number;selected_threshold?:number;validation_threshold_results?:{threshold:number;incomplete_precision:number;incomplete_recall:number;incomplete_f1:number}[];results:ClassificationMetrics;constant_prior_baseline:Partial<ClassificationMetrics>;always_complete_baseline:{accuracy:number};calibration_bins:CalibrationBin[];match_bootstrap?:{unit:string;resamples:number;brier_improvement_model_vs_prior:{lower_95:number;upper_95:number}};limitations:string};
const percent=(v:number)=>`${(v*100).toFixed(1)}%`;
function EvaluationPanel({data}:{data:ModelEvaluation}){
 const r=data.results,base=data.constant_prior_baseline,cm=r.confusion_matrix;
 const threshold=data.selected_threshold??0.5;
 const bins=data.calibration_bins||[];
 const points=bins.map(b=>`${35+b.mean_predicted*290},${244-b.observed_completion*195}`).join(' ');
 const selected=data.validation_threshold_results?.find(t=>Math.abs(t.threshold-threshold)<0.00001);
 const ci=data.match_bootstrap?.brier_improvement_model_vs_prior;
 return <section className="panel" id="evaluation" style={{marginTop:24,padding:28}}>
  <div className="panel-header"><div><div className="section-overline">TRANSPARENT MODEL VALIDATION</div><h2>AI Model Performance</h2><p>Unseen 2018 World Cup matches · grouped by match, not individual passes</p></div><ShieldCheck size={24} color="#63d4b0"/></div>
  <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(155px,1fr))',gap:12,marginBottom:22}}>
   {[
    ['ROC-AUC',r.roc_auc===null?'—':fmt(r.roc_auc,3),'0.500 = chance'],
    ['Brier score',fmt(r.brier,4),`Prior baseline: ${fmt(base.brier??NaN,4)} · lower is better`],
    ['Incomplete recall',percent(r.incomplete_recall),'Share of failed passes flagged'],
    ['Incomplete F1',percent(r.incomplete_f1),'Balance of recall and precision'],
    ['Accuracy',percent(r.accuracy),`Always complete: ${percent(data.always_complete_baseline.accuracy)}`],
    ['Selected cutoff',percent(threshold),'Chosen on validation matches']
   ].map(([label,value,detail])=><div className="stat" key={label} style={{minWidth:0,padding:16}}><div className="stat-top">{label}</div><div className="stat-value" style={{fontSize:26}}>{value}</div><div className="stat-sub">{detail}</div></div>)}
  </div>
  <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(270px,1fr))',gap:20}}>
   <div><h3 style={{marginBottom:6}}>Confusion matrix</h3><p style={{fontSize:13,marginBottom:12}}>Rows: actual outcome · Columns: predicted outcome</p>
    <div style={{display:'grid',gridTemplateColumns:'105px repeat(2,minmax(0,1fr))',gap:5,textAlign:'center',fontSize:12}}>
     <div/><strong>Pred. incomplete</strong><strong>Pred. completed</strong>
     <strong style={{alignSelf:'center'}}>Actual incomplete</strong>
     <div style={{background:'#dcfce7',color:'#14532d',borderRadius:8,padding:'20px 6px'}}><b style={{fontSize:20}}>{cm[0][0].toLocaleString()}</b><div>Correct</div></div>
     <div style={{background:'#fee2e2',color:'#7f1d1d',borderRadius:8,padding:'20px 6px'}}><b style={{fontSize:20}}>{cm[0][1].toLocaleString()}</b><div>Missed</div></div>
     <strong style={{alignSelf:'center'}}>Actual completed</strong>
     <div style={{background:'#fee2e2',color:'#7f1d1d',borderRadius:8,padding:'20px 6px'}}><b style={{fontSize:20}}>{cm[1][0].toLocaleString()}</b><div>False alarm</div></div>
     <div style={{background:'#dcfce7',color:'#14532d',borderRadius:8,padding:'20px 6px'}}><b style={{fontSize:20}}>{cm[1][1].toLocaleString()}</b><div>Correct</div></div>
    </div>
    <p style={{fontSize:13,marginTop:12}}>At this cutoff, the model catches {percent(r.incomplete_recall)} of failed passes, but only {percent(r.incomplete_precision)} of flagged passes actually failed.</p>
   </div>
   <div><h3 style={{marginBottom:6}}>Probability calibration</h3><p style={{fontSize:13,marginBottom:4}}>Predicted completion probability vs. observed completion rate</p>
    <svg viewBox="0 0 350 280" style={{width:'100%',maxWidth:430}} role="img" aria-label="Calibration curve, diagonal ideal calibration line and observed points">
     <rect x="35" y="49" width="290" height="195" fill="#f7fafc" rx="5"/>
     {[0,.25,.5,.75,1].map(v=><g key={v}><line x1="35" y1={244-v*195} x2="325" y2={244-v*195} stroke="#dce4eb"/><text x="29" y={248-v*195} fontSize="10" textAnchor="end" fill="#687588">{Math.round(v*100)}</text><text x={35+v*290} y="260" fontSize="10" textAnchor="middle" fill="#687588">{Math.round(v*100)}</text></g>)}
     <line x1="35" y1="244" x2="325" y2="49" stroke="#8a99ab" strokeWidth="2" strokeDasharray="5 5"/>
     {points&&<polyline points={points} fill="none" stroke="#159e7c" strokeWidth="3"/>}
     {bins.map((b,i)=><circle key={i} cx={35+b.mean_predicted*290} cy={244-b.observed_completion*195} r={b.count<100?3.5:5} fill="#159e7c"><title>{`${b.count} passes; predicted ${percent(b.mean_predicted)}; observed ${percent(b.observed_completion)}`}</title></circle>)}
     <text x="180" y="276" textAnchor="middle" fontSize="11" fill="#526176">Predicted completion (%)</text>
     <text x="12" y="35" fontSize="11" fill="#526176">Observed completion (%)</text>
    </svg>
    <p style={{fontSize:12}}>Dashed = perfect calibration; green = observed. Small low-probability bins are less reliable.</p>
   </div>
  </div>
  {data.validation_threshold_results&&<div style={{marginTop:22}}><h3>Threshold selected on validation matches</h3><div style={{overflowX:'auto'}}><table style={{width:'100%',borderCollapse:'collapse',fontSize:13,marginTop:10}}><thead><tr>{['Completion cutoff','Incomplete precision','Incomplete recall','Incomplete F1'].map(k=><th key={k} style={{padding:9,textAlign:'left',borderBottom:'1px solid #dce3eb'}}>{k}</th>)}</tr></thead><tbody>{data.validation_threshold_results.map(v=><tr key={v.threshold} style={{background:Math.abs(v.threshold-threshold)<0.00001?'#d9f6ea':'transparent',color:Math.abs(v.threshold-threshold)<0.00001?'#153b31':'inherit'}}><td style={{padding:9}}>{percent(v.threshold)}{Math.abs(v.threshold-threshold)<0.00001?' · selected':''}</td><td style={{padding:9}}>{percent(v.incomplete_precision)}</td><td style={{padding:9}}>{percent(v.incomplete_recall)}</td><td style={{padding:9}}>{percent(v.incomplete_f1)}</td></tr>)}</tbody></table></div>{selected&&<p style={{fontSize:13}}>The {percent(threshold)} cutoff maximized incomplete-pass F1 among the tested candidates on validation matches. It was not chosen using final test outcomes.</p>}</div>}
  <div className="analysis-note" style={{marginTop:20}}><CircleHelp size={19}/><div><strong>Evidence and limitations</strong><p>{data.test_pass_count.toLocaleString()} passes from {data.test_match_count} held-out matches. {ci?`The match-bootstrap 95% interval for Brier improvement over the constant-prior baseline is ${fmt(ci.lower_95,4)} to ${fmt(ci.upper_95,4)}.`:''} {data.limitations}</p><p>These are retrospective results for recorded pass endpoints; this model does not establish that an alternative pass would succeed or produce a goal. Classification accuracy changes with the cutoff, whereas ROC-AUC and Brier score do not.</p></div></div>
 </section>
}


type XtQuintile={quintile:number;possessions:number;shot_rate:number;mean_positive_xt:number;mean_signed_xt?:number;mean_shot_xg?:number};
type XtValidation={shot_discrimination_auc:number;quintiles:XtQuintile[];limitations?:string[];dataset?:string;total_possessions?:number};
function ThreatValidation({data,error}:{data:XtValidation|null;error:string}){
  if(!data) return <section className="panel model-report-card"><div className="section-overline">06 / ATTACKING THREAT VALIDATION</div><h2>Does xT correspond to observed shots?</h2><p>{error||'Loading xt-validation.json…'}</p><p>Run <code>python analytics/validate_xt.py --limit 50</code> from the project root, then refresh.</p></section>;
  const groups=[...data.quintiles].sort((a,b)=>a.quintile-b.quintile);
  const total=groups.reduce((n,g)=>n+g.possessions,0);
  const first=groups[0], last=groups[groups.length-1];
  const pct=(n:number)=>(n*100).toFixed(1)+'%';
  const auc=Number.isFinite(data.shot_discrimination_auc)?data.shot_discrimination_auc.toFixed(3):'—';
  return <section className="panel model-report-card" id="threat-validation">
    <div className="section-overline">06 / ATTACKING THREAT VALIDATION</div>
    <h2>Does measured attacking threat relate to shots?</h2>
    <p>Independent descriptive check using historical StatsBomb possessions. We group possessions by accumulated <strong>positive completed-pass xT</strong> and measure how often each group contained a shot.</p>
    <div className="xtv-kpis"><div><small>RETROSPECTIVE SHOT AUC</small><strong>{auc}</strong></div><div><small>POSSESSIONS</small><strong>{total.toLocaleString()}</strong></div><div><small>HIGHEST xT SHOT RATE</small><strong>{last?pct(last.shot_rate):'—'}</strong></div><div><small>LOWEST xT SHOT RATE</small><strong>{first?pct(first.shot_rate):'—'}</strong></div></div>
    <div className="xtv-chart" role="img" aria-label="Shot rates by positive xT quintile">
    {groups.map((g,i)=><div className="xtv-barrow" key={g.quintile}>
      <div className="xtv-barlabel">{i===0?'Lowest xT':i===groups.length-1?'Highest xT':`Quintile ${g.quintile}`}</div>
      <div className="xtv-bartrack"><div className="xtv-barfill" style={{width:`${Math.min(100,Math.max(0,g.shot_rate*100))}%`}}/></div>
      <strong>{pct(g.shot_rate)}</strong><small>{g.possessions.toLocaleString()} possessions</small>
    </div>)}
    </div>
    <div className="xtv-interpretation"><strong>What the result means</strong><p>{first&&last?`The highest-xT group contained shots in ${pct(last.shot_rate)} of possessions, versus ${pct(first.shot_rate)} in the lowest group. `:''}This supports an association between completed-pass threat gain and attacking activity, but the group rates are not guaranteed to rise monotonically.</p></div>
    <div className="analysis-note xtv-limitations"><CircleHelp size={19}/><div><strong>Methodological limits</strong><p>This is not a prediction of future shots. xT gains and shot occurrence were measured within the same possessions. The analysis excludes carries, dribbles and turnover costs, and rank-based groups can split tied zero-xT possessions. It does not establish causality.</p>{data.limitations?.length?<details><summary>Full validation limitations</summary><ul>{data.limitations.map((v,i)=><li key={i}>{v}</li>)}</ul></details>:null}</div></div>
  </section>;
}
function ModelReport({data,model,error,xtValidation,xtError}:{data:ModelEvaluation|null;model:PassModel|null;error:string;xtValidation:XtValidation|null;xtError:string}){
 const [showDetails,setShowDetails]=useState(false);
 const coefficients=model?.coefficients||[];
 const featureNames=['Starting x','Starting y','Forward distance (dx)','Sideways distance (dy)','Pass distance','Match minute'];
 const importance=featureNames.map((name,i)=>({name,value:coefficients[i]||0})).sort((a,b)=>Math.abs(b.value)-Math.abs(a.value));
 const maxWeight=Math.max(0.01,...importance.map(v=>Math.abs(v.value)));
 const r=data?.results,baseline=data?.constant_prior_baseline;
 return <div className="model-report-page">
   <div className="model-report-hero"><div><div className="eyebrow">MODEL AUDIT / TRANSPARENT EVALUATION</div><h1>Model Report</h1><p>Can our pass-completion model outperform a simple baseline on unseen football matches?</p></div><button className="outline-btn" onClick={()=>window.print()}><Download size={17}/> Print / Save PDF</button></div>
   {!data?<section className="panel model-report-card"><h2>Evaluation not loaded</h2><p>{error||'Loading model-evaluation.json…'}</p><p>Run <code>python analytics/train_pass_model.py --limit 50</code> from the project root, then refresh.</p></section>:<>
    <div className="model-report-meta"><span>StatsBomb · 2018 FIFA World Cup</span><span>{data.test_match_count} held-out matches</span><span>{data.test_pass_count.toLocaleString()} test passes</span><span>Validation-selected cutoff: {percent(data.selected_threshold??0.5)}</span></div>
    <section className="panel model-report-card"><div className="section-overline">01 / BASELINE COMPARISON</div><h2>Does the model really add value?</h2><p>Probability scoring is evaluated independently of the classification threshold. Lower Brier and log loss are better; higher ROC-AUC is better.</p>
     <div className="model-report-table-wrap"><table className="model-report-table"><thead><tr><th>Metric</th><th>Trained logistic regression</th><th>Constant-prior baseline</th><th>Interpretation</th></tr></thead><tbody>
      <tr><td>ROC-AUC ↑</td><td><strong>{r?.roc_auc==null?'—':fmt(r.roc_auc,4)}</strong></td><td>{baseline?.roc_auc==null?'0.5000':fmt(baseline.roc_auc,4)}</td><td>Separates completed from incomplete passes</td></tr>
      <tr><td>Brier score ↓</td><td><strong>{fmt(r?.brier??NaN,5)}</strong></td><td>{fmt(baseline?.brier??NaN,5)}</td><td>Probability prediction error</td></tr>
      <tr><td>Log loss ↓</td><td><strong>{fmt(r?.log_loss??NaN,5)}</strong></td><td>{fmt(baseline?.log_loss??NaN,5)}</td><td>Penalizes confident wrong predictions</td></tr>
      <tr><td>Accuracy at selected cutoff</td><td><strong>{percent(r?.accuracy??0)}</strong></td><td>{percent(baseline?.accuracy??0)}</td><td>Depends on cutoff; not the primary goal</td></tr>
     </tbody></table></div>
     {data.match_bootstrap&&<p className="model-report-evidence">Match-bootstrap 95% interval for baseline Brier score minus model Brier score: <strong>{fmt(data.match_bootstrap.brier_improvement_model_vs_prior.lower_95,5)} to {fmt(data.match_bootstrap.brier_improvement_model_vs_prior.upper_95,5)}</strong> ({data.match_bootstrap.resamples.toLocaleString()} resamples). Positive values favor the model on this test sample.</p>}
    </section>
    <section className="model-report-two-col">
     <div className="panel model-report-card"><div className="section-overline">02 / CALIBRATION</div><h2>Do predicted probabilities match reality?</h2><p>Each dot represents a probability bin. Dashed diagonal = perfect calibration. Bins with fewer observations are less stable.</p>
      <svg viewBox="0 0 410 325" className="model-report-calibration" role="img" aria-label="Reliability curve comparing predicted and observed pass completion probabilities">
       <rect x="52" y="20" width="322" height="265" rx="6" fill="#f7fafb"/>
       {[0,.25,.5,.75,1].map(t=><g key={t}><line x1="52" x2="374" y1={285-t*265} y2={285-t*265} stroke="#dae5eb"/><text x="44" y={289-t*265} fontSize="11" textAnchor="end" fill="#65758a">{Math.round(t*100)}</text><text x={52+t*322} y="302" textAnchor="middle" fontSize="11" fill="#65758a">{Math.round(t*100)}</text></g>)}
       <line x1="52" y1="285" x2="374" y2="20" stroke="#90a0af" strokeWidth="2" strokeDasharray="6 6"/>
       <polyline fill="none" stroke="#129b79" strokeWidth="3" points={data.calibration_bins.map(b=>`${52+b.mean_predicted*322},${285-b.observed_completion*265}`).join(' ')}/>
       {data.calibration_bins.map((b,i)=><circle key={i} cx={52+b.mean_predicted*322} cy={285-b.observed_completion*265} r={b.count<100?4:6} fill="#129b79"><title>{`${b.count} passes, predicted ${percent(b.mean_predicted)}, observed ${percent(b.observed_completion)}`}</title></circle>)}
       <text x="213" y="322" fontSize="12" textAnchor="middle" fill="#596a7c">Predicted completion (%)</text>
       <text x="52" y="12" fontSize="11" fill="#596a7c">Observed completion (%)</text>
      </svg>
     </div>
     <div className="panel model-report-card"><div className="section-overline">03 / ERROR ANALYSIS</div><h2>When does the model fail?</h2><p>Confusion matrix at the validation-selected cutoff of {percent(data.selected_threshold??0.5)}. Rows are actual outcomes; columns are predictions.</p>
      <div className="model-report-matrix"><div/><strong>Predicted incomplete</strong><strong>Predicted completed</strong><strong>Actual incomplete</strong><div className="matrix-good"><b>{r?.confusion_matrix?.[0]?.[0]?.toLocaleString()??'—'}</b><small>Correctly flagged</small></div><div className="matrix-bad"><b>{r?.confusion_matrix?.[0]?.[1]?.toLocaleString()??'—'}</b><small>Missed failure</small></div><strong>Actual completed</strong><div className="matrix-bad"><b>{r?.confusion_matrix?.[1]?.[0]?.toLocaleString()??'—'}</b><small>False alert</small></div><div className="matrix-good"><b>{r?.confusion_matrix?.[1]?.[1]?.toLocaleString()??'—'}</b><small>Correctly completed</small></div></div>
      <div className="model-report-kpis"><div><span>Incomplete recall</span><strong>{percent(r?.incomplete_recall??0)}</strong></div><div><span>Incomplete precision</span><strong>{percent(r?.incomplete_precision??0)}</strong></div><div><span>Incomplete F1</span><strong>{percent(r?.incomplete_f1??0)}</strong></div></div>
      <p className="model-report-caveat">Increasing failure detection also increases false alerts. Threshold choice does not change ROC-AUC, Brier, or log loss.</p>
     </div>
    </section>
    <section className="panel model-report-card"><div className="section-overline">04 / MODEL INTERPRETABILITY</div><h2>Which features influence the prediction?</h2><p>Signed standardized logistic-regression coefficients. Positive values raise predicted completion odds; negative values lower them, with other variables held fixed. <strong>Coefficient size is not causal importance.</strong> Correlated features may share information.</p>
     {model?<div className="model-feature-chart">{importance.map(f=><div className="model-feature-row" key={f.name}><span>{f.name}</span><div className="model-feature-track"><div className="model-feature-center"/><div className={'model-feature-bar '+(f.value>=0?'positive':'negative')} style={{width:`${Math.abs(f.value)/maxWeight*49}%`,left:f.value>=0?'50%':`${50-Math.abs(f.value)/maxWeight*49}%`}}/></div><b>{f.value>=0?'+':''}{fmt(f.value,3)}</b></div>)}</div>:<p>Model coefficients unavailable. Verify that <code>frontend/public/pass-model.json</code> is present.</p>}
     <button className="model-report-details" onClick={()=>setShowDetails(!showDetails)}>{showDetails?'Hide':'Show'} model feature descriptions</button>{showDetails&&<p className="model-report-caveat">Features: starting x/y, recorded endpoint displacement dx/dy, pass distance, and match minute. Recorded endpoints on incomplete passes may not represent intended targets. This is retrospective modeling, not a valid real-time coaching predictor.</p>}
    </section>
    {data.validation_threshold_results&&<section className="panel model-report-card"><div className="section-overline">05 / VALIDATION CUT-OFF</div><h2>Thresholds selected before final testing</h2><p>The cutoff was chosen on separate validation matches by incomplete-pass F1, then evaluated on held-out test matches.</p><div className="model-report-table-wrap"><table className="model-report-table"><thead><tr><th>Cutoff</th><th>Validation precision (incomplete)</th><th>Validation recall (incomplete)</th><th>Validation F1 (incomplete)</th></tr></thead><tbody>{data.validation_threshold_results.map(t=><tr key={t.threshold} className={t.threshold===(data.selected_threshold??0.5)?'chosen':''}><td>{percent(t.threshold)}</td><td>{percent(t.incomplete_precision)}</td><td>{percent(t.incomplete_recall)}</td><td>{percent(t.incomplete_f1)}</td></tr>)}</tbody></table></div></section>}
    <ThreatValidation data={xtValidation} error={xtError}/>
     <section className="panel model-report-card"><div className="section-overline">07 / LIMITATIONS</div><h2>What this model cannot claim</h2><p>{data.limitations}</p><p>The results come from one match-grouped test split from the 2018 World Cup; they do not establish generalization to future seasons or leagues. Feature coefficients must not be described as causal effects. Scouting recommendations are exploratory.</p></section>
   </>}
  </div>;
}

function predictCompletion(model:PassModel,p:Pass):number{const dx=p.endX-p.x,dy=p.endY-p.y;const values=[p.x,p.y,dx,dy,Math.hypot(dx,dy),p.minute];const z=model.intercept+values.reduce((sum,v,i)=>sum+(v-model.means[i])/Math.max(1e-12,model.scales[i])*model.coefficients[i],0);return 1/(1+Math.exp(-Math.max(-35,Math.min(35,z))))}
type Match={id:string;label:string;competition:string;date:string;home:string;away:string;passes:Pass[];source:'demo'|'statsbomb'};
const teams=['Arsenal','Barcelona','Inter Miami','Bayern Munich'];
const clamp=(x:number,a:number,b:number)=>Math.max(a,Math.min(b,x));
const rnd=(seed:number)=>{let x=seed;return()=>((x=(x*1664525+1013904223)>>>0)/4294967296)};
function demoMatch(id:number):Match{const r=rnd(600+id*97);const a=teams[id%teams.length],b=teams[(id+1)%teams.length];const passes:Pass[]=[];const names=['#4 Rice','#8 Ødegaard','#7 Saka','#11 Martinelli','#9 Striker','#6 Pivot','#2 Fullback','#10 Playmaker'];for(let i=0;i<360;i++){const t=i%3===0?b:a;const x=10+r()*92,y=5+r()*70;const dy=(r()-.5)*45;const dx=(r()-.28)*48;const endX=clamp(x+dx,0,120),endY=clamp(y+dy,0,80);const completed=r()<(dx>25?.63:.86);passes.push({team:t,player:names[Math.floor(r()*names.length)],recipient:names[Math.floor(r()*names.length)],x,y,endX,endY,completed,minute:Math.floor(i/360*90)})}return{id:'demo-'+id,label:`${a} vs ${b}`,competition:'Illustrative sample match',date:'Demo data only',home:a,away:b,passes,source:'demo'}}
const demos=[demoMatch(0),demoMatch(1),demoMatch(2)];
// Published 12x8 xT grid lookup on StatsBomb 120x80 attack-relative coordinates.
// This is a descriptive grid-based metric, not a fitted model on our match sample.
function threat(x: number, y: number): number {
  const col = Math.max(0, Math.min(11, Math.floor((x / 120) * 12)));
  const row = Math.max(0, Math.min(7, Math.floor((y / 80) * 8)));
  return xtGrid[row][col];
}
// StatsBomb event locations are attack-relative: +x toward the opposition goal.
// y=0 is one touchline; we label top/left and bottom/right for this display.
const zone=(p:Pass)=>p.y<80/3?'Left flank':p.y<160/3?'Central corridor':'Right flank';
const fmt=(v:number,d=2)=>Number.isFinite(v)?v.toFixed(d):'—';
const validPass=(p:Pass)=>[p.x,p.y,p.endX,p.endY].every(Number.isFinite)&&p.x>=0&&p.x<=120&&p.endX>=0&&p.endX<=120&&p.y>=0&&p.y<=80&&p.endY>=0&&p.endY<=80;
function analyze(passes:Pass[]){const all=passes.filter(validPass);const total=all.length,complete=all.filter(p=>p.completed);const attemptValue=(p:Pass)=>p.completed?threat(p.endX,p.endY)-threat(p.x,p.y):0;const routes=['Left flank','Central corridor','Right flank'].map(name=>{const group=all.filter(p=>zone(p)===name);const good=group.filter(p=>p.completed);const net=group.reduce((s,p)=>s+attemptValue(p),0);return{name,attempts:group.length,completed:good.length,completion:group.length?good.length/group.length*100:0,net,per100:group.length?net/group.length*100:0}});return{total,complete:complete.length,rate:total?complete.length/total*100:0,net:all.reduce((s,p)=>s+attemptValue(p),0),routes,forward:complete.filter(p=>p.endX>p.x+10).length}}
function Field({passes,view}:{passes:Pass[],view:string}){const shown=passes.filter(p=>validPass(p)&&p.completed&&p.endX>p.x).slice(0,85);const playerMap=new Map<string,{x:number;y:number;n:number}>();for(const p of passes.filter(validPass)){const old=playerMap.get(p.player)||{x:0,y:0,n:0};playerMap.set(p.player,{x:old.x+p.x,y:old.y+p.y,n:old.n+1})}const players=[...playerMap.entries()].sort((a,b)=>b[1].n-a[1].n).slice(0,9).map(([name,val])=>({name,x:val.x/val.n*10,y:val.y/val.n*10}));const connections=new Map<string,number>();for(const p of passes.filter(p=>validPass(p)&&p.completed)){if(p.player!==p.recipient&&players.some(q=>q.name===p.player)&&players.some(q=>q.name===p.recipient)){const key=[p.player,p.recipient].sort().join('|');connections.set(key,(connections.get(key)||0)+1)}}return <svg className="pitch" viewBox="0 0 1200 800" role="img" aria-label="Football pitch with completed forward passes">
<defs><pattern id="grass" width="200" height="800" patternUnits="userSpaceOnUse"><rect width="100" height="800" fill="#185e56"/><rect x="100" width="100" height="800" fill="#1a675d"/></pattern><linearGradient id="warm"><stop stopColor="#d8fbac"/><stop offset="1" stopColor="#b9ed91"/></linearGradient></defs>
<rect width="1200" height="800" fill="url(#grass)"/>
<g fill="none" stroke="#d2ebe3" strokeWidth="3" opacity=".75"><rect x="17" y="17" width="1166" height="766"/><path d="M600 17v766"/><circle cx="600" cy="400" r="94"/><circle cx="600" cy="400" r="4" fill="#d2ebe3"/><rect x="17" y="170" width="190" height="460"/><rect x="17" y="290" width="72" height="220"/><rect x="993" y="170" width="190" height="460"/><rect x="1111" y="290" width="72" height="220"/><path d="M207 320 Q290 400 207 480 M993 320 Q910 400 993 480"/></g>
{view==='zones'?Array.from({length:12*8},(_,i)=>{const col=i%12,row=Math.floor(i/12),v=threat((col+.5)*10,(row+.5)*10);return <rect key={i} x={col*100+2} y={row*100+2} width="96" height="96" rx="8" fill="#f9c76a" opacity={Math.min(.78,.06+(v/.25745362)*.72)}/> }):view==='passes'?shown.map((p,i)=><line key={i} x1={p.x*10} y1={p.y*10} x2={p.endX*10} y2={p.endY*10} stroke={view==='passes'?'#d8fbac':'#8ed6fb'} strokeWidth={view==='passes'?3:2.5} opacity={.35}/>):null}
{view==='passes'&&shown.slice(-28).map((p,i)=><circle key={i} cx={p.endX*10} cy={p.endY*10} r="5" fill="#e5ffc3" opacity=".7"/>)}
{view==='network'&&<g>{[...connections.entries()].filter(([,count])=>count>=2).map(([key,count])=>{const [a,b]=key.split('|');const u=players.find(p=>p.name===a),v=players.find(p=>p.name===b);return u&&v?<line key={key} x1={u.x} y1={u.y} x2={v.x} y2={v.y} stroke="#b9ed91" strokeWidth={Math.min(17,2+count)} opacity=".55"/>:null})}{players.map(p=><g key={p.name}><circle cx={p.x} cy={p.y} r="16" fill="#b9ed91" stroke="#165247" strokeWidth="3"/><text x={p.x} y={p.y-24} fill="#ffffff" fontWeight="bold" fontSize="20" textAnchor="middle">{p.name.split(' ').slice(-1)[0]?.slice(0,10)}</text></g>)}</g>}
<g fill="#e3f5eb" fontSize="22" fontFamily="sans-serif" opacity=".8"><text x="40" y="48">DEFENSE</text><text x="1030" y="48">ATTACK →</text></g></svg>}

function ScoutingReport({matches,selected,onMatch,team,onTeam,model}:{matches:Match[];selected:string;onMatch:(id:string)=>void;team:string;onTeam:(team:string)=>void;model:PassModel|null}){
  const match=matches.find(m=>m.id===selected) ?? matches[0];
  const passes=useMemo(()=>match.passes.filter(p=>p.team===team&&validPass(p)),[match,team]);
  const stats=useMemo(()=>analyze(passes),[passes]);
  const corridors=[...stats.routes].sort((a,b)=>b.per100-a.per100);
  const qualified=corridors.filter(r=>r.attempts>=20);
  const top=qualified[0];
  const bottom=[...qualified].sort((a,b)=>a.per100-b.per100)[0];
  const ranked=useMemo(()=>model?passes.map(p=>({p,prob:predictCompletion(model,p),xt:threat(p.endX,p.endY)-threat(p.x,p.y)})):[],[passes,model]);
  const meanProbability=ranked.length?ranked.reduce((n,e)=>n+e.prob,0)/ranked.length*100:null;
  const predictedFailure=ranked.filter(e=>e.prob<0.8).length;
  const takeaways:string[]=[];
  if(top){takeaways.push(`${top.name} generated the highest observed xT per 100 attempted passes (${fmt(top.per100,3)} over ${top.attempts} attempts). Investigate this pattern in video before changing tactics.`)}
  else{takeaways.push('Insufficient attempts per corridor for a reliable route comparison. Add more matches before making corridor-level decisions.')}
  if(bottom && bottom.name!==top?.name){takeaways.push(`${bottom.name} returned ${fmt(bottom.per100,3)} completed-pass xT per 100 attempts. Review pass selection and positioning in this area; these are observational results.`)}
  else {takeaways.push(`The team completed ${fmt(stats.rate,1)}% of its ${stats.total} observed passes. Compare progression as well as completion rate.`)}
  if(model && meanProbability!==null){takeaways.push(`The retrospective ML model estimated ${fmt(meanProbability,1)}% average completion probability across these recorded passes; ${predictedFailure} were flagged as higher-risk at its 80% cutoff. Verify video context and defender pressure.`)}
  else{takeaways.push('AI risk analysis unavailable until pass-model.json is loaded. Existing passing and xT summaries remain usable.')}
  const download=()=>{const previous=document.title;document.title=`TactiQ - ${team} - ${match.label}`;window.print();setTimeout(()=>{document.title=previous},700)};
  return <div className="scouting-page">
    <div className="scouting-controls no-print">
      <div><div className="eyebrow">COACHING INTELLIGENCE / PRINTABLE REPORT</div><h1>Scouting Report</h1><p>A clear, evidence-led summary for a real match.</p></div>
      <button className="scout-print" onClick={download}><Download size={17}/> Download as PDF</button>
    </div>
    <div className="scouting-filters no-print">
      <label>Match<select value={selected} onChange={e=>onMatch(e.target.value)}>{matches.map(m=><option key={m.id} value={m.id}>{m.label}{m.source==='demo'?' (demo)':''}</option>)}</select></label>
      <label>Team<select value={team} onChange={e=>onTeam(e.target.value)}><option value={match.home}>{match.home}</option><option value={match.away}>{match.away}</option></select></label>
    </div>
    <article className="scout-paper">
      <header className="scout-paper-header"><div><div className="scout-kicker">TACTIQ AI / MATCH SCOUTING REPORT</div><h2>{team}</h2><p>{match.label} · {match.competition} · {match.date}</p></div><div className="scout-brand">TQ<span>●</span></div></header>
      {match.source==='demo'&&<p className="scout-demo">DEMO DATA: All findings on this report are illustrative, not real match evidence.</p>}
      <div className="scout-metrics"><div><small>PASS ATTEMPTS</small><strong>{stats.total.toLocaleString()}</strong></div><div><small>COMPLETION RATE</small><strong>{fmt(stats.rate,1)}%</strong></div><div><small>PROGRESSIVE PASSES</small><strong>{stats.forward}</strong></div><div><small>NET COMPLETED-PASS xT</small><strong>{stats.net>=0?'+':''}{fmt(stats.net,2)}</strong></div></div>
      <div className="scout-columns"><section><h3>Observed attacking corridors</h3><p className="scout-muted">Completed-pass ΔxT per 100 attempts · pass starting zones</p>{stats.routes.map(r=><div className="scout-route" key={r.name}><div><b>{r.name}</b><strong>{fmt(r.per100,3)}</strong></div><div className="scout-track"><span style={{width:`${Math.min(100,Math.abs(r.per100)/Math.max(.001,...stats.routes.map(t=>Math.abs(t.per100)))*100)}%`,background:r.per100<0?'#e7a27b':'#259579'}}/></div><small>{r.attempts} attempts · {fmt(r.completion,1)}% completed</small></div>)}</section><section><h3>Passing network</h3><div className="scout-field"><Field passes={passes} view="network"/></div><p className="scout-muted">Nodes: average pass origin; links: completed passes. These are not player tracking positions.</p></section></div>
      <section className="scout-risk"><h3>Pass risk and attacking reward</h3><div className="scout-risk-grid"><div><small>MODEL-ESTIMATED COMPLETION</small><strong>{meanProbability===null?'Not loaded':`${fmt(meanProbability,1)}%`}</strong><p>Average retrospective pass probability</p></div><div><small>OBSERVED xT / 100 PASSES</small><strong>{stats.total?fmt(stats.net/stats.total*100,3):'—'}</strong><p>Completed-pass threat change, not a goal forecast</p></div></div></section>
      <section className="scout-takeaways"><h3>Three coaching takeaways</h3>{takeaways.map((t,i)=><div className="scout-takeaway" key={i}><b>{String(i+1).padStart(2,'0')}</b><p>{t}</p></div>)}</section>
      <footer className="scout-footer"><strong>METHOD & LIMITATIONS</strong><p>Pass data: StatsBomb Open Data (2018 World Cup when real match selected). xT: Karun Singh's published grid; failed passes contribute zero, with no possession-loss penalty. Model: standardized logistic regression trained on historical events; uses observed pass endpoints (which can differ from intended targets), so this is retrospective analysis, not a live tactical recommendation. Route rankings are descriptive and may be unstable across matches. Source: github.com/hudl/open-data · xT: karun.in/blog/expected-threat.html</p></footer>
    </article>
    <p className="scout-print-help no-print">Choose “Save as PDF” in the print dialog. On Windows: Destination → Save as PDF → Save.</p>
  </div>
}


/** Retrospective display: probability of completion vs zone-based potential threat gain.
 * The end coordinate for a failed pass is recorded after the event, not a known intended target.
 */
function RiskRewardChart({passes,model}:{passes:Pass[];model:PassModel|null}){
 const [hovered,setHovered]=useState<number|null>(null);
 const items=useMemo(()=>!model?[]:passes.filter(validPass).map((p,i)=>({
    id:i,player:p.player,minute:p.minute,completed:p.completed,
    probability:predictCompletion(model,p),
    gain:threat(p.endX,p.endY)-threat(p.x,p.y)
 })),[passes,model]);
 const cutoff=.8;
 const rewardCutoff=.01;
 const counts={smart:0,bold:0,safe:0,wasted:0};
 for(const p of items){if(p.gain>=rewardCutoff){if(p.probability>=cutoff) counts.smart++;else counts.bold++;}else{if(p.probability>=cutoff)counts.safe++;else counts.wasted++;}}
 const left=60,right=756,top=45,bottom=385;
 const minGain=-.05,maxGain=.12;
 const px=(v:number)=>left+Math.max(0,Math.min(1,v))*(right-left);
 const py=(v:number)=>bottom-(Math.max(minGain,Math.min(maxGain,v))-minGain)/(maxGain-minGain)*(bottom-top);
 const dot=items.filter((_,i)=>i%Math.max(1,Math.ceil(items.length/220))===0);
 const selected=hovered===null?null:dot[hovered];
 if(!model) return <section className="panel risk-card" id="risk-reward"><div className="section-overline">RISK VS REWARD</div><h2>Risk vs. Reward Explorer</h2><p>Run <code>python analytics/train_pass_model.py --limit 50</code> and refresh to see recorded passes classified by model-estimated risk and potential xT.</p></section>;
 return <section className="panel risk-card" id="risk-reward">
   <div className="panel-header"><div><div className="section-overline">TACTICAL DECISION ANALYSIS</div><h2>Risk vs. Reward Explorer</h2><p>Each point is a historical pass attempt. Right = easier to complete; up = greater potential attacking threat.</p></div><div className="risk-badge">{items.length} passes analyzed</div></div>
   <div className="risk-chart-scroll"><svg className="risk-chart" viewBox="0 0 800 445" role="img" aria-label="Scatterplot of predicted pass completion against potential expected threat gain, with four quadrants">
     <rect x={left} y={top} width={px(cutoff)-left} height={py(rewardCutoff)-top} fill="#fff4e4"/>
     <rect x={px(cutoff)} y={top} width={right-px(cutoff)} height={py(rewardCutoff)-top} fill="#e5f7ee"/>
     <rect x={left} y={py(rewardCutoff)} width={px(cutoff)-left} height={bottom-py(rewardCutoff)} fill="#fff0ec"/>
     <rect x={px(cutoff)} y={py(rewardCutoff)} width={right-px(cutoff)} height={bottom-py(rewardCutoff)} fill="#f0f4f9"/>
     {[0,.2,.4,.6,.8,1].map(t=><g key={t}><line x1={px(t)} x2={px(t)} y1={top} y2={bottom} stroke="#d0dce5" strokeDasharray="3 7"/><text x={px(t)} y={bottom+21} fontSize="12" textAnchor="middle" fill="#62738c">{Math.round(t*100)}%</text></g>)}
     {[-.05,0,.05,.1].map(t=><g key={t}><line x1={left} x2={right} y1={py(t)} y2={py(t)} stroke="#d0dce5" strokeDasharray="3 7"/><text x={left-9} y={py(t)+4} fontSize="12" textAnchor="end" fill="#62738c">{t.toFixed(2)}</text></g>)}
     <line x1={px(cutoff)} y1={top} x2={px(cutoff)} y2={bottom} stroke="#526b80" strokeWidth="2" strokeDasharray="7 6"/>
     <line x1={left} y1={py(rewardCutoff)} x2={right} y2={py(rewardCutoff)} stroke="#526b80" strokeWidth="2" strokeDasharray="7 6"/>
     <g fill="#233852" fontWeight="700" fontSize="16"><text x="80" y="70">Bold gambles</text><text x="613" y="70">Smart passes</text><text x="80" y="366">Wasted risk</text><text x="604" y="366">Safe but pointless</text></g>
     {dot.map((p,i)=><circle key={p.id} cx={px(p.probability)} cy={py(p.gain)} r={hovered===i?7:3.4} fill={p.completed?'#0d9873':'#e27760'} fillOpacity={hovered===i?1:.5} stroke={hovered===i?'#19354c':'none'} strokeWidth="1.8" onMouseEnter={()=>setHovered(i)} onMouseLeave={()=>setHovered(null)}><title>{`${p.player} · ${p.minute}' · ${p.completed?'Complete':'Incomplete'} · Predicted ${(p.probability*100).toFixed(1)}% · potential ΔxT ${p.gain.toFixed(3)}`}</title></circle>)}
     <text x="407" y="434" textAnchor="middle" fontWeight="600" fontSize="14" fill="#3d5268">Predicted pass completion probability →</text>
     <text transform="translate(16 220) rotate(-90)" textAnchor="middle" fontWeight="600" fontSize="14" fill="#3d5268">Potential ΔxT →</text>
   </svg></div>
   <div className="risk-summary">{[{name:'Smart passes',count:counts.smart},{name:'Bold gambles',count:counts.bold},{name:'Safe but pointless',count:counts.safe},{name:'Wasted risk',count:counts.wasted}].map(item=><div className="risk-count" key={item.name}><strong>{item.count}</strong><span>{item.name}</span></div>)}</div>
   <div className="risk-legend"><span><i className="risk-dot complete"/> Completed</span><span><i className="risk-dot incomplete"/> Incomplete</span><span>Probability split: 80% · high xT gain: ≥0.01</span></div>
   <p className="risk-note">This is an <strong>exploratory, retrospective</strong> display of observed passes, not a coaching prescription. Thresholds are presentation choices, not validated tactical categories. Failed passes may have recorded endpoints different from intended targets; model probabilities do not account for defensive pressure. Plot is sampled to 220 points for legibility; counts include all filtered passes. {selected?`Selected: ${selected.player}, ${Math.round(selected.probability*100)}% estimated completion.`:''}</p>
 </section>;
}


/** Risk-adjusted retrospective pass valuation, NOT a counterfactual model.
 * Failed passes use the *recorded* endpoint, which need not be the intended target.
 * A failed pass is penalized at its starting-zone xT as a simple possession-loss proxy.
 */
function PassValuePanel({passes,model}:{passes:Pass[];model:PassModel|null}){
 const [minimumGain,setMinimumGain]=useState(0.01);
 const rows=useMemo(()=>!model?[]:passes.filter(validPass).map((pass,i)=>{
  const probability=predictCompletion(model,pass);
  const startXt=threat(pass.x,pass.y);
  const endXt=threat(pass.endX,pass.endY);
  const potentialGain=endXt-startXt;
  const estimatedValue=probability*potentialGain-(1-probability)*startXt;
  return {id:i,pass,probability,startXt,potentialGain,estimatedValue};
 }),[passes,model]);
 const missed=useMemo(()=>rows.filter(r=>!r.pass.completed && r.potentialGain>=minimumGain)
  .sort((a,b)=>b.potentialGain-a.potentialGain).slice(0,10),[rows,minimumGain]);
 const meanValue=rows.length?rows.reduce((total,r)=>total+r.estimatedValue,0)/rows.length:0;
 const risky=rows.filter(r=>r.probability<0.8).length;
 return <section className="panel pv-panel" id="pass-value">
  <div className="panel-header"><div><div className="section-overline">PASS VALUATION · EXPLORATORY</div><h2>Pass Value &amp; Unsuccessful High-Potential Attempts</h2><p>Quantifying the trade-off between estimated completion probability and potential attacking threat.</p></div><Target size={22} color="#35b58a"/></div>
  {!model?<div className="analysis-note"><CircleHelp size={19}/><div><strong>Trained model not loaded</strong><p>Run your model training script and ensure <code>frontend/public/pass-model.json</code> is available. No estimated pass values are shown without an actual model.</p></div></div>:<>
   <div className="pv-metrics"><div><small>HISTORICAL PASS ATTEMPTS</small><strong>{rows.length}</strong></div><div><small>MEAN RISK-ADJUSTED VALUE</small><strong>{meanValue>=0?'+':''}{fmt(meanValue,4)}</strong></div><div><small>FLAGGED HIGHER RISK (&lt;80%)</small><strong>{risky}</strong></div><div><small>QUALIFYING FAILED ATTEMPTS</small><strong>{rows.filter(r=>!r.pass.completed && r.potentialGain>=minimumGain).length}</strong></div></div>
   <div className="pv-formula"><strong>Experimental value score</strong><code>V = P(completion) × [xT(end) − xT(start)] − [1 − P(completion)] × xT(start)</code><p>We treat failure as losing the starting-zone threat. This penalty is a simplifying assumption, not a validated measure of turnover cost. Scores are not expected goals or guaranteed coaching gains.</p></div>
   <div className="pv-table-head"><div><h3>Top 10 unsuccessful high-potential passes</h3><p>Recorded failed attempts ordered by potential ΔxT, not invented alternative passes.</p></div><label>Minimum potential ΔxT<select value={minimumGain} onChange={e=>setMinimumGain(Number(e.target.value))}><option value={0}>0.000</option><option value={0.005}>0.005</option><option value={0.01}>0.010</option><option value={0.02}>0.020</option></select></label></div>
   <div className="pv-table-scroll"><table className="pv-table"><thead><tr><th>#</th><th>Player / minute</th><th>Attacking corridor</th><th>Potential ΔxT</th><th>ML completion</th><th>Value score</th></tr></thead><tbody>{missed.map((r,i)=><tr key={r.id}><td>{i+1}</td><td><strong>{r.pass.player}</strong><small>{r.pass.minute}' · failed pass</small></td><td>{zone(r.pass)}</td><td className="pv-positive">+{fmt(r.potentialGain,4)}</td><td>{fmt(r.probability*100,1)}%</td><td>{r.estimatedValue>=0?'+':''}{fmt(r.estimatedValue,4)}</td></tr>)}{missed.length===0&&<tr><td colSpan={6}>No unsuccessful attempts meet this potential-threat threshold for the selected team/player filter.</td></tr>}</tbody></table></div>
   <div className="analysis-note pv-caution"><CircleHelp size={19}/><div><strong>What this does—and does not—mean</strong><p>These were failed passes whose recorded endpoints fall in more threatening zones. Because the recorded endpoint on a failed pass may not equal the intended target, the rankings can be misleading. Review match video before calling a pass a missed opportunity. The model has no defender or pressure context and is not a real-time pass-choice simulator.</p></div></div>
  </>}
 </section>;
}

function App(){const[page,setPage]=useState<'dashboard'|'scouting'|'model-report'>('dashboard');const[matches,setMatches]=useState<Match[]>(demos);const[selected,setSelected]=useState(demos[0].id);const[team,setTeam]=useState(demos[0].home);const[view,setView]=useState('passes');const[query,setQuery]=useState('');const[mobile,setMobile]=useState(false);const[status,setStatus]=useState('');const[passModel,setPassModel]=useState<PassModel|null>(null);const[evaluation,setEvaluation]=useState<ModelEvaluation|null>(null);const[evalError,setEvalError]=useState('');const[xtValidation,setXtValidation]=useState<XtValidation|null>(null);const[xtError,setXtError]=useState('');const[modelError,setModelError]=useState('');const match=matches.find(m=>m.id===selected)||matches[0];useEffect(()=>{setTeam(match.home)},[match.id]);
useEffect(()=>{fetch('/matches.json').then(r=>{if(!r.ok)throw Error();return r.json()}).then((data:Match[])=>{if(Array.isArray(data)&&data.length){setMatches([...data,...demos]);setSelected(data[0].id);setStatus(`Loaded ${data.length} StatsBomb match(es). Showing grid-based xT.`);}}).catch(()=>{})},[]);
useEffect(()=>{fetch('/pass-model.json').then(r=>{if(!r.ok)throw Error('No trained model exported yet');return r.json()}).then((model:PassModel)=>{if(model.trained&&model.coefficients?.length===6&&model.means?.length===6&&model.scales?.length===6){setPassModel(model)}else{setModelError('Model export format invalid')}}).catch(()=>setModelError('Train the model to enable AI Pass Intelligence'))},[]);
 useEffect(()=>{fetch('/model-evaluation.json').then(r=>{if(!r.ok)throw Error('Missing evaluation export');return r.json()}).then((data:ModelEvaluation)=>{if(data?.results?.confusion_matrix?.length===2&&data.calibration_bins){setEvaluation(data)}else{setEvalError('Evaluation export has an unexpected format')}}).catch(()=>setEvalError('Run the updated Python evaluation script and refresh the dashboard.'))},[]);
 useEffect(()=>{fetch('/xt-validation.json').then(r=>{if(!r.ok)throw Error('Missing xT validation export');return r.json()}).then((data:XtValidation)=>{if(typeof data?.shot_discrimination_auc==='number'&&Array.isArray(data.quintiles)&&data.quintiles.length>0){setXtValidation(data)}else{setXtError('xT validation JSON format is unexpected')}}).catch(()=>setXtError('Run the xT validation script and ensure frontend/public/xt-validation.json exists.'))},[]);
const filtered=useMemo(()=>match.passes.filter(p=>p.team===team&&validPass(p)&&(!query||p.player.toLowerCase().includes(query.toLowerCase()))),[match,team,query]);const stats=useMemo(()=>analyze(filtered),[filtered]);const best=[...stats.routes].filter(r=>r.attempts>=20).sort((a,b)=>b.per100-a.per100)[0];const mlRoutes=useMemo(()=>passModel?stats.routes.map(r=>{const group=filtered.filter(p=>zone(p)===r.name);const predicted=group.map(p=>predictCompletion(passModel,p));const mean=predicted.length?predicted.reduce((a,b)=>a+b,0)/predicted.length:0;const expectedXt=group.reduce((sum,p,i)=>sum+predicted[i]*(threat(p.endX,p.endY)-threat(p.x,p.y)),0);return {...r,predictedPct:mean*100,expectedXtPer100:group.length?expectedXt/group.length*100:0}}):[],[passModel,filtered,stats]);
const maxBar=Math.max(.001,...stats.routes.map(r=>Math.abs(r.per100)));const sample=match.source==='demo';
function exportCsv(){const header=['team','player','recipient','minute','x','y','endX','endY','completed'];const csv=[header.join(','),...filtered.map(p=>header.map(k=>String(p[k as keyof Pass])).join(','))].join('\n');const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));link.download='tactiq_passes.csv';link.click();URL.revokeObjectURL(link.href)}
return <div className="shell"><aside className={'sidebar '+(mobile?'shown':'')}><div className="brand"><div className="brand-icon"><Activity size={24}/></div><div>TactiQ <span>AI</span><small>FOOTBALL INTELLIGENCE</small></div></div><div className="nav-label">WORKSPACE</div><a className={"nav "+(page==='dashboard'?'active':'')} href="#overview" onClick={()=>setPage('dashboard')}><LayoutDashboard size={18}/> Tactical overview</a><a className={"nav "+(page==='scouting'?'active':'')} href="#scouting" onClick={(e)=>{e.preventDefault();setPage('scouting')}}><Target size={18}/> Scouting Report</a><a className={"nav "+(page==='model-report'?'active':'')} href="#model-report" onClick={(e)=>{e.preventDefault();setPage('model-report');setMobile(false)}}><ShieldCheck size={18}/> Model Report</a><a className="nav" href="#field"><Target size={18}/> Match explorer</a><a className="nav" href="#routes"><GitBranch size={18}/> Route analytics</a><div className="nav-label nav-gap">PROJECT</div><a className="nav" href="https://github.com/hudl/open-data" target="_blank"><Layers size={18}/> Data source <ExternalLink size={14}/></a><div className="sidebar-bottom"><div className="source-pill"><span className="pulse"/> {sample?'Sample dataset':'StatsBomb match data'}</div><p>Build v0.1 · AQX Data Bowl 4.0</p></div></aside>
<main className="main" id="overview"><header className="topbar"><button className="mobile-toggle" onClick={()=>setMobile(!mobile)}><Menu size={22}/></button><div className="crumb">Workspace <span>/</span> <strong>{page==='dashboard'?'Tactical overview':page==='scouting'?'Scouting Report':'Model Report'}</strong></div><div className="top-right"><span className="status-dot"/> Analysis workspace <div className="avatar">TQ</div></div></header><div className="content">{page==='scouting'?<ScoutingReport matches={matches} selected={selected} onMatch={setSelected} team={team} onTeam={setTeam} model={passModel}/>:page==='model-report'?<ModelReport data={evaluation} model={passModel} error={evalError} xtValidation={xtValidation} xtError={xtError}/>:<><div className="headline"><div><div className="eyebrow">MATCH INTELLIGENCE PLATFORM <span className="eyebrow-line"/></div><h1>See beyond the score.</h1><p>Understand the patterns behind every attack — from passing connections to spatial threat.</p></div><button className="outline-btn" onClick={exportCsv}><Download size={17}/> Export passes</button></div>
{sample&&<div className="notice"><Info size={18}/><div><strong>Illustrative demo data</strong> — values are generated for UI testing, not results from real football matches. Run the bundled StatsBomb importer to replace them. The xT grid is published, but these sample matches and metrics are illustrative.</div></div>}{status&&<div className="loaded">{status}</div>}
<section className="selectors"><label>SELECT MATCH <span><ChevronDown size={15}/></span><select value={selected} onChange={e=>setSelected(e.target.value)}>{matches.map(m=><option key={m.id} value={m.id}>{m.label}{m.source==='demo'?' (demo)':''}</option>)}</select></label><label>ANALYZE TEAM <span><ChevronDown size={15}/></span><select value={team} onChange={e=>setTeam(e.target.value)}><option>{match.home}</option><option>{match.away}</option></select></label><div className="competition"><span>COMPETITION</span><strong>{match.competition}</strong></div><div className="competition date"><span>MATCH DATE</span><strong>{match.date}</strong></div></section>
<section className="stats-grid"><div className="stat"><div className="stat-top">Pass attempts <span className="stat-icon"><Activity size={20}/></span></div><div className="stat-value">{stats.total.toLocaleString()}</div><div className="stat-sub">Selected team and player filter</div></div><div className="stat"><div className="stat-top">Pass completion <span className="stat-icon"><ShieldCheck size={20}/></span></div><div className="stat-value">{fmt(stats.rate,1)}<small>%</small></div><div className="stat-sub">{stats.complete} successful passes</div></div><div className="stat"><div className="stat-top">Progressive passes <span className="stat-icon"><MoveUpRight size={20}/></span></div><div className="stat-value">{stats.forward}</div><div className="stat-sub">Successful passes gaining &gt;10 units</div></div><div className="stat featured"><div className="stat-top">Net completed-pass xT <span className="stat-icon"><TrendingUp size={20}/></span></div><div className="stat-value">{stats.net>=0?'+':''}{fmt(stats.net,2)}</div><div className="stat-sub">Sum of completed-pass ΔxT</div></div></section>
<RiskRewardChart passes={filtered} model={passModel}/>
<PassValuePanel passes={filtered} model={passModel}/>
<section className="dashboard-grid" id="field"><div className="panel pitch-panel"><div className="panel-header"><div><div className="section-overline">SPATIAL EXPLORER</div><h2>Attacking movement map</h2><p>{view==='network'?'Player connections from completed passes':view==='zones'?'Published 12 × 8 xT surface':'Up to 85 completed forward passes'} · 120 × 80 pitch</p></div><div className="segmented"><button className={view==='passes'?'on':''} onClick={()=>setView('passes')}>Passes</button><button className={view==='zones'?'on':''} onClick={()=>setView('zones')}>xT zones</button><button className={view==='network'?'on':''} onClick={()=>setView('network')}>Network</button></div></div><div className="pitch-wrap"><Field passes={filtered} view={view}/></div><div className="pitch-foot"><span><i className="legend lime"/> Forward completed passes (sampled)</span><span><i className="legend gold"/> xT grid: Karun Singh</span><span>Network nodes = average pass origin (not tracking)</span></div></div><div className="panel insights"><div className="panel-header"><div><div className="section-overline">INTELLIGENCE PANEL</div><h2>Tactical insights</h2><p>Evidence-based summaries</p></div><div className="lightbulb"><Lightbulb size={19}/></div></div><div className="insight-card"><div className="insight-label"><span className="pulse"/> TOP OBSERVED ROUTE (≥20 PASSES)</div><h3>{best?.name||'Not enough data'}</h3><p>{best?`This starting corridor leads the current sample at ${fmt(best.per100,3)} xT units per 100 attempted passes.`:'Load at least 20 passes in a corridor to compare routes.'}</p><div className="insight-footer"><span>{best?.attempts||0} attempts</span><span>{best?fmt(best.completion,1):'—'}% completed</span></div></div><div className="analysis-note"><CircleHelp size={19}/><div><strong>How to interpret this</strong><p>Each completed pass contributes xT(end) − xT(start); failed passes contribute 0 (no turnover cost). Negative values are possible. The highest route is an observed association, not proof of a better tactical decision. Rankings need multi-match validation and uncertainty estimates.</p></div></div><button className="learn-btn" onClick={()=>document.getElementById('routes')?.scrollIntoView({behavior:'smooth'})}>Explore route comparison <ArrowRight size={16}/></button></div></section>
<section className="panel" id="ai-model" style={{marginTop:24,padding:28}}><div className="panel-header"><div><div className="section-overline">MACHINE LEARNING · PASS INTELLIGENCE</div><h2>AI-powered passing analysis</h2><p>Historical completion probability estimated by a trained logistic regression model</p></div><div className="lightbulb"><Lightbulb size={19}/></div></div>
{passModel? <><div className="route-list">{mlRoutes.map(r=><div className="route-row" key={r.name}><div className="route-top"><strong>{r.name}</strong><span>{fmt(r.predictedPct,1)}% <small>predicted completion</small></span></div><div className="bar-bg"><div className="bar-fill" style={{width:`${r.predictedPct}%`,background:'#63d4b0'}}/></div><div className="route-caption">{r.attempts} attempts · observed {fmt(r.completion,1)}% · model-weighted ΔxT /100: {fmt(r.expectedXtPer100,3)}</div></div>)}</div><div className="analysis-note" style={{marginTop:16}}><CircleHelp size={19}/><div><strong>Held-out match evaluation</strong><p>ROC-AUC {fmt(passModel.metrics.roc_auc,3)} · Brier score {fmt(passModel.metrics.brier,3)} (prior-only baseline {fmt(passModel.metrics.baseline_brier,3)}) · {passModel.metrics.test_matches} test matches, {passModel.metrics.test_passes.toLocaleString()} test passes.</p><p>Actual results must improve on baseline before describing the predictions as effective. Model uses recorded endpoints, not intended targets, and does not account for defenders or pressure. Model-weighted ΔxT is an exploratory score, not a causal forecast of tactical changes.</p></div></div></>:<div className="analysis-note"><CircleHelp size={19}/><div><strong>Trained model not loaded</strong><p>{modelError || 'No trained model file found.'} Run <code>python analytics/train_pass_model.py --limit 50</code> in the project root, then refresh. This dashboard does not pretend that a model exists before training.</p></div></div>}
</section>
 {evaluation?<EvaluationPanel data={evaluation}/>:<section className="panel" style={{marginTop:24,padding:24}}><h2>AI Model Performance</h2><p>{evalError||'Loading model evaluation…'}</p></section>}
<section className="bottom-grid" id="routes"><div className="panel route-panel"><div className="panel-header"><div><div className="section-overline">PATTERN COMPARISON</div><h2>Attacking corridors</h2><p>Net completed-pass xT per 100 attempts · signed values</p></div><BarChart3 size={19} color="#8f9bad"/></div><div className="route-list">{stats.routes.map(r=><div className="route-row" key={r.name}><div className="route-top"><strong>{r.name}</strong><span>{fmt(r.per100,3)} <small>xT /100</small></span></div><div className="bar-bg"><div className="bar-fill" style={{width:`${Math.abs(r.per100)/maxBar*100}%`,background:r.per100<0?'#ed9a80':'#63d4b0'}}/></div><div className="route-caption">{r.attempts} attempts <span>·</span> {fmt(r.completion,1)}% completed</div></div>)}</div></div><div className="panel filter-panel"><div className="panel-header"><div><div className="section-overline">EXPLORE EVENTS</div><h2>Pass explorer</h2><p>Inspect which players contributed to these results</p></div><Filter size={18} color="#8f9bad"/></div><div className="search"><Search size={17}/><input placeholder="Filter by player name..." value={query} onChange={e=>setQuery(e.target.value)}/>{query&&<button onClick={()=>setQuery('')} aria-label="Clear"><X size={15}/></button>}</div><div className="event-list">{filtered.slice(0,5).map((p,i)=><div className="event" key={i}><div className={'event-icon '+(p.completed?'success':'failed')}>{p.completed?<ArrowRight size={17}/>:<X size={15}/>}</div><div><strong>{p.player}</strong><small>{p.minute}' · {zone(p)}</small></div><span className={p.completed?'success-text':'failed-text'}>{p.completed?'Completed':'Incomplete'}</span></div>)}{!filtered.length&&<p className="empty">No passes match your filter.</p>}</div></div></section><footer><span>TactiQ AI · Analytics prototype</span><span>Real-match data © StatsBomb when loaded · <a href="https://github.com/hudl/open-data" target="_blank" rel="noreferrer">Source and usage terms ↗</a> · <a href="https://karun.in/blog/expected-threat.html" target="_blank" rel="noreferrer">xT model ↗</a></span></footer></> }</div></main></div>}
export default App;
