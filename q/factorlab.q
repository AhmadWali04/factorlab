/ factorlab.q -- equity factor research in kdb+/q
/
/ Pipeline: daily prices (+ point-in-time fundamentals)
/   -> month-end panel of characteristics (as-of join for book value)
/   -> cross-sectional winsorize + z-score (optionally sector-neutral)
/   -> quintile portfolios, rank IC, Fama-MacBeth regression with Newey-West t-stats
/
/ Every transform has an exact twin in factorlab/xsec.py and factorlab/engine.py,
/ so results can be checked number-for-number against the numpy engine.
/
/ Usage from the repo root:   q q/run.q -data data -out out -lags 6 -neutral 0

\d .fp

WINSOR:0.01

/ ---------- cross-sectional helpers ----------
winsor:{[p;x] s:asc x; n:count s; k:floor p*n-1; s[k]|s[n-1+k]&x}  / symmetric: k values clipped per tail
zscore:{(x-avg x)%dev x}                                         / dev = population sd
prep:{zscore winsor[WINSOR;x]}
qbin:{1+floor 5*(rank x)%count x}                                / quintile 1..5
spearman:{cor[`float$rank x;`float$rank y]}

/ ---------- loading ----------
loadPrices:{[f] `sym`date xasc ("DSFJJS";enlist csv) 0: f}       / date,sym,close,volume,shares,sector
loadFund:{[f] `sym`date xasc ("DSF";enlist csv) 0: f}            / date(public),sym,bps

/ ---------- month-end characteristic panel ----------
/ returns (panel; factor list). Formation at month-end t; fwd = return over month t+1.
panel:{[px;fd]
  px:update lret:log close%prev close by sym from px;
  px:update vol:sqrt[252]*63 mdev lret by sym from px;              / 63-day realized vol
  me:0!select date:last date, close:last close, shares:last shares, sector:last sector, vol:last vol
      by sym, month:`month$date from px;
  me:`sym`month xasc me;
  me:update rev:-1+close%prev close,                                / 1-month reversal
            mom:-1+(prev close)%12 xprev close,                     / 12-1 momentum
            fwd:-1+(next close)%close by sym from me;               / next-month return
  me:update size:log close*shares from me;
  hasF:98h=type fd;
  if[hasF; me:aj[`sym`date;me;fd]; me:update btp:log bps%close from me];  / book as known on that date
  fs:`mom`btp`size`vol`rev where 10111b|01000b&hasF;
  me:?[me;{(not;(null;x))} each fs,`fwd;0b;()];
  (me;fs)}

/ winsorize + z-score each factor within each month; optional sector neutralization
standardize:{[t;fs;neutral]
  zs:`$"z",/:string fs;
  t:![t;();(enlist`month)!enlist`month;zs!{(prep;x)} each fs];
  if[neutral;
    t:![t;();`month`sector!`month`sector;zs!{(-;x;(avg;x))} each zs];
    t:![t;();(enlist`month)!enlist`month;zs!{(zscore;x)} each zs]];
  t}

/ ---------- portfolio sorts and IC ----------
quintiles:{[t;z;f]
  t:![t;();(enlist`month)!enlist`month;(enlist`bucket)!enlist(qbin;z)];
  r:0!select ret:avg fwd, n:count i by month, bucket from t;
  update factor:f from r}

ictab:{[t;z;f]
  r:0!?[t;();(enlist`month)!enlist`month;(enlist`ic)!enlist(spearman;z;`fwd)];
  update factor:f from r}

/ ---------- Fama-MacBeth ----------
/ one cross-section: y = X'b + e, X is k x n (rows = intercept, factors). returns b, R^2
xsreg:{[y;X]
  b:inv[X mmu flip X] mmu X mmu y;
  e:y-b mmu X;
  b,1-(e wsum e)%{x wsum x} y-avg y}

/ run xsreg every month -> table of gammas (factor returns) + R^2
fmb:{[t;zs]
  g:0!?[t;();(enlist`month)!enlist`month;(`y,zs)!`fwd,zs];
  est:{[zs;r] y:r`y; xsreg[y;(enlist count[y]#1f),r zs]}[zs] each g;
  flip (`month`const,zs,`r2)!(enlist g`month),flip est}

/ Newey-West (Bartlett kernel) standard error of a time-series mean
nw:{[x;L]
  n:count x; m:avg x; e:x-m;
  g:{[e;n;l] (e[l _ til n] wsum e til n-l)%n}[e;n] each til 1+L;
  w:1f,2*1-(1+til L)%L+1;
  se:sqrt (w wsum g)%n;
  `mean`se`t`pos!(m;se;m%se;avg x>0)}

summary:{[gam;zs;fs;L] ([] factor:fs),'nw[;L] each gam zs}

/ ---------- output ----------
write:{[dir;name;t]
  if[`month in cols t; t:update month:`date$month from t];
  (hsym`$dir,"/",string[name],".csv") 0: csv 0: t}

run:{[datadir;outdir;L;neutral]
  px:loadPrices hsym`$datadir,"/prices.csv";
  ff:hsym`$datadir,"/fundamentals.csv";
  fd:$[()~key ff;();loadFund ff];
  pf:panel[px;fd]; fs:pf 1;
  p:standardize[pf 0;fs;neutral];
  zs:`$"z",/:string fs;
  gam:fmb[p;zs];
  res:`summary`gammas`quintiles`ic!(summary[gam;zs;fs;L];gam;raze quintiles[p]'[zs;fs];raze ictab[p]'[zs;fs]);
  write[outdir]'[key res;value res];
  res}

\d .
