/ Command-line entry point. Run from the repo root:
/   q q/run.q -data data -out out -lags 6 -neutral 0
\l q/factorlab.q

a:.Q.opt .z.x;
arg:{[a;k;d] $[k in key a;first a k;d]}[a];
datadir:arg[`data;"data"];
outdir:arg[`out;"out"];
lags:"J"$arg[`lags;"6"];
neutral:"1"~arg[`neutral;"0"];

t0:.z.p;
r:.fp.run[datadir;outdir;lags;neutral];
-1 "Fama-MacBeth factor premia (monthly), Newey-West lags=",string[lags],", sector-neutral=",string neutral;
show r`summary;
-1 "elapsed: ",string .z.p-t0;
exit 0
