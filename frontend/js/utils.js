const HISTORY_KEY="enterpriseCreditRiskHistory";
function getHistory(){try{return JSON.parse(localStorage.getItem(HISTORY_KEY)||"[]")}catch(e){return[]}}
function savePrediction(x){const h=getHistory();h.unshift(x);localStorage.setItem(HISTORY_KEY,JSON.stringify(h.slice(0,100)))}
function clearHistory(){localStorage.removeItem(HISTORY_KEY)}
function riskClass(x){return String(x||"").toLowerCase()}
function esc(x){const d=document.createElement("div");d.textContent=String(x??"");return d.innerHTML}
function formatPercent(x){return `${Number(x).toFixed(2)}%`}
