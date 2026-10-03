let distributionChart,trendChart;
function renderRiskCharts(h){const c={LOW:0,MEDIUM:0,HIGH:0};h.forEach(x=>{if(c[x.risk_level]!==undefined)c[x.risk_level]++});
const a=document.getElementById("riskDistribution");if(a)distributionChart=new Chart(a,{type:"doughnut",data:{labels:["Low","Medium","High"],datasets:[{data:[c.LOW,c.MEDIUM,c.HIGH]}]},options:{plugins:{legend:{labels:{color:"#8fa0b5"}}}}});
const b=document.getElementById("riskTrend");if(b){const r=h.slice(0,12).reverse();trendChart=new Chart(b,{type:"line",data:{labels:r.map(x=>new Date(x.timestamp).toLocaleDateString()),datasets:[{label:"Default probability %",data:r.map(x=>x.default_probability_percent),tension:.3}]},options:{scales:{x:{ticks:{color:"#8fa0b5"}},y:{ticks:{color:"#8fa0b5"}}}}})}}
