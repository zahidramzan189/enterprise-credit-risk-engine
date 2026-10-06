const API_BASE = location.port === "8080"
    ? "http://localhost:8000"
    : "https://enterprise-credit-risk-engine.onrender.com";
const API={async request(path,options={}){let r;try{r=await fetch(API_BASE+path,options)}catch(e){throw Error("Backend unavailable. Start FastAPI and try again.")}let d={};try{d=await r.json()}catch(e){}if(!r.ok)throw Error(d.detail||`Request failed (${r.status})`);return d},
checkHealth(){return this.request("/health")},getModelInfo(){return this.request("/api/v1/model-info")},
predictCreditRisk(p){return this.request("/api/v1/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(p)})},
explainCreditRisk(p){return this.request("/api/v1/explain",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(p)})}};
