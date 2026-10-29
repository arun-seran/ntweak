const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { JSDOM } = require("jsdom");

const root = path.join(__dirname, "../ntweak/static");
const example = { name: "Imported recipe", model: "Qwen/Qwen3.5-4B", source: "", neurons: [{layer: 22, neuron: 123, dose: -0.7, mode: "add", enabled: true}], sampling: {temperature: 0.7, top_p: 0.95, min_p: 0.05, repetition_penalty: 1.1, max_new_tokens: 512, seed: 42} };
const delay = () => new Promise(resolve => setTimeout(resolve, 10));
async function until(fn) { for(let i=0;i<80;i++) { if(fn()) return; await delay(); } assert.ok(fn(), "UI condition did not become true"); }

function setup() {
  const dom = new JSDOM(fs.readFileSync(path.join(root, "index.html"), "utf8"), {url:"http://127.0.0.1:8787", runScripts:"outside-only"});
  const w=dom.window, calls=[];
  w.structuredClone=structuredClone; w.TextDecoder=TextDecoder;
  w.URL.createObjectURL=()=>"blob:test"; w.URL.revokeObjectURL=()=>{};
  w.HTMLAnchorElement.prototype.click=function(){};
  if(!w.HTMLDialogElement.prototype.showModal) w.HTMLDialogElement.prototype.showModal=function(){this.open=true;};
  if(!w.HTMLDialogElement.prototype.close) w.HTMLDialogElement.prototype.close=function(){this.open=false;};
  const model={id:"qwen-test",repo:"Qwen/Qwen3.5-4B",revision:"a".repeat(40),path:"/tmp/test-model",source:"Downloaded in Ntweak",available:true};
  let server={runtime:{state:"idle",model:null,device:null,layers:0,widths:[],steering_supported:false}, dependencies:{ready:true},models:[],jobs:[],data_dir:"/tmp/test-data"};
  w.fetch=async (url,options={})=>{
    const body=options.body?JSON.parse(options.body):undefined; calls.push({url,body});
    let data;
    if(url==="/api/status") data=structuredClone(server);
    else if(url==="/api/recipes/import") data=body.url?structuredClone(example):{...structuredClone(example),...body.recipe, sampling:{...example.sampling,...body.recipe.sampling}};
    else if(url==="/api/models/inspect") data={repo:body.repo,commit:"a".repeat(40),files:[{name:"config.json",size:100,selected:true},{name:"model.safetensors",size:2048,selected:true},{name:"pytorch_model.bin",size:2048,selected:false}]};
    else if(url==="/api/models/download") {server.models=[model];data={id:"download-test"};}
    else if(url==="/api/models/load") {server.runtime={state:"ready",model,device:"cpu",layers:32,widths:Array(32).fill(9216),steering_supported:true};data={id:"load-test"};}
    else if(url==="/api/chat") {
      const events=[{type:"phase",message:"Generating locally"},{type:"calibration",neurons:body.steering?[{layer:22,neuron:123,dose:-0.7,mode:"add",residual_median:10,column_norm:2,activation_delta:-3.5}]:[],input_tokens:10},{type:"token",text:body.steering?"Steered ☀ reply":"Baseline reply"},{type:"done",model:model.repo,output_tokens:3,tokens_per_second:5,cancelled:false}];
      const encoded=new TextEncoder().encode(events.map(x=>JSON.stringify(x)).join("\n")+"\n");
      return new Response(new ReadableStream({start(controller){for(let i=0;i<encoded.length;i+=7)controller.enqueue(encoded.slice(i,i+7));controller.close();}}),{headers:{"Content-Type":"application/x-ndjson"}});
    } else if(url==="/api/models/unload") {server.runtime={state:"idle",model:null};data={ok:true};}
    else throw new Error("Unexpected request: "+url);
    return new Response(JSON.stringify(data),{headers:{"Content-Type":"application/json"}});
  };
  w.eval(fs.readFileSync(path.join(root,"app.js"),"utf8"));
  return {w,calls,dom,$:id=>w.document.getElementById(id),ready:()=>until(()=>!w.document.getElementById("connection").textContent.includes("Connecting"))};
}

test("model library inspects files, selects weights, downloads and loads", async()=>{
  const ctx=setup(); const {$,w,calls}=ctx;
  try {
    await ctx.ready(); assert.equal($("send").disabled,true);
    $("openModels").click(); assert.equal($("modelsDialog").open,true);
    $("inspectQwen").click(); await until(()=>!$("downloadFiles").hidden);
    assert.equal(w.document.querySelectorAll(".file-row input:checked").length,2);
    $("selectNone").click(); assert.equal($("downloadButton").disabled,true);
    $("selectRecommended").click(); assert.equal($("downloadButton").disabled,false);
    $("downloadButton").click(); await until(()=>w.document.querySelector(".local-model button"));
    const download=calls.find(x=>x.url==="/api/models/download");
    assert.deepEqual(download.body.files,["config.json","model.safetensors"]);
    assert.equal(download.body.revision,"a".repeat(40));
    w.document.querySelector(".local-model button").click(); await until(()=>!$("send").disabled);
    assert.equal($("activeModel").textContent,"Qwen/Qwen3.5-4B");
  } finally {ctx.dom.window.close();}
});

test("recipe import, enabled controls, local save and streamed comparison keep separate histories",async()=>{
  const ctx=setup();const {$,w,calls}=ctx;
  try {
    await ctx.ready();
    $("openImport").click(); $("importButton").click();
    await until(()=>$("recipeName").value==="Imported recipe");
    assert.equal(w.document.querySelectorAll(".tweak-card").length,1);
    assert.equal($("steeringSummary").textContent.includes("1 active tweak"),true);
    const toggle=w.document.querySelector(".neuron-label input");toggle.click();
    assert.equal($("steeringSummary").textContent.includes("No active"),true);toggle.click();
    $("saveRecipe").click(); await until(()=>$("savedRecipes").options.length===2);
    $("inspectQwen").click();await until(()=>!$("downloadFiles").hidden);
    $("downloadButton").click();await until(()=>w.document.querySelector(".local-model button"));
    w.document.querySelector(".local-model button").click();await until(()=>!$("send").disabled);
    $("compare").click(); $("prompt").value="hello";$("send").click();
    await until(()=>w.document.querySelectorAll(".response-stats")[1]?.textContent.includes("3 tokens") && !$("send").hidden);
    assert.equal(w.document.querySelector(".responses").classList.contains("comparing"),true);
    assert.equal(w.document.querySelectorAll(".response .message-text")[1].textContent,"Steered ☀ reply");
    $("prompt").value="again";$("send").click();
    await until(()=>calls.filter(x=>x.url==="/api/chat").length===4 && !$("send").hidden);
    const chats=calls.filter(x=>x.url==="/api/chat");
    assert.equal(chats[2].body.messages[1].content,"Baseline reply");
    assert.equal(chats[3].body.messages[1].content,"Steered ☀ reply");
    const stored=JSON.parse(w.localStorage.getItem("ntweak.workspace.v1"));
    assert.equal(stored.turns.length,2);assert.equal(stored.turns[0].steered.view,undefined);
    assert.equal(stored.turns[0].steered.calibration.neurons[0].activation_delta,-3.5);
    $("newChat").click();assert.equal(w.document.querySelectorAll(".turn").length,0);
    assert.equal($("recipeName").value,"Imported recipe");
  } finally {ctx.dom.window.close();}
});

test("untrusted recipe text is rendered as text, without creating markup",async()=>{
  const ctx=setup();const {$,w}=ctx;
  try {
    await ctx.ready();$("openImport").click();$("jsonTab").click();
    $("recipeJson").value=JSON.stringify({...example,name:"<img src=x onerror=alert(1)>",neurons:[{...example.neurons[0],label:"<script>bad()</script>"}]});
    $("importButton").click();await until(()=>$("recipeName").value.startsWith("<img"));
    assert.equal(w.document.querySelector(".neuron-label").textContent,"<script>bad()</script>");
    assert.equal(w.document.querySelectorAll(".tweak-card script").length,0);
  } finally {ctx.dom.window.close();}
});
