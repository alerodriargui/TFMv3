(() => {
  const canvas = document.querySelector('#daeCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const epochSlider = document.querySelector('#vizEpoch');
  const playButton = document.querySelector('#vizPlay');
  const motionReduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  // Validation MSE from the recorded seed-42 run, also used by the results chart.
  const validationLoss = [0.007729,0.006426,0.005913,0.005353,0.005324,0.004896,0.004710,0.004502,0.004253,0.003920,0.004214,0.004349,0.004059,0.004356,0.004820,0.003983,0.003790,0.003920,0.003649,0.003772,0.004252,0.004047,0.003840,0.003751,0.003338,0.003994,0.003477,0.003746,0.003561,0.003766,0.003562,0.003640,0.003257,0.003682,0.003604,0.003506,0.003706,0.003186,0.003356,0.003336,0.003669,0.003264,0.003416,0.003784,0.003274,0.003514,0.002984,0.003307,0.003445,0.003286];
  const images = {};
  for (const [key, path] of Object.entries({ noisy:'assets/brain-normal-coarse-noise.png', clean:'assets/brain-normal-model-input.png', anomalous:'assets/brain-anomaly-model-input.png' })) {
    const image = new Image(); image.src = path; image.onload = () => draw(performance.now()); images[key] = image;
  }
  const stages = [
    {x:259,y:170,w:55,h:120,label:'128²',channels:'64 ch',color:'#70dbc9'},
    {x:359,y:190,w:49,h:100,label:'64²',channels:'128 ch',color:'#70dbc9'},
    {x:453,y:212,w:42,h:76,label:'32²',channels:'256 ch',color:'#70dbc9'},
    {x:538,y:232,w:47,h:55,label:'16²',channels:'512 ch',color:'#c8ff67'},
    {x:628,y:212,w:42,h:76,label:'32²',channels:'256 ch',color:'#b9ff47'},
    {x:719,y:190,w:49,h:100,label:'64²',channels:'128 ch',color:'#b9ff47'},
    {x:814,y:170,w:55,h:120,label:'128²',channels:'64 ch',color:'#b9ff47'}
  ];
  let mode = 'train', playing = !motionReduced, visible = true, epoch = 1, lastTick = 0;
  const setEpoch = value => {
    epoch = Math.max(1, Math.min(50, Number(value)));
    epochSlider.value = epoch;
    document.querySelector('#vizEpochValue').textContent = `Época ${epoch} / 50`;
    document.querySelector('#vizLossValue').textContent = validationLoss[epoch-1].toFixed(6).replace('.', ',');
    draw(performance.now());
  };
  const setPlaying = value => {
    playing = value;
    playButton.innerHTML = value ? '⏸ <span>Pausar</span>' : '▶ <span>Reproducir</span>';
    playButton.setAttribute('aria-label', value ? 'Pausar animación' : 'Reproducir animación');
    if (value) requestAnimationFrame(frame);
  };
  function line(x1,y1,x2,y2,color,width=1) {
    ctx.strokeStyle=color; ctx.lineWidth=width; ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
  }
  function label(text,x,y,color='#728c81',size=11,align='left') {
    ctx.fillStyle=color; ctx.font=`${size}px "DM Mono", monospace`; ctx.textAlign=align; ctx.fillText(text,x,y);
  }
  function scan(image,x,y,caption) {
    ctx.fillStyle='#030806'; ctx.fillRect(x,y,147,147);
    ctx.strokeStyle='#2d5447'; ctx.strokeRect(x+.5,y+.5,146,146);
    if (image?.complete && image.naturalWidth) {
      ctx.imageSmoothingEnabled=false; ctx.drawImage(image,x+8,y+8,131,131); ctx.imageSmoothingEnabled=true;
    }
    label(caption,x+73.5,y+170,'#a2b6a9',11,'center');
  }
  function illustrativeResidual(x,y) {
    ctx.fillStyle='#030806';ctx.fillRect(x,y,147,147);
    ctx.strokeStyle='#2d5447';ctx.strokeRect(x+.5,y+.5,146,146);
    // Conceptual residue, not a heatmap computed from the checkpoint.
    for(let row=0;row<22;row++) for(let col=0;col<22;col++) {
      const dx=(col-15)/5,dy=(row-10)/4;
      const glow=Math.exp(-(dx*dx+dy*dy));
      if(glow>.10) {
        ctx.fillStyle=`rgba(255,${Math.round(70+glow*115)},64,${Math.min(.95,glow*.8)})`;
        ctx.fillRect(x+8+col*6,y+8+row*6,6,6);
      }
    }
    label('RESIDUO · ESQUEMA',x+73.5,y+170,'#a2b6a9',11,'center');
  }
  function draw(time) {
    const w=canvas.width,h=canvas.height;
    ctx.fillStyle='#081510'; ctx.fillRect(0,0,w,h);
    ctx.strokeStyle='#1a3127'; ctx.lineWidth=1;
    for(let x=0;x<w;x+=22) for(let y=0;y<h;y+=22) {
      ctx.fillStyle='#1a3328'; ctx.fillRect(x,y,2,2);
    }
    label(mode==='train'?'01 / RESTAURAR UN CORTE SANO':'02 / ANALIZAR UN CORTE DESCONOCIDO',38,38,'#b9ff47',12);
    label('ESQUEMA ILUSTRATIVO · NO ACTIVACIONES REALES',1062,38,'#668276',10,'right');
    scan(images[mode==='train'?'noisy':'anomalous'],35,156,mode==='train'?'NORMAL + RUIDO':'CORTE DESCONOCIDO');
    if(mode==='train') scan(images.clean,916,156,'OBJETIVO LIMPIO');
    else illustrativeResidual(916,156);
    // Moving signals travel through the actual U-Net stages, including three skip links.
    const route=[{x:184,y:230},...stages.map(s=>({x:s.x+s.w/2,y:s.y+s.h/2})),{x:916,y:230}];
    for(let i=0;i<route.length-1;i++) {
      const a=route[i],b=route[i+1]; line(a.x,a.y,b.x,b.y,'#315c48',2);
      if(playing && visible && !motionReduced) {
        const p=(time/1150-i*.18)%1;
        ctx.fillStyle=i<4?'#75ded0':'#c3fa6a'; ctx.shadowColor=ctx.fillStyle; ctx.shadowBlur=12;
        ctx.fillRect(a.x+(b.x-a.x)*p-3,a.y+(b.y-a.y)*p-3,6,6); ctx.shadowBlur=0;
      }
    }
    [[0,6,102],[1,5,133],[2,4,165]].forEach(([a,b,height],i)=>{
      const left=stages[a].x+stages[a].w/2,right=stages[b].x+stages[b].w/2;
      ctx.strokeStyle='#51755e'; ctx.lineWidth=1; ctx.setLineDash([5,5]);
      ctx.beginPath(); ctx.moveTo(left,stages[a].y);ctx.lineTo(left,height);ctx.lineTo(right,height);ctx.lineTo(right,stages[b].y);ctx.stroke();ctx.setLineDash([]);
      if(i===0) label('SKIP CONNECTIONS',550,88,'#72947d',10,'center');
    });
    stages.forEach((s,i)=>{
      for(let n=3;n>=0;n--) { ctx.fillStyle='#0b211a'; ctx.fillRect(s.x+n*4,s.y-n*5,s.w,s.h);ctx.strokeStyle=s.color;ctx.globalAlpha=n===0?.85:.18;ctx.strokeRect(s.x+n*4+.5,s.y-n*5+.5,s.w-1,s.h-1);ctx.globalAlpha=1; }
      ctx.fillStyle='#132d22';ctx.fillRect(s.x,s.y,s.w,s.h);
      ctx.strokeStyle=s.color;ctx.strokeRect(s.x+.5,s.y+.5,s.w-1,s.h-1);
      for(let p=0;p<10;p++) {
        const px=s.x+7+(p*17)%Math.max(16,s.w-13),py=s.y+8+(p*29)%Math.max(16,s.h-18);
        ctx.fillStyle=s.color;ctx.globalAlpha=.13+((Math.sin(time/500+p+i)+1)/2)*.38;ctx.fillRect(px,py,4,4);ctx.globalAlpha=1;
      }
      label(s.label,s.x+s.w/2,s.y+s.h+22,'#dce9dc',11,'center');
      label(s.channels,s.x+s.w/2,s.y+s.h+38,'#729282',9,'center');
    });
    label('ENCODER  ↓',350,357,'#70dbc9',11,'center'); label('ESPACIO LATENTE',561,357,'#d1f697',10,'center'); label('DECODER  ↑',754,357,'#b9ff47',11,'center');
    line(38,401,1062,401,'#254237');
    if(mode==='train') {
      label('ÉPOCA '+String(epoch).padStart(2,'0')+' / 50',38,436,'#e0e9dc',13);
      label('VALIDACIÓN MSE  '+validationLoss[epoch-1].toFixed(6),350,436,'#b9ff47',12);
      const start=630, end=1060, top=418, bottom=455;
      ctx.strokeStyle='#b9ff47';ctx.lineWidth=2;ctx.beginPath();
      validationLoss.slice(0,epoch).forEach((v,i)=>{const x=start+i/49*(end-start),y=bottom-(v-.0027)/(.008-.0027)*(bottom-top);if(!i)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();
      label('PÉRDIDA REGISTRADA',1060,474,'#668276',9,'right');
    } else {
      label('SIN RUIDO ARTIFICIAL',38,436,'#e0e9dc',12);
      label('ERROR ABSOLUTO  →  MÁSCARA  →  MEDIANA  →  SCORE',350,436,'#ff9c71',11);
      label('* MAPA DE ERROR ILUSTRATIVO',1060,474,'#668276',9,'right');
    }
  }
  function frame(time) {
    if(!playing) return;
    if(visible && mode==='train' && time-lastTick>450) {setEpoch(epoch===50?1:epoch+1);lastTick=time;}
    if(visible) draw(time);
    requestAnimationFrame(frame);
  }
  epochSlider.addEventListener('input',e=>{setPlaying(false);setEpoch(e.target.value);});
  playButton.addEventListener('click',()=>setPlaying(!playing));
  document.querySelectorAll('[data-viz-mode]').forEach(button=>button.addEventListener('click',()=>{
    mode=button.dataset.vizMode;
    document.querySelectorAll('[data-viz-mode]').forEach(b=>{b.classList.toggle('is-active',b===button);b.setAttribute('aria-pressed',String(b===button));});
    document.querySelector('#vizInputLabel').textContent=mode==='train'?'ENTRADA · NORMAL + RUIDO':'ENTRADA · SIN RUIDO';
    document.querySelector('#vizOutputLabel').textContent=mode==='train'?'OBJETIVO · CORTE LIMPIO':'SALIDA · MAPA DE ERROR';
    document.querySelector('#vizOutputValue').textContent=mode==='train'?'Pérdida MSE en primer plano':'Residuo → score → umbral';
    document.querySelector('.viz-timeline').hidden=mode==='infer';
    document.querySelector('#vizNote').textContent=mode==='train'?'La animación de capas y señales es ilustrativa. La pérdida por época procede de la ejecución registrada; las imágenes y activaciones no son una reproducción de cada paso del entrenamiento.':'La animación y el mapa de error son ilustrativos, no una predicción calculada aquí. Para ejecutar el checkpoint real, usa «Probar el modelo».';
    draw(performance.now());
  }));
  new IntersectionObserver(([entry])=>{visible=entry.isIntersecting;if(visible)draw(performance.now());}).observe(canvas);
  document.addEventListener('visibilitychange',()=>{visible=!document.hidden;});
  setEpoch(1);setPlaying(playing);
})();
