const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const progress = $('#progress');
const updateProgress = () => {
  const max = document.documentElement.scrollHeight - innerHeight;
  progress.style.width = `${max ? (scrollY / max) * 100 : 0}%`;
};
addEventListener('scroll', updateProgress, { passive: true });

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => entry.isIntersecting && entry.target.classList.add('visible'));
}, { threshold: .12 });
$$('.reveal').forEach(el => observer.observe(el));

const modes = {
  train: [
    ['Corte normal limpio', 'Se toma una imagen de train/good, sin ninguna anomalía.'],
    ['Añadir ruido gaussiano grueso', 'n ~ N(0, I) a 16×16; interpolación bilineal a 128×128, desplazamiento aleatorio y suma con σ = 0,2 solo donde x > 0,01.'],
    ['Recuperar lo limpio', 'La U-Net recibe la versión ruidosa e intenta producir la original.'],
    ['Minimizar la MSE', 'El error cuadrático actualiza 8,56 M de parámetros para restaurar mejor.']
  ],
  infer: [
    ['Corte desconocido', 'La imagen llega limpia y puede ser normal o anómala. No se usa su etiqueta.'],
    ['Sin ruido artificial', 'En inferencia no se corrompe la entrada: el modelo ve el corte tal cual.'],
    ['Reconstruir', 'La U-Net genera la versión más compatible con la normalidad que aprendió.'],
    ['Calcular el score', 'El error absoluto filtrado se reduce a un máximo y se compara con 0,18611.']
  ]
};
$$('.mode-switch button').forEach(button => button.addEventListener('click', () => {
  $$('.mode-switch button').forEach(b => b.classList.toggle('active', b === button));
  const mode = button.dataset.mode;
  modes[mode].forEach((step, i) => {
    $(`#step${i + 1}-title`).textContent = step[0];
    $(`#step${i + 1}-copy`).textContent = step[1];
  });
  $('#arrow1').textContent = mode === 'train' ? 'corromper' : 'entrada directa';
  $('#arrow3').textContent = mode === 'train' ? 'comparar' : 'puntuar';
  $('#pipeline').classList.toggle('inference', mode === 'infer');
}));

// Recorrido animado por las resoluciones espaciales de la U-Net.
const unetStep = $('#unetStep');
if (unetStep) {
  const nodes = [...document.querySelectorAll('[data-unet-step]')];
  const links = [...document.querySelectorAll('.unet .u-link')];
  const skips = [...document.querySelectorAll('.skip-lines path')];
  const playButton = $('#unetPlay');
  const stages = [
    ['Entrada · 128²','La red recibe el corte completo y extrae sus primeros patrones.'],
    ['Encoder · 64²','Reduce a la mitad el ancho y el alto; aumenta los canales para conservar más tipos de características.'],
    ['Encoder · 32²','La representación pierde detalle fino y gana contexto sobre regiones más amplias.'],
    ['Espacio latente · 16²','Es la resolución mínima: una representación compacta de la estructura cerebral.'],
    ['Decoder · 32²','La imagen empieza a expandirse y recibe detalle del encoder mediante la conexión de salto interna.'],
    ['Decoder · 64²','Se recupera más resolución y se combina con la información guardada en el encoder.'],
    ['Salida · 128²','La red vuelve al tamaño original y produce una intensidad reconstruida para cada píxel.']
  ];
  let current = 0, playing = !matchMedia('(prefers-reduced-motion: reduce)').matches, timer;

  const renderUnetStep = index => {
    current = index; unetStep.value = String(index);
    nodes.forEach((node,i) => {
      node.classList.toggle('active', i === index);
      node.classList.toggle('visited', i < index);
    });
    links.forEach((link,i) => link.classList.toggle('active', i < index));
    skips.forEach((path,i) => path.classList.toggle('active', index >= 4 && i === 6-index));
    $('#unetStepValue').textContent = `${String(index+1).padStart(2,'0')} / 07`;
    $('#unetStageTitle').textContent = stages[index][0]; $('#unetStageText').textContent = stages[index][1];
  };
  const restartTimer = () => {
    clearInterval(timer);
    if (playing) timer = setInterval(() => renderUnetStep((current + 1) % stages.length), 1250);
    playButton.innerHTML = playing ? '⏸ <span>Pausar</span>' : '▶ <span>Reproducir</span>';
    playButton.setAttribute('aria-label', playing ? 'Pausar animación de la U-Net' : 'Reproducir animación de la U-Net');
  };
  playButton.addEventListener('click', () => { playing = !playing; restartTimer(); });
  unetStep.addEventListener('input', () => { renderUnetStep(Number(unetStep.value)); playing = false; restartTimer(); });

  const unetImage = new Image();
  unetImage.addEventListener('load', () => {
    const visualSizes = [64,32,16,8,16,32,64];
    nodes.forEach((node,index) => {
      const canvas = node.querySelector('canvas'), target = canvas.getContext('2d');
      const reduced = document.createElement('canvas'), reducedSize = visualSizes[index];
      reduced.width = reducedSize; reduced.height = reducedSize;
      reduced.getContext('2d').drawImage(unetImage,0,0,reducedSize,reducedSize);
      target.imageSmoothingEnabled = false; target.drawImage(reduced,0,0,canvas.width,canvas.height);
    });
  });
  unetImage.src = 'assets/brain-anomaly-model-input.png';
  renderUnetStep(0); restartTimer();
}

// Demostración visual del scoring usando los ocho pares reales guardados.
const scoreCase = $('#scoreCase');
if (scoreCase) {
  const originalCanvas = $('#scoreOriginal'), reconstructionCanvas = $('#scoreReconstruction');
  const errorCanvas = $('#scoreError'), errorMiniCanvas = $('#scoreErrorMini');
  const maskedCanvas = $('#scoreMasked'), filteredCanvas = $('#scoreFiltered');
  const size = 128;
  const ctx = canvas => canvas.getContext('2d', { willReadFrequently: true });
  const originalCtx = ctx(originalCanvas), reconstructionCtx = ctx(reconstructionCanvas);
  const clampByte = value => Math.max(0, Math.min(255, Math.round(value)));
  const heatStops = [[0,15,28],[0,184,180],[247,225,74],[255,70,35]];

  const paintMap = (canvas, values) => {
    const target = canvas.getContext('2d').createImageData(size, size);
    for (let i = 0; i < values.length; i += 1) {
      const scale = Math.max(0, Math.min(1, values[i] / .45));
      const position = scale * (heatStops.length - 1), lower = Math.floor(position);
      const upper = Math.min(lower + 1, heatStops.length - 1), weight = position - lower;
      const base = i * 4;
      for (let channel = 0; channel < 3; channel += 1) target.data[base + channel] = clampByte(heatStops[lower][channel] * (1-weight) + heatStops[upper][channel] * weight);
      target.data[base + 3] = 255;
    }
    canvas.getContext('2d').putImageData(target, 0, 0);
  };

  const reflect = coordinate => coordinate < 0 ? -coordinate : coordinate >= size ? 2 * size - coordinate - 2 : coordinate;
  const renderScoreCase = source => {
    const index = Number(scoreCase.value);
    originalCtx.clearRect(0, 0, size, size); reconstructionCtx.clearRect(0, 0, size, size);
    originalCtx.drawImage(source, index * size, 20, size, size, 0, 0, size, size);
    reconstructionCtx.drawImage(source, index * size, 20 + size, size, size, 0, 0, size, size);
    const original = originalCtx.getImageData(0, 0, size, size).data;
    const reconstruction = reconstructionCtx.getImageData(0, 0, size, size).data;
    const raw = new Float32Array(size * size), masked = new Float32Array(size * size);
    for (let y = 0; y < size; y += 1) for (let x = 0; x < size; x += 1) {
      const pixel = y * size + x, offset = pixel * 4;
      raw[pixel] = Math.abs(original[offset] - reconstruction[offset]) / 255;
      let foreground = 0;
      for (let dy = -2; dy <= 2; dy += 1) for (let dx = -2; dx <= 2; dx += 1) {
        const nx = x + dx, ny = y + dy;
        if (nx >= 0 && nx < size && ny >= 0 && ny < size && original[(ny * size + nx) * 4] / 255 > .01) foreground += 1;
      }
      masked[pixel] = foreground / 25 > .95 ? raw[pixel] : 0;
    }
    const filtered = new Float32Array(size * size), windowValues = new Float32Array(25);
    let maximum = 0;
    for (let y = 0; y < size; y += 1) for (let x = 0; x < size; x += 1) {
      let k = 0;
      for (let dy = -2; dy <= 2; dy += 1) for (let dx = -2; dx <= 2; dx += 1) windowValues[k++] = masked[reflect(y+dy) * size + reflect(x+dx)];
      windowValues.sort();
      const value = windowValues[12], pixel = y * size + x;
      filtered[pixel] = value; maximum = Math.max(maximum, value);
    }
    paintMap(errorCanvas, raw); errorMiniCanvas.getContext('2d').drawImage(errorCanvas, 0, 0);
    paintMap(maskedCanvas, masked); paintMap(filteredCanvas, filtered);
    $('#scoreFinalValue').textContent = maximum.toFixed(3).replace('.', ',');
    $('#scoreCaseValue').textContent = String(index + 1);
  };

  const stageCopy = {
    error:['01','Comparar original y reconstrucción','Se calcula la diferencia absoluta de cada píxel. Las zonas más claras tienen mayor error de reconstrucción.','E = |x − fθ(x)|'],
    mask:['02','Quitar el fondo y los bordes','Se conserva un píxel solo si al menos 24 de los 25 píxeles de su vecindario 5×5 pertenecen al cerebro.','Eₘ = E ⊙ 𝟙[avg₅×₅(M) > 0,95]'],
    median:['03','Eliminar respuestas aisladas','Cada valor se sustituye por la mediana de su ventana 5×5. Un punto brillante aislado desaparece, pero una región consistente permanece.','A = mediana₅×₅(Eₘ)'],
    maximum:['04','Convertir el mapa en un solo número','Se toma el mayor valor del mapa filtrado. Ese número es el score de anomalía utilizado para clasificar todo el corte.','s(x) = max A(x)']
  };
  $$('[data-score-stage]').forEach(button => button.addEventListener('click', () => {
    $$('[data-score-stage]').forEach(item => item.classList.toggle('active', item === button));
    const [number,title,copy,formula] = stageCopy[button.dataset.scoreStage];
    $('#scoreExplainNumber').textContent = number; $('#scoreExplainTitle').textContent = title;
    $('#scoreExplainText').textContent = copy; $('#scoreExplainFormula').textContent = formula;
  }));

  const scoreSource = new Image();
  scoreSource.addEventListener('load', () => renderScoreCase(scoreSource));
  scoreSource.src = 'assets/reconstructions.png';
  scoreCase.addEventListener('input', () => {
    if (scoreSource.complete && scoreSource.naturalWidth) renderScoreCase(scoreSource);
  });
}

const normalBins = [0,1,36,124,177,129,89,39,19,14,6,2,0,1,0,3,0,0,0,0,0,0,0,0,0,0];
const anomalyBins = [0,0,3,28,30,73,70,100,118,145,161,199,195,251,238,272,287,293,201,172,121,61,33,19,5,0];
const ns = 'http://www.w3.org/2000/svg';
const svgEl = (name, attrs) => { const el = document.createElementNS(ns, name); Object.entries(attrs).forEach(([k,v]) => el.setAttribute(k,v)); return el; };
const hist = $('#histogram');
if (hist) {
const W = 700, H = 300, pad = { l: 42, r: 12, t: 25, b: 35 }, max = 300;
for (let i = 0; i <= 6; i++) {
  const x = pad.l + i / 6 * (W - pad.l - pad.r);
  hist.append(svgEl('line', { x1:x, y1:pad.t, x2:x, y2:H-pad.b, stroke:'#d8d9d2', 'stroke-width':1 }));
  const t = svgEl('text', { x, y:H-12, fill:'#728077', 'font-size':10, 'text-anchor':'middle', 'font-family':'DM Mono' });
  t.textContent = (i * .1).toFixed(1).replace('.', ','); hist.append(t);
}
const barW = (W-pad.l-pad.r)/normalBins.length;
normalBins.forEach((v,i) => {
  const h = v/max*(H-pad.t-pad.b);
  hist.append(svgEl('rect',{x:pad.l+i*barW+1,y:H-pad.b-h,width:barW-2,height:h,fill:'#74e1d0','fill-opacity':.72}));
});
anomalyBins.forEach((v,i) => {
  const h = v/max*(H-pad.t-pad.b);
  hist.append(svgEl('rect',{x:pad.l+i*barW+1,y:H-pad.b-h,width:barW-2,height:h,fill:'#ff6a3d','fill-opacity':.58}));
});
const thresholdLine = svgEl('line',{y1:pad.t,y2:H-pad.b,stroke:'#173f33','stroke-width':3,'stroke-dasharray':'5 4'});
hist.append(thresholdLine);
const thresholdLabel = svgEl('text',{y:16,fill:'#173f33','font-size':10,'text-anchor':'middle','font-family':'DM Mono'}); hist.append(thresholdLabel);
const slider = $('#threshold');
const renderThreshold = () => {
  const value = +slider.value;
  const x = pad.l + value/.65*(W-pad.l-pad.r);
  thresholdLine.setAttribute('x1',x); thresholdLine.setAttribute('x2',x);
  thresholdLabel.setAttribute('x',x); thresholdLabel.textContent = value.toFixed(3).replace('.',',');
  $('#thresholdValue').textContent = value.toFixed(3).replace('.',',');
  const note = $('#thresholdNote');
  if (Math.abs(value-.186)<.004) note.textContent = 'Umbral calibrado: equilibrio elegido antes de mirar el test.';
  else if (value < .186) note.textContent = 'Más sensible: detectará más anomalías, pero aumentarán las falsas alarmas.';
  else note.textContent = 'Más específico: reducirá falsas alarmas, pero escaparán más anomalías.';
};
slider.addEventListener('input',renderThreshold); renderThreshold();
}

// Laboratorio interactivo: ruido gaussiano grueso 16×16 ampliado a 128×128.
const noiseSlider = $('#noiseSigma');
const noiseOriginal = $('#noiseOriginal');
const noiseApplied = $('#noiseApplied');
const gaussianChart = $('#gaussianChart');

if (noiseSlider && noiseOriginal && noiseApplied && gaussianChart) {
  const size = 256, coarseSize = 16;
  const originalCtx = noiseOriginal.getContext('2d', { willReadFrequently: true });
  const appliedCtx = noiseApplied.getContext('2d');
  const graphCtx = gaussianChart.getContext('2d');
  const coarseCanvas = $('#noiseCoarse');
  const coarseCtx = coarseCanvas.getContext('2d');
  const upscaledCtx = $('#noiseUpscaled').getContext('2d');
  const shiftedCtx = $('#noiseShifted').getContext('2d');
  const maskCtx = $('#noiseMask').getContext('2d');
  const fieldCanvas = document.createElement('canvas');
  fieldCanvas.width = size; fieldCanvas.height = size;
  const fieldCtx = fieldCanvas.getContext('2d', { willReadFrequently: true });
  let field;
  let seed = 42042;
  const random = () => ((seed = (1664525 * seed + 1013904223) >>> 0) / 4294967296);

  const buildNoiseField = () => {
    const coarsePixels = coarseCtx.createImageData(coarseSize, coarseSize);
    for (let i = 0; i < coarseSize * coarseSize; i += 1) {
      const u = Math.max(random(), 1e-9), v = random();
      const value = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      const encoded = Math.round(Math.max(0, Math.min(255, 128 + value * 38)));
      coarsePixels.data.set([encoded, encoded, encoded, 255], i * 4);
    }
    coarseCtx.putImageData(coarsePixels, 0, 0);
    fieldCtx.clearRect(0, 0, size, size);
    fieldCtx.imageSmoothingEnabled = true;
    fieldCtx.drawImage(coarseCanvas, 0, 0, size, size);
    const unshifted = fieldCtx.getImageData(0, 0, size, size);
    upscaledCtx.putImageData(unshifted, 0, 0);

    const shiftXModel = Math.floor(random() * 128), shiftYModel = Math.floor(random() * 128);
    const shiftX = shiftXModel * 2, shiftY = shiftYModel * 2;
    const shifted = shiftedCtx.createImageData(size, size);
    for (let y = 0; y < size; y += 1) for (let x = 0; x < size; x += 1) {
      const sourceX = (x - shiftX + size) % size;
      const sourceY = (y - shiftY + size) % size;
      const sourceIndex = (sourceY * size + sourceX) * 4;
      const targetIndex = (y * size + x) * 4;
      shifted.data.set(unshifted.data.subarray(sourceIndex, sourceIndex + 4), targetIndex);
    }
    shiftedCtx.putImageData(shifted, 0, 0);
    field = shifted.data;
    $('#noiseShiftLabel').textContent = `roll(${shiftXModel}, ${shiftYModel})`;
  };

  const drawGaussian = sigma => {
    const ctx = graphCtx, w = gaussianChart.width, h = gaussianChart.height;
    const p = { l: 48, r: 20, t: 24, b: 42 };
    ctx.clearRect(0, 0, w, h); ctx.fillStyle = '#091712'; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = '#294138'; ctx.lineWidth = 1; ctx.fillStyle = '#71887c'; ctx.font = '11px "DM Mono"';
    for (let i = -2; i <= 2; i += 1) {
      const x = p.l + (i + 2) / 4 * (w - p.l - p.r);
      ctx.beginPath(); ctx.moveTo(x, p.t); ctx.lineTo(x, h - p.b); ctx.stroke();
      const axisValue = i * .4;
      ctx.textAlign = 'center'; ctx.fillText(i === 0 ? 'μ = 0' : `${axisValue > 0 ? '+' : ''}${axisValue.toFixed(1).replace('.', ',')}`, x, h - 16);
    }
    const visualSigma = Math.max(sigma, .012), peak = 1 / (visualSigma * Math.sqrt(2 * Math.PI));
    if (sigma > 0) {
      ctx.strokeStyle = '#ff9c71'; ctx.setLineDash([5, 5]);
      [-sigma, sigma].forEach(value => {
        const x = p.l + (value + .8) / 1.6 * (w - p.l - p.r);
        ctx.beginPath(); ctx.moveTo(x, p.t); ctx.lineTo(x, h - p.b); ctx.stroke();
      });
      ctx.setLineDash([]);
    }
    ctx.beginPath();
    for (let i = 0; i <= 240; i += 1) {
      const value = -.8 + i / 240 * 1.6;
      const density = Math.exp(-.5 * (value / visualSigma) ** 2) / (visualSigma * Math.sqrt(2 * Math.PI));
      const x = p.l + i / 240 * (w - p.l - p.r);
      const y = h - p.b - density / peak * (h - p.t - p.b - 12);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.lineTo(w - p.r, h - p.b); ctx.lineTo(p.l, h - p.b); ctx.closePath();
    const gradient = ctx.createLinearGradient(0, p.t, 0, h - p.b);
    gradient.addColorStop(0, 'rgba(116,225,208,.65)'); gradient.addColorStop(1, 'rgba(116,225,208,.03)');
    ctx.fillStyle = gradient; ctx.fill(); ctx.strokeStyle = '#b9ff47'; ctx.lineWidth = 3; ctx.stroke();
  };

  const renderNoise = () => {
    if (!field) return;
    const sigma = Number(noiseSlider.value), label = sigma.toFixed(2).replace('.', ',');
    $('#noiseSigmaValue').textContent = label; $('#noiseSigmaLabel').textContent = `σ = ${label}`;
    const clean = originalCtx.getImageData(0, 0, size, size);
    const noisy = appliedCtx.createImageData(size, size);
    const mask = maskCtx.createImageData(size, size);
    for (let i = 0; i < clean.data.length; i += 4) {
      const value = clean.data[i] / 255;
      const n = (field[i] - 128) / 38;
      const result = value > .01 ? Math.max(0, Math.min(1, value + sigma * n)) : 0;
      const gray = Math.round(result * 255);
      noisy.data.set([gray, gray, gray, 255], i);
      const maskValue = value > .01 ? 255 : 0;
      mask.data.set([maskValue, maskValue, maskValue, 255], i);
    }
    appliedCtx.putImageData(noisy, 0, 0); maskCtx.putImageData(mask, 0, 0); drawGaussian(sigma);
  };

  const source = new Image();
  source.addEventListener('load', () => {
    originalCtx.fillStyle = '#000'; originalCtx.fillRect(0, 0, size, size);
    originalCtx.drawImage(source, 0, 0, size, size);
    buildNoiseField();
    renderNoise();
  });
  source.src = 'assets/brain-normal-model-input.png';
  noiseSlider.addEventListener('input', renderNoise);
  $('#regenerateNoise').addEventListener('click', () => { buildNoiseField(); renderNoise(); });
}

const losses = [.007729,.006426,.005913,.005353,.005324,.004896,.004710,.004502,.004253,.003920,.004214,.004349,.004059,.004356,.004820,.003983,.003790,.003920,.003649,.003772,.004252,.004047,.003840,.003751,.003338,.003994,.003477,.003746,.003561,.003766,.003562,.003640,.003257,.003682,.003604,.003506,.003706,.003186,.003356,.003336,.003669,.003264,.003416,.003784,.003274,.003514,.002984,.003307,.003445,.003286];
const chart = $('#lossChart'), cw=320, ch=100, cpad=8, lo=.0027, hi=.008;
const points = losses.map((v,i)=>`${cpad+i/(losses.length-1)*(cw-2*cpad)},${ch-cpad-(v-lo)/(hi-lo)*(ch-2*cpad)}`).join(' ');
chart.append(svgEl('polyline',{points,fill:'none',stroke:'#b9ff47','stroke-width':2}));
chart.append(svgEl('line',{x1:cpad+46/49*(cw-2*cpad),x2:cpad+46/49*(cw-2*cpad),y1:5,y2:95,stroke:'#74e1d0','stroke-dasharray':'3 3'}));
const best = svgEl('circle',{cx:cpad+46/49*(cw-2*cpad),cy:ch-cpad-(losses[46]-lo)/(hi-lo)*(ch-2*cpad),r:4,fill:'#74e1d0'}); chart.append(best);

// Selección dinámica del umbral con Youden sobre los 83 scores de validación.
const youdenChart = $('#youdenChart');
const youdenSlider = $('#youdenThreshold');
if (youdenChart && youdenSlider) {
  const validationScores = [[0,.12242],[0,.130399],[0,.101904],[0,.126672],[0,.16302],[0,.128576],[0,.283775],[0,.141106],[0,.148212],[0,.131453],[0,.136569],[0,.112285],[0,.14278],[0,.170621],[0,.087533],[0,.094425],[0,.107925],[0,.099033],[0,.09132],[0,.135504],[0,.145491],[0,.184166],[0,.070108],[0,.252228],[0,.262578],[0,.098919],[0,.119015],[0,.125419],[0,.129915],[0,.134074],[0,.078042],[0,.172827],[0,.129461],[0,.159734],[0,.125875],[0,.081302],[0,.134471],[0,.091539],[0,.151494],[1,.487625],[1,.549221],[1,.583415],[1,.565049],[1,.331747],[1,.410193],[1,.471011],[1,.513444],[1,.442343],[1,.485035],[1,.428756],[1,.346943],[1,.485139],[1,.485066],[1,.468468],[1,.202436],[1,.299213],[1,.370044],[1,.348792],[1,.3154],[1,.498584],[1,.381535],[1,.450253],[1,.398618],[1,.249619],[1,.127565],[1,.133964],[1,.153493],[1,.517145],[1,.512801],[1,.369689],[1,.564729],[1,.186114],[1,.383475],[1,.456796],[1,.416729],[1,.273679],[1,.437658],[1,.439828],[1,.42065],[1,.133754],[1,.159414],[1,.178581],[1,.152498]];
  const positives = validationScores.filter(([label]) => label === 1).length;
  const negatives = validationScores.length - positives;
  const scoreValues = [...new Set(validationScores.map(([,score]) => score))].sort((a,b) => b-a);
  const thresholds = [scoreValues[0]+1e-6,...scoreValues,scoreValues.at(-1)-1e-6];
  const points = thresholds.map(threshold => {
    const tp = validationScores.filter(([label,score]) => label === 1 && score >= threshold).length;
    const fp = validationScores.filter(([label,score]) => label === 0 && score >= threshold).length;
    const tpr = tp/positives, fpr = fp/negatives;
    return {threshold,tpr,fpr,j:tpr-fpr};
  });
  const bestIndex = points.reduce((winner,point,index) => point.j > points[winner].j ? index : winner,0);
  const yw=460,yh=360,yp={l:58,r:20,t:22,b:52};
  const yx=value => yp.l+value*(yw-yp.l-yp.r), yy=value => yh-yp.b-value*(yh-yp.t-yp.b);
  youdenSlider.max=String(points.length-1); youdenSlider.value=String(bestIndex);
  for(let i=0;i<=4;i+=1){
    const value=i/4,x=yx(value),y=yy(value);
    youdenChart.append(svgEl('line',{x1:x,y1:yp.t,x2:x,y2:yh-yp.b,stroke:'#263b32'}));
    youdenChart.append(svgEl('line',{x1:yp.l,y1:y,x2:yw-yp.r,y2:y,stroke:'#263b32'}));
    const tx=svgEl('text',{x,y:yh-25,fill:'#71847a','font-size':10,'text-anchor':'middle','font-family':'DM Mono'});tx.textContent=value.toFixed(2).replace('.',',');youdenChart.append(tx);
    const ty=svgEl('text',{x:43,y:y+4,fill:'#71847a','font-size':10,'text-anchor':'end','font-family':'DM Mono'});ty.textContent=value.toFixed(2).replace('.',',');youdenChart.append(ty);
  }
  youdenChart.append(svgEl('line',{x1:yx(0),y1:yy(0),x2:yx(1),y2:yy(1),stroke:'#65766e','stroke-dasharray':'6 6'}));
  const curve=points.map(point=>`${yx(point.fpr)},${yy(point.tpr)}`).join(' ');
  youdenChart.append(svgEl('polyline',{points:curve,fill:'none',stroke:'#74e1d0','stroke-width':4,'stroke-linejoin':'round'}));
  const bestPoint=points[bestIndex];
  youdenChart.append(svgEl('circle',{cx:yx(bestPoint.fpr),cy:yy(bestPoint.tpr),r:11,fill:'none',stroke:'#b9ff47','stroke-width':3}));
  const distanceLine=svgEl('line',{stroke:'#ff6a3d','stroke-width':5,'stroke-linecap':'round'});youdenChart.append(distanceLine);
  const currentPoint=svgEl('circle',{r:7,fill:'#ff6a3d',stroke:'#fff','stroke-width':2});youdenChart.append(currentPoint);
  const xTitle=svgEl('text',{x:(yp.l+yw-yp.r)/2,y:yh-4,fill:'#91a39a','font-size':11,'text-anchor':'middle','font-family':'DM Mono'});xTitle.textContent='TASA DE FALSOS POSITIVOS · FPR';youdenChart.append(xTitle);
  const yTitle=svgEl('text',{x:13,y:(yp.t+yh-yp.b)/2,fill:'#91a39a','font-size':11,'text-anchor':'middle','font-family':'DM Mono',transform:`rotate(-90 13 ${(yp.t+yh-yp.b)/2})`});yTitle.textContent='SENSIBILIDAD · TPR';youdenChart.append(yTitle);

  const renderYouden=()=>{
    const index=Number(youdenSlider.value),point=points[index];
    currentPoint.setAttribute('cx',yx(point.fpr));currentPoint.setAttribute('cy',yy(point.tpr));
    distanceLine.setAttribute('x1',yx(point.fpr));distanceLine.setAttribute('x2',yx(point.fpr));
    distanceLine.setAttribute('y1',yy(point.fpr));distanceLine.setAttribute('y2',yy(point.tpr));
    $('#youdenThresholdValue').textContent=point.threshold.toFixed(5).replace('.',',');
    $('#youdenTpr').textContent=`${(point.tpr*100).toFixed(2).replace('.',',')}%`;
    $('#youdenFpr').textContent=`${(point.fpr*100).toFixed(2).replace('.',',')}%`;
    $('#youdenJ').textContent=point.j.toFixed(4).replace('.',',');
    $('#youdenTprBar').style.width=`${point.tpr*100}%`;$('#youdenFprBar').style.width=`${point.fpr*100}%`;
    const status=$('#youdenStatus');
    status.textContent=index===bestIndex?'Máximo encontrado':`${((bestPoint.j-point.j)*100).toFixed(2).replace('.',',')} puntos por debajo del máximo`;
    status.style.color=index===bestIndex?'#b9ff47':'#ff9c71';
  };
  youdenSlider.addEventListener('input',renderYouden);
  $('#youdenBest').addEventListener('click',()=>{youdenSlider.value=String(bestIndex);renderYouden();});
  renderYouden();
}

// Curva ROC del test y traducción dinámica a sensibilidad, especificidad y BA.
const rocChart = $('#rocChart');
const rocSlider = $('#rocPoint');
if (rocChart && rocSlider) {
  const rocPoints = [[0,0],[0,.0312],[0,.0621],[0,.0930],[0,.1242],[0,.1551],[0,.1860],[0,.2169],[0,.2481],[0,.2790],[0,.3099],[0,.3408],[0,.3720],[0,.4029],[0,.4338],[.0031,.4641],[.0047,.4950],[.0047,.5259],[.0047,.5567],[.0047,.5876],[.0063,.6185],[.0063,.6494],[.0063,.6803],[.0063,.7112],[.0078,.7421],[.0094,.7727],[.0141,.8026],[.0234,.8315],[.0391,.8595],[.0531,.8875],[.0813,.9125],[.1016,.9210],[.1297,.9333],[.2047,.9489],[.3031,.9593],[.3969,.9707],[.4938,.9815],[.6234,.9857],[.7516,.9899],[.8703,.9961],[1,1]];
  const calibratedIndex = 31;
  const rw = 460, rh = 360, rp = { l:58, r:20, t:22, b:52 };
  const sx = value => rp.l + value * (rw - rp.l - rp.r);
  const sy = value => rh - rp.b - value * (rh - rp.t - rp.b);
  rocSlider.max = String(rocPoints.length - 1); rocSlider.value = String(calibratedIndex);

  for (let i = 0; i <= 4; i += 1) {
    const value = i / 4, x = sx(value), y = sy(value);
    rocChart.append(svgEl('line',{x1:x,y1:rp.t,x2:x,y2:rh-rp.b,stroke:'#263b32','stroke-width':1}));
    rocChart.append(svgEl('line',{x1:rp.l,y1:y,x2:rw-rp.r,y2:y,stroke:'#263b32','stroke-width':1}));
    const tx = svgEl('text',{x,y:rh-25,fill:'#71847a','font-size':10,'text-anchor':'middle','font-family':'DM Mono'}); tx.textContent=value.toFixed(2).replace('.',','); rocChart.append(tx);
    const ty = svgEl('text',{x:43,y:y+4,fill:'#71847a','font-size':10,'text-anchor':'end','font-family':'DM Mono'}); ty.textContent=value.toFixed(2).replace('.',','); rocChart.append(ty);
  }
  rocChart.append(svgEl('line',{x1:sx(0),y1:sy(0),x2:sx(1),y2:sy(1),stroke:'#65766e','stroke-width':1.5,'stroke-dasharray':'6 6'}));
  const curve = rocPoints.map(([fpr,tpr]) => `${sx(fpr)},${sy(tpr)}`).join(' ');
  rocChart.append(svgEl('polygon',{points:`${sx(0)},${sy(0)} ${curve} ${sx(1)},${sy(0)}`,fill:'#74e1d0','fill-opacity':.12}));
  rocChart.append(svgEl('polyline',{points:curve,fill:'none',stroke:'#74e1d0','stroke-width':4,'stroke-linejoin':'round'}));
  const rocPoint = svgEl('circle',{r:7,fill:'#ff6a3d',stroke:'#fff','stroke-width':2}); rocChart.append(rocPoint);
  const xLabel = svgEl('text',{x:(rp.l+rw-rp.r)/2,y:rh-4,fill:'#91a39a','font-size':11,'text-anchor':'middle','font-family':'DM Mono'}); xLabel.textContent='TASA DE FALSOS POSITIVOS · 1 − especificidad'; rocChart.append(xLabel);
  const yLabel = svgEl('text',{x:13,y:(rp.t+rh-rp.b)/2,fill:'#91a39a','font-size':11,'text-anchor':'middle','font-family':'DM Mono',transform:`rotate(-90 13 ${(rp.t+rh-rp.b)/2})`}); yLabel.textContent='SENSIBILIDAD'; rocChart.append(yLabel);

  const renderRocPoint = () => {
    const index = Number(rocSlider.value), [fpr,tpr] = rocPoints[index];
    const specificity = 1 - fpr, balanced = (tpr + specificity) / 2;
    rocPoint.setAttribute('cx', sx(fpr)); rocPoint.setAttribute('cy', sy(tpr));
    $('#sensitivityValue').textContent = `${(tpr*100).toFixed(2).replace('.',',')}%`;
    $('#specificityValue').textContent = `${(specificity*100).toFixed(2).replace('.',',')}%`;
    $('#baValue').textContent = `${(balanced*100).toFixed(2).replace('.',',')}%`;
    $('#baEquationValue').textContent = `= ${(balanced*100).toFixed(2).replace('.',',')}%`;
    $('#sensitivityBar').style.width = `${tpr*100}%`; $('#specificityBar').style.width = `${specificity*100}%`;
    $('#rocMode').textContent = index === calibratedIndex ? 'Umbral calibrado · 0,18611' : index < calibratedIndex ? 'Umbral más estricto' : 'Umbral más permisivo';
  };
  rocSlider.addEventListener('input', renderRocPoint); renderRocPoint();
}

const sections = $$('main section[id]');
const navLinks = $$('.nav-links a');
const navObserver = new IntersectionObserver(entries => entries.forEach(entry => {
  if (entry.isIntersecting) navLinks.forEach(a => a.style.color = a.getAttribute('href') === `#${entry.target.id}` ? '#b9ff47' : '');
}), { rootMargin:'-35% 0px -60% 0px' });
sections.forEach(s => navObserver.observe(s));
