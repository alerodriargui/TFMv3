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

const normalBins = [0,1,36,124,177,129,89,39,19,14,6,2,0,1,0,3,0,0,0,0,0,0,0,0,0,0];
const anomalyBins = [0,0,3,28,30,73,70,100,118,145,161,199,195,251,238,272,287,293,201,172,121,61,33,19,5,0];
const hist = $('#histogram');
const W = 700, H = 300, pad = { l: 42, r: 12, t: 25, b: 35 }, max = 300;
const ns = 'http://www.w3.org/2000/svg';
const svgEl = (name, attrs) => { const el = document.createElementNS(ns, name); Object.entries(attrs).forEach(([k,v]) => el.setAttribute(k,v)); return el; };
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

const losses = [.007729,.006426,.005913,.005353,.005324,.004896,.004710,.004502,.004253,.003920,.004214,.004349,.004059,.004356,.004820,.003983,.003790,.003920,.003649,.003772,.004252,.004047,.003840,.003751,.003338,.003994,.003477,.003746,.003561,.003766,.003562,.003640,.003257,.003682,.003604,.003506,.003706,.003186,.003356,.003336,.003669,.003264,.003416,.003784,.003274,.003514,.002984,.003307,.003445,.003286];
const chart = $('#lossChart'), cw=320, ch=100, cpad=8, lo=.0027, hi=.008;
const points = losses.map((v,i)=>`${cpad+i/(losses.length-1)*(cw-2*cpad)},${ch-cpad-(v-lo)/(hi-lo)*(ch-2*cpad)}`).join(' ');
chart.append(svgEl('polyline',{points,fill:'none',stroke:'#b9ff47','stroke-width':2}));
chart.append(svgEl('line',{x1:cpad+46/49*(cw-2*cpad),x2:cpad+46/49*(cw-2*cpad),y1:5,y2:95,stroke:'#74e1d0','stroke-dasharray':'3 3'}));
const best = svgEl('circle',{cx:cpad+46/49*(cw-2*cpad),cy:ch-cpad-(losses[46]-lo)/(hi-lo)*(ch-2*cpad),r:4,fill:'#74e1d0'}); chart.append(best);

const sections = $$('main section[id]');
const navLinks = $$('.nav-links a');
const navObserver = new IntersectionObserver(entries => entries.forEach(entry => {
  if (entry.isIntersecting) navLinks.forEach(a => a.style.color = a.getAttribute('href') === `#${entry.target.id}` ? '#b9ff47' : '');
}), { rootMargin:'-35% 0px -60% 0px' });
sections.forEach(s => navObserver.observe(s));
