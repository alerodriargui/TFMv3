const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];

const state = { blob: null, name: 'Corte 01', objectUrl: null, caseId: null, version: 0 };
const inputImage = $('#inputImage');
const compareBefore = $('#compareBefore');
const runButton = $('#runButton');
const errorBanner = $('#errorBanner');

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.classList.add('visible');
}

function clearResult() {
  $('#localizationImages').hidden = true;
  $('#localizationMetrics').hidden = true;
  $('#localizationStatus').textContent = 'Ejecuta un caso del dataset para comparar con su referencia.';
  errorBanner.classList.remove('visible');
  $('.result-stage').classList.remove('ready');
  $('.heatmap-wrap').classList.remove('ready');
  $('#scoreValue').textContent = '—';
  $('#gaugeFill').style.width = '0';
  $('#verdict').className = 'verdict idle';
  $('#verdict').innerHTML = '<i></i><div><b>Sin analizar</b><span>Ejecuta el modelo para obtener una lectura.</span></div>';
  clearImagePipeline();
}

function selectSource(src, name, blob = null) {
  state.version += 1;
  state.caseId = null;
  $('#referenceCase').value = '';
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = blob ? URL.createObjectURL(blob) : null;
  state.blob = blob;
  state.name = name;
  const displaySrc = state.objectUrl || src;
  inputImage.src = displaySrc;
  compareBefore.src = displaySrc;
  $('#caseName').textContent = name;
  clearResult();
}

$$('.sample').forEach(button => button.addEventListener('click', () => {
  $$('.sample').forEach(item => item.classList.toggle('active', item === button));
  selectSource(button.dataset.src, button.dataset.name);
  if (button.dataset.caseId !== undefined) {
    state.caseId = Number(button.dataset.caseId);
    $('#referenceCase').value = button.dataset.caseId;
  }
}));

function acceptFile(file) {
  if (!file || !file.type.startsWith('image/')) return showError('Selecciona una imagen PNG, JPG o WEBP.');
  if (file.size > 10 * 1024 * 1024) return showError('La imagen supera el límite de 10 MB.');
  $$('.sample').forEach(item => item.classList.remove('active'));
  selectSource('', file.name, file);
}

$('#fileInput').addEventListener('change', event => acceptFile(event.target.files[0]));
const dropzone = $('#dropzone');
['dragenter', 'dragover'].forEach(type => dropzone.addEventListener(type, event => {
  event.preventDefault();
  dropzone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach(type => dropzone.addEventListener(type, event => {
  event.preventDefault();
  dropzone.classList.remove('dragging');
}));
dropzone.addEventListener('drop', event => acceptFile(event.dataTransfer.files[0]));

async function sourceBlob() {
  if (state.blob) return state.blob;
  const response = await fetch(inputImage.src);
  if (!response.ok) throw new Error('No se pudo leer el ejemplo seleccionado.');
  return response.blob();
}

async function runInference() {
  const version = state.version;
  clearResult();
  runButton.classList.add('loading');
  runButton.querySelector('span').textContent = 'Reconstruyendo';
  $('.result-stage').classList.add('processing');
  try {
    const blob = state.caseId === null ? await sourceBlob() : new Blob([]);
    const response = await fetch(state.caseId === null ? '/api/reconstruct' : `/api/reconstruct?case=${state.caseId}`, {
      method: 'POST',
      headers: { 'Content-Type': blob.type || 'application/octet-stream' },
      body: blob
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || 'El servidor no pudo ejecutar la inferencia.');
    if (version !== state.version) return;

    const inputSrc = `data:image/png;base64,${payload.input}`;
    const reconSrc = `data:image/png;base64,${payload.reconstruction}`;
    inputImage.src = inputSrc;
    compareBefore.src = inputSrc;
    $('#reconImage').src = reconSrc;
    $('#compareRecon').src = reconSrc;
    $('#heatmapImage').src = `data:image/png;base64,${payload.heatmap}`;
    $('.result-stage').classList.add('ready');
    $('.heatmap-wrap').classList.add('ready');

    $('#scoreValue').textContent = payload.score.toFixed(3).replace('.', ',');
    $('#gaugeFill').style.width = `${Math.min(payload.score / 0.6 * 100, 100)}%`;
    const verdict = $('#verdict');
    verdict.className = `verdict ${payload.is_anomaly ? 'anomaly' : 'normal'}`;
    verdict.innerHTML = payload.is_anomaly
      ? '<i></i><div><b>ANÓMALA</b><span>El residuo supera el umbral calibrado.</span></div>'
      : '<i></i><div><b>NORMAL</b><span>El residuo queda por debajo del umbral.</span></div>';
    loadImagePipeline(payload);
    if (payload.localization) {
      const metrics = payload.localization;
      $('#localizationImages').hidden = false;
      $('#referenceMask').src = `data:image/png;base64,${payload.reference_overlay}`;
      $('#predictionMask').src = `data:image/png;base64,${payload.prediction_overlay}`;
      $('#overlapImage').src = `data:image/png;base64,${payload.overlap}`;
      const format = value => value === null ? 'no definida' : value.toFixed(3);
      $('#localizationMetrics').hidden = false;
      $('#diceValue').textContent = format(metrics.dice);
      $('#iouValue').textContent = format(metrics.iou);
      $('#sensitivityValue').textContent = format(metrics.sensitivity);
      $('#localizationStatus').textContent = (payload.reference_pixels === 0 ? 'Referencia sin lesión a esta resolución. ' : 'Verde: referencia; naranja: red; cian: coincidencia. ') + `Umbral por píxel: ${payload.pixel_threshold.toFixed(5)}`;
    } else {
      $('#localizationStatus').textContent = state.caseId === null
        ? 'Esta imagen no tiene una máscara de referencia asociada. Selecciona un caso del dataset.'
        : 'Falta calibración: ejecuta python -m tfm_ae.localization y reinicia el servidor.';
    }
  } catch (error) {
    const hint = location.protocol === 'file:' ? ' Inicia la demo con “python -m tfm_ae.demo_server”.' : '';
    showError(`${error.message}${hint}`);
  } finally {
    runButton.classList.remove('loading');
    runButton.querySelector('span').textContent = 'Ejecutar modelo';
    $('.result-stage').classList.remove('processing');
  }
}

runButton.addEventListener('click', runInference);

$$('.view-tabs button').forEach(button => button.addEventListener('click', () => {
  $$('.view-tabs button').forEach(item => item.classList.toggle('active', item === button));
  $('.heatmap-wrap').classList.toggle('compare-mode', button.dataset.view === 'compare');
}));

$('#compareSlider').addEventListener('input', event => {
  $('#compareAfter').style.right = `${100 - event.target.value}%`;
});

// Laboratorio interactivo de las dos operaciones 5×5 del scoring.
const maskGrid = $('#maskGrid');
const medianGrid = $('#medianGrid');

if (maskGrid && medianGrid) {
  let maskValues = Array(25).fill(true);
  const maskButtons = maskValues.map((_, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.setAttribute('aria-label', `Vecino ${index + 1}`);
    if (index === 12) button.classList.add('center');
    button.addEventListener('click', () => {
      maskValues[index] = !maskValues[index];
      $$('[data-mask-preset]').forEach(item => item.classList.remove('active'));
      renderMask();
    });
    maskGrid.append(button);
    return button;
  });

  const renderMask = () => {
    const count = maskValues.filter(Boolean).length;
    maskButtons.forEach((button, index) => {
      button.classList.toggle('on', maskValues[index]);
      button.setAttribute('aria-pressed', String(maskValues[index]));
      button.title = maskValues[index] ? 'Cerebro' : 'Fondo';
    });
    const fraction = count / 25, keep = fraction > .95;
    $('#maskCount').textContent = count;
    $('#maskFraction').textContent = `${count} ÷ 25 = ${fraction.toFixed(2).replace('.', ',')}`;
    const verdict = $('#maskVerdict');
    verdict.className = `window-verdict ${keep ? 'keep' : 'remove'}`;
    verdict.innerHTML = keep
      ? '<b>✓ CONSERVAR</b><span>El píxel central está en el interior cerebral; se conserva su error.</span>'
      : '<b>× DESCARTAR</b><span>El píxel central está demasiado cerca del fondo o del borde; su error se pone a cero.</span>';
  };

  const maskPresets = { inside:25, limit:24, edge:18 };
  $$('[data-mask-preset]').forEach(button => button.addEventListener('click', () => {
    const count = maskPresets[button.dataset.maskPreset];
    maskValues = Array.from({ length:25 }, (_,index) => index < count);
    if (count < 25) {
      const shift = button.dataset.maskPreset === 'edge' ? 0 : 24;
      maskValues = Array(25).fill(true);
      if (button.dataset.maskPreset === 'limit') maskValues[shift] = false;
      else [0,1,2,5,6,10,11].forEach(index => { maskValues[index] = false; });
    }
    $$('[data-mask-preset]').forEach(item => item.classList.toggle('active', item === button));
    renderMask();
  }));

  let medianValues = Array(25).fill(.03);
  const medianButtons = medianValues.map((_, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.setAttribute('aria-label', `Error del vecino ${index + 1}`);
    if (index === 12) button.classList.add('center');
    button.addEventListener('click', () => {
      medianValues[index] = medianValues[index] < .1 ? .45 : medianValues[index] < .6 ? .90 : .03;
      $$('[data-median-preset]').forEach(item => item.classList.remove('active'));
      renderMedian();
    });
    medianGrid.append(button);
    return button;
  });

  const renderMedian = () => {
    medianButtons.forEach((button, index) => {
      const value = medianValues[index];
      button.textContent = value.toFixed(2).replace('.', ',');
      button.classList.toggle('low', value < .1);
      button.classList.toggle('mid', value >= .1 && value < .6);
      button.classList.toggle('high', value >= .6);
    });
    const ordered = [...medianValues].sort((a,b) => a-b);
    const median = ordered[12], before = medianValues[12];
    const orderedContainer = $('#medianOrdered');
    orderedContainer.replaceChildren(...ordered.map((value,index) => {
      const item = document.createElement('i');
      item.textContent = value.toFixed(2).replace('.', ',');
      if (index === 12) item.classList.add('selected');
      item.title = `${index + 1}.º valor${index === 12 ? ' · mediana seleccionada' : ''}`;
      return item;
    }));
    $('#medianBefore').textContent = before.toFixed(2).replace('.', ',');
    const conserved = Math.abs(median - before) < 1e-9, verdict = $('#medianVerdict');
    verdict.className = `window-verdict ${conserved ? 'keep' : 'remove'}`;
    verdict.innerHTML = conserved
      ? `<b>${before.toFixed(2).replace('.', ',')} → ${median.toFixed(2).replace('.', ',')} · CONSERVADO</b><span>El valor central permanece alto porque la región es consistente.</span>`
      : `<b>${before.toFixed(2).replace('.', ',')} → ${median.toFixed(2).replace('.', ',')} · SUSTITUIDO</b><span>El valor central se reemplaza por la mediana de sus vecinos.</span>`;
  };

  const setMedianPreset = preset => {
    medianValues = Array(25).fill(.03);
    if (preset === 'isolated') medianValues[12] = .90;
    if (preset === 'region') [2,6,7,8,10,11,12,13,14,16,17,18,22].forEach(index => { medianValues[index] = .78; });
    if (preset === 'noise') [0,4,7,12,16,20,23].forEach(index => { medianValues[index] = .90; });
    renderMedian();
  };
  $$('[data-median-preset]').forEach(button => button.addEventListener('click', () => {
    $$('[data-median-preset]').forEach(item => item.classList.toggle('active', item === button));
    setMedianPreset(button.dataset.medianPreset);
  }));

  maskValues[24] = false;
  setMedianPreset('isolated');
  renderMask();
}

// Recorrido por las operaciones reales aplicadas a la imagen inferida.
let processPayload = null;
let processStage = 0;
let processPlaying = false;
let processTimer = null;

const processStages = [
  ['01 · COMPARAR','Original frente a reconstrucción','El modelo produce una reconstrucción del mismo tamaño. La miniatura muestra la salida que se compara píxel a píxel con la entrada.','x frente a fθ(x)','input','reconstruction'],
  ['02 · ERROR BRUTO','Calcular la diferencia absoluta','Cada píxel contiene cuánto difiere la reconstrucción de la entrada. Todavía aparecen el fondo y los bordes.','E = |x − fθ(x)|','raw_heatmap',null],
  ['03 · MÁSCARA 5×5','Localizar el interior cerebral','Una celda se conserva cuando al menos 24 de sus 25 vecinos pertenecen al cerebro. Blanco significa interior válido.','Mᵢⱼ = 𝟙[avg₅×₅(x > 0,01) > 0,95]','mask','input'],
  ['04 · ENMASCARAR','Eliminar fondo y bordes','El error bruto se multiplica por la máscara. Todo lo que queda fuera del interior cerebral pasa a cero.','Eₘ = E ⊙ M','masked_heatmap',null],
  ['05 · MEDIANA 5×5','Suprimir errores aislados','En cada posición se ordenan los 25 errores vecinos y se usa el 13.º. Los picos aislados desaparecen; las regiones consistentes permanecen.','Aᵢⱼ = mediana₅×₅(Eₘ)','heatmap',null],
  ['06 · MÁXIMO Y UMBRAL','Convertir el mapa en una decisión','La cruz señala el mayor error restante. Ese único valor se compara con el umbral calibrado para clasificar la imagen completa.','s(x) = max A(x)  ·  s(x) ≥ 0,18611','heatmap',null]
];

function processDataUrl(key) {
  return `data:image/png;base64,${processPayload[key]}`;
}

function renderImagePipeline(stage) {
  if (!processPayload) return;
  processStage = stage;
  const [number,title,text,formula,imageKey,insetKey] = processStages[stage];
  const imageStage = $('.process-image-stage');
  $('#processImage').src = processDataUrl(imageKey);
  imageStage.classList.toggle('has-inset', Boolean(insetKey));
  imageStage.classList.toggle('show-maximum', stage === 5);
  if (insetKey) {
    $('#processInset').src = processDataUrl(insetKey);
    $('#processInsetLabel').textContent = insetKey === 'reconstruction' ? 'RECONSTRUCCIÓN' : 'ENTRADA';
  }
  $('#processNumber').textContent = number;
  $('#processTitle').textContent = title;
  $('#processText').textContent = text;
  $('#processFormula').textContent = formula;
  $('#processRange').value = String(stage);
  $$('[data-process-stage]').forEach((button,index) => button.classList.toggle('active', index === stage));
}

function setProcessPlaying(value) {
  processPlaying = value;
  clearInterval(processTimer);
  if (processPlaying) processTimer = setInterval(() => renderImagePipeline((processStage + 1) % processStages.length), 1500);
  const button = $('#processPlay');
  if (!button) return;
  button.innerHTML = processPlaying ? '⏸ <span>Pausar</span>' : '▶ <span>Reproducir</span>';
}

function loadImagePipeline(payload) {
  if (!payload.raw_heatmap || !$('#processShell')) return;
  processPayload = payload;
  $('#processShell').classList.add('ready');
  $('#processRange').disabled = false;
  $('#processPlay').disabled = false;
  $('#processScore').textContent = payload.score.toFixed(3).replace('.', ',');
  const marker = $('#maximumMarker');
  marker.style.left = `${(payload.maximum.x + .5) / 128 * 100}%`;
  marker.style.top = `${(payload.maximum.y + .5) / 128 * 100}%`;
  renderImagePipeline(0);
  setProcessPlaying(!matchMedia('(prefers-reduced-motion: reduce)').matches);
}

function clearImagePipeline() {
  processPayload = null;
  clearInterval(processTimer);
  processPlaying = false;
  const shell = $('#processShell');
  if (!shell) return;
  shell.classList.remove('ready');
  $('#processRange').disabled = true;
  $('#processPlay').disabled = true;
  $('#processScore').textContent = '—';
}

$$('[data-process-stage]').forEach((button,index) => button.addEventListener('click', () => {
  if (!processPayload) return;
  renderImagePipeline(index);
  setProcessPlaying(false);
}));
$('#processRange').addEventListener('input', event => {
  renderImagePipeline(Number(event.target.value));
  setProcessPlaying(false);
});
$('#processPlay').addEventListener('click', () => setProcessPlaying(!processPlaying));

$('#referenceCase').addEventListener('change', event => {
  if (event.target.value === '') return;
  const id = Number(event.target.value);
  const name = event.target.selectedOptions[0].textContent;
  $$('.sample').forEach(item => item.classList.remove('active'));
  selectSource(`/api/case-image?id=${id}`, name);
  state.caseId = id;
  $('#referenceCase').value = String(id);
});
fetch('/api/cases').then(response => {
  if (!response.ok) throw new Error('No se pudieron cargar los casos');
  return response.json();
}).then(payload => {
  if (payload.evaluation) {
    const test = payload.evaluation.splits.test;
    $('#localizationAggregate').textContent = `Test ${payload.evaluation.max_images_per_class == null ? "completo" : "reducido"} (${test.images} cortes): Dice micro ${test.micro_dice.toFixed(3)} | IoU micro ${test.micro_iou.toFixed(3)} | Sensibilidad ${test.sensitivity.toFixed(3)}. Medias sobre cortes con lesion: Dice ${test.mean_dice_lesion.toFixed(3)}, IoU ${test.mean_iou_lesion.toFixed(3)}.`;
  }
  if (payload.evaluation?.splits.test.anomalous_empty_reference_images) {
    $('#localizationAggregate').textContent += ` Hay ${payload.evaluation.splits.test.anomalous_empty_reference_images} cortes anomalos con referencia vacia a esta resolucion; no entran en las medias de cortes con lesion visible.`;
  }
  const examples = [
    ...payload.cases.filter(item => item.name.startsWith('Ungood')).slice(0, 3),
    ...payload.cases.filter(item => item.name.startsWith('good')).slice(0, 3)
  ];
  $$('.sample').forEach((button, index) => {
    const item = examples[index];
    if (!item) { button.hidden = true; return; }
    button.dataset.caseId = String(item.id);
    button.dataset.src = `/api/case-image?id=${item.id}`;
    button.dataset.name = item.name;
    button.querySelector('img').src = button.dataset.src;
    button.querySelector('img').alt = item.name;
    button.querySelector('span').textContent = item.name.startsWith('Ungood') ? 'Anómalo' : 'Normal';
    button.title = item.name;
  });
  if (state.version === 0 && examples.length) $('.sample').click();
  for (const item of payload.cases) {
    const option = document.createElement('option');
    option.value = item.id; option.textContent = item.name;
    $('#referenceCase').append(option);
  }
  if (state.caseId !== null) $('#referenceCase').value = String(state.caseId);
  $('#localizationStatus').textContent = payload.calibrated
    ? 'Selecciona un caso y pulsa Ejecutar modelo.'
    : 'Falta calibración: ejecuta python -m tfm_ae.localization y reinicia el servidor.';
}).catch(() => { $('#localizationStatus').textContent = 'Inicia la demo con python -m tfm_ae.demo_server para cargar las referencias.'; });
