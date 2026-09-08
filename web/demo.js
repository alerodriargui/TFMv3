const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];

const state = { blob: null, name: 'Corte 01', objectUrl: null };
const inputImage = $('#inputImage');
const compareBefore = $('#compareBefore');
const runButton = $('#runButton');
const errorBanner = $('#errorBanner');

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.classList.add('visible');
}

function clearResult() {
  errorBanner.classList.remove('visible');
  $('.result-stage').classList.remove('ready');
  $('.heatmap-wrap').classList.remove('ready');
  $('#scoreValue').textContent = '—';
  $('#gaugeFill').style.width = '0';
  $('#verdict').className = 'verdict idle';
  $('#verdict').innerHTML = '<i></i><div><b>Sin analizar</b><span>Ejecuta el modelo para obtener una lectura.</span></div>';
}

function selectSource(src, name, blob = null) {
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
  clearResult();
  runButton.classList.add('loading');
  runButton.querySelector('span').textContent = 'Reconstruyendo';
  $('.result-stage').classList.add('processing');
  try {
    const blob = await sourceBlob();
    const response = await fetch('/api/reconstruct', {
      method: 'POST',
      headers: { 'Content-Type': blob.type || 'application/octet-stream' },
      body: blob
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || 'El servidor no pudo ejecutar la inferencia.');

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
      ? '<i></i><div><b>Patrón atípico</b><span>El residuo supera el umbral calibrado.</span></div>'
      : '<i></i><div><b>Compatible con normalidad</b><span>El residuo queda por debajo del umbral.</span></div>';
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
