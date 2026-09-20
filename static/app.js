const form = document.querySelector('#search-form');
const input = document.querySelector('#query');
const topK = document.querySelector('#top-k');
const status = document.querySelector('#status');
const results = document.querySelector('#results');
const submit = document.querySelector('#search-button');
const template = document.querySelector('#result-template');

function clearResults() { results.replaceChildren(); }
function setStatus(message, className = '') { status.textContent = message; status.className = className; }
function render(items) {
  clearResults();
  if (!items.length) { const p = document.createElement('p'); p.textContent = 'No matching songs were returned.'; p.className = 'empty'; results.append(p); return; }
  for (const item of items) {
    const node = template.content.cloneNode(true);
    node.querySelector('.rank').textContent = `#${item.rank}`;
    node.querySelector('h2').textContent = item.song;
    node.querySelector('.artist').textContent = item.artist;
    node.querySelector('.score').textContent = `Similarity: ${Number(item.similarity_score).toFixed(3)}`;
    const excerpt = node.querySelector('.excerpt'); excerpt.textContent = item.excerpt || '';
    const source = node.querySelector('.source');
    if (item.safe_source_url) source.href = item.safe_source_url; else source.remove();
    results.append(node);
  }
}
async function performSearch(query = input.value) {
  const text = query.trim(); if (!text) { setStatus('Enter a mood, theme, or atmosphere.', 'error'); return; }
  input.value = text; submit.disabled = true; setStatus('Searching lyrics…'); clearResults();
  try {
    const response = await fetch('/search', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({query:text, top_k:Number(topK.value)}) });
    const body = await response.json(); if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Search request failed.');
    setStatus(`${body.results.length} results`); render(body.results);
  } catch (error) { setStatus(error.message || 'Search failed.', 'error'); } finally { submit.disabled = false; }
}
form.addEventListener('submit', event => { event.preventDefault(); performSearch(); });
document.querySelectorAll('[data-query]').forEach(button => button.addEventListener('click', () => performSearch(button.dataset.query)));
