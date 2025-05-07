// File: static/app.js

$(async function () {
  // Cache selectors
  const $modelsContainer = $('#model-checkboxes');
  const $docInputs       = $('#docInputs');
  const $numRandom       = $('#numRandom');
  const $splitSelect     = $('#splitSelect');
  const $summariseBtn    = $('#summariseBtn');
  const $resultsRow      = $('#results-row');

  // 1) Fetch and render model checkboxes
  const models = await fetch('/api/models').then(r => r.json());
  models.forEach(name => {
    const id = `model-${name}`;
    $modelsContainer.append(`
      <div class="form-check mr-3">
        <input class="form-check-input" type="checkbox" value="${name}" id="${id}">
        <label class="form-check-label" for="${id}">${name}</label>
      </div>
    `);
  });

  // 2) Doc‐entry add/remove
  $docInputs
    .on('click', '.add-doc', function () {
      const $row = $(this).closest('.doc-entry');
      const $clone = $row.clone();
      $clone.find('.doc-text').val('');
      $clone.find('.add-doc')
        .removeClass('btn-success add-doc')
        .addClass('btn-danger remove-doc')
        .text('−');
      $row.after($clone);
    })
    .on('click', '.remove-doc', function () {
      $(this).closest('.doc-entry').remove();
    });

  // 3) Summarise button
  $summariseBtn.click(async () => {
    // a) Clear previous results
    $resultsRow.empty();

    // b) Insert a heading
    const $heading = $(`
      <div class="col-12 mb-2">
        <h4 class="text-primary">Generated Summaries</h4>
        <hr>
      </div>`);
    $resultsRow.append($heading);

    // c) Gather inputs
    const docs = $('.doc-text')
      .map((i,el) => $(el).val().trim())
      .get()
      .filter(Boolean);

    const numRand = parseInt($numRandom.val()) || null;
    const split   = $splitSelect.val();
    const selectedModels = models.filter(m => $(`#model-${m}`).is(':checked'));

    // d) Validate
    if (!selectedModels.length) {
      return alert('Please select at least one model.');
    }
    if (!docs.length && !numRand) {
      return alert('Enter docs or set a random count.');
    }

    // e) Build payload
    const payload = { model_names: selectedModels, split };
    if (docs.length) payload.input_docs = docs;
    else             payload.num_random  = numRand;

    // f) Fetch
    let json;
    try {
      const resp = await fetch('/api/summarise', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) throw await resp.json();
      json = await resp.json();
    } catch (err) {
      console.error(err);
      return alert(err.detail || 'Error generating summaries');
    }

    const { results, references } = json;

    // g) If random, show references once (full width)
    if (references) {
      const $refCol  = $('<div class="col-12 mb-3"></div>');
      const $refCard = $(`
        <div class="card card-outline card-warning">
          <div class="card-header">Reference Summaries</div>
          <div class="card-body"></div>
        </div>`);
      references.forEach((r,i) => {
        $refCard.find('.card-body').append(`
          <h5>Example ${i+1}</h5>
          <p>${r}</p>
          ${i < references.length-1 ? '<hr>' : ''}
        `);
      });
      $refCol.append($refCard);
      $resultsRow.append($refCol);
    }

    // h) Determine column size per model
    const mCount = selectedModels.length;
    let colClass = 'col-12';
    if (mCount === 2)      colClass = 'col-12 col-md-6';
    else if (mCount >= 3)  colClass = 'col-12 col-md-4';

    // i) Render each model’s summaries side by side
    selectedModels.forEach(model => {
      const sums = results[model] || [];
      const $col  = $(`<div class="${colClass} mb-3"></div>`);
      const $card = $(`
        <div class="card card-outline card-info h-100">
          <div class="card-header">
            <h5 class="card-title mb-0">${model}</h5>
          </div>
          <div class="card-body p-3"></div>
        </div>`);
      sums.forEach((s,i) => {
        $card.find('.card-body').append(`
          <h6>Example ${i+1}</h6>
          <p>${s}</p>
          ${i < sums.length-1 ? '<hr>' : ''}
        `);
      });
      $col.append($card);
      $resultsRow.append($col);
    });
  }); // end click handler

}); // end IIFE
